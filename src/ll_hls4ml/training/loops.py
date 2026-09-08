"""Training and validation loops."""

from __future__ import annotations

import copy
import csv
import gc
import json
import os
import time
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from ll_hls4ml.data.dataset import HeteroGraphDataset
from ll_hls4ml.training.targets import (
    apply_hurdle_prediction,
    compute_target_z_stats,
    normalize_target,
    wahls4ml_metrics,
)
from ll_hls4ml.training.loaders import make_loader
from ll_hls4ml.training.distributed import (
    is_distributed,
    is_main_process,
    unwrap_model,
    wrap_ddp,
)
from ll_hls4ml.data.splits import random_train_val_test_split
from ll_hls4ml.io.schema import LABEL_KEYS


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "float32":
        return nullcontext()
    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16}
    if precision not in dtypes:
        raise ValueError("precision must be 'float32', 'fp16', or 'bf16'")
    return torch.autocast(device_type="cuda", dtype=dtypes[precision])


def _use_progress_bar() -> bool:
    """Use tqdm only on an interactive terminal; log files get one-line epoch summaries."""
    if not is_main_process():
        return False
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return False
    if os.environ.get("LL_HLS4ML_TQDM", "1").lower() in {"0", "false", "no"}:
        return False
    return True


def _persist_on_cpu(value):
    """Recursively copy tensors to CPU numpy for long-lived training results."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {k: _persist_on_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_persist_on_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_persist_on_cpu(v) for v in value)
    if isinstance(value, np.ndarray):
        return np.copy(value)
    return value


def _json_converter(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_converter(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_converter(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_json_converter(v) for v in obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return str(obj)


def _release_cuda_memory(*objects: object) -> None:
    """Move modules off GPU, collect, and return cached memory to the driver."""
    for obj in objects:
        if isinstance(obj, torch.nn.Module):
            unwrap_model(obj).cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _history_validation_metrics(metrics: dict) -> dict[str, float]:
    """Flatten validation vectors while retaining useful target-group macros."""

    record: dict[str, float] = {}
    for name, value in metrics.items():
        if isinstance(value, torch.Tensor):
            values = value.detach().float().cpu().flatten()
            record[f"val_{name}"] = float(values.mean())
            for index, label in enumerate(LABEL_KEYS):
                if index < values.numel():
                    record[f"val_{name}_{label}"] = float(values[index])
            if values.numel() >= len(LABEL_KEYS):
                record[f"val_resource_{name}"] = float(values[:4].mean())
                record[f"val_timing_{name}"] = float(values[4:6].mean())
        else:
            record[f"val_{name}"] = float(value)
    return record


def _write_history(path: str | Path, history: list[dict]) -> None:
    """Atomically refresh a monitorable per-epoch CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        fieldnames = list(
            dict.fromkeys(key for row in history for key in row)
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


def _optimizer_for_model(template: torch.optim.Optimizer, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Fresh optimizer for ``model`` with the same hyperparameters as ``template``."""
    cls = type(template)
    params = unwrap_model(model).parameters()
    if len(template.param_groups) == 1:
        kwargs = {k: v for k, v in template.param_groups[0].items() if k != "params"}
        return cls(params, **kwargs)
    raise ValueError("Leave-family-out training currently requires one optimizer group")


def _save_checkpoint(
    epoch,
    model,
    optimizer,
    scheduler,
    path,
    training_state: dict | None = None,
    grad_scaler=None,
):
    checkpoint = {
        "epoch": epoch,
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "grad_scaler": grad_scaler.state_dict() if grad_scaler is not None else None,
        # Epoch-boundary RNG state makes a single-GPU Colab/Kaggle restart
        # continue the same shuffle/dropout trajectory instead of replaying it.
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    if training_state is not None:
        checkpoint["training_state"] = training_state
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    pbar=None,
    distributed: bool = False,
    precision: str = "float32",
    return_profile: bool = False,
    gradient_clip_norm: float | None = 1.0,
    grad_scaler=None,
):
    model.train()
    base_model = unwrap_model(model)
    loss_sum = torch.zeros((), device=device)
    num_samples = 0
    num_targets = len(base_model.y_means)
    data_wait_seconds = 0.0
    gradient_norm_sum = 0.0
    optimizer_steps = 0
    step_started = time.perf_counter()
    iterator = iter(train_loader)

    while True:
        data_started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        data_wait_seconds += time.perf_counter() - data_started
        batch = batch.to(device, non_blocking=device.type == "cuda")
        target = normalize_target(
            batch.y.view(-1, num_targets),
            base_model.y_means,
            base_model.y_stds,
            getattr(base_model, "target_log_shift", None),
        )

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, precision):
            pred = model(batch)
            loss = criterion(pred, target)
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
        else:
            loss.backward()
        if gradient_clip_norm is not None:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip_norm
            )
            gradient_norm_sum += float(gradient_norm.detach())
        if grad_scaler is not None:
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            optimizer.step()
        optimizer_steps += 1

        n = batch.num_graphs
        loss_sum += loss.detach() * n
        num_samples += n

        if pbar is not None:
            pbar.update(1)

    if distributed and dist.is_initialized():
        stats = torch.stack(
            [loss_sum, loss_sum.new_tensor(float(num_samples))]
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        loss_sum, num_samples = stats[0], int(stats[1].item())

    mean_loss = float((loss_sum / max(num_samples, 1)).item())
    if not return_profile:
        return mean_loss
    wall_seconds = time.perf_counter() - step_started
    return mean_loss, {
        "data_wait_seconds": data_wait_seconds,
        "compute_submission_seconds": wall_seconds - data_wait_seconds,
        "data_wait_fraction": data_wait_seconds / max(wall_seconds, 1e-12),
        "mean_pre_clip_gradient_norm": (
            gradient_norm_sum / max(optimizer_steps, 1)
            if gradient_clip_norm is not None
            else 0.0
        ),
    }


def validate_one_epoch(
    model,
    val_loader,
    criterion,
    device,
    pbar=None,
    distributed: bool = False,
    precision: str = "float32",
):
    if distributed and not is_main_process():
        return 0.0, None, None, None, None

    base_model = unwrap_model(model)
    evaluation_model = base_model if distributed else model
    evaluation_model.eval()
    all_preds, all_targets = [], []
    loss_sum = torch.zeros((), device=device)
    num_samples = 0
    num_targets = len(base_model.y_means)

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            target = normalize_target(
                batch.y.view(-1, num_targets),
                base_model.y_means,
                base_model.y_stds,
                getattr(base_model, "target_log_shift", None),
            )

            with _autocast(device, precision):
                pred = evaluation_model(batch)
                loss = criterion(pred, target)

            all_preds.append(
                apply_hurdle_prediction(
                    pred,
                    base_model.y_means,
                    base_model.y_stds,
                    mode=getattr(
                        base_model,
                        "hurdle_prediction_mode",
                        "expected",
                    ),
                )
            )
            all_targets.append(target)

            n = batch.num_graphs
            loss_sum += loss.detach() * n
            num_samples += n

            if pbar is not None:
                pbar.update(1)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    # print(model.y_means, model.y_stds, "\n", preds, "\n", targets)
    # print(preds.std(dim=0))

    metrics = wahls4ml_metrics(
        preds,
        targets,
        base_model.y_means,
        base_model.y_stds,
        getattr(base_model, "target_log_shift", None),
    )
    metrics["loss"] = float((loss_sum / max(num_samples, 1)).item())
    return metrics


def fit(
    model,
    train_loader,
    val_loader,
    epochs,
    criterion,
    optimizer,
    scheduler,
    device,
    patience=50,
    mode="min",
    restore_best_weights=True,
    writer=None,
    verbose=2,
    experiment_name="",
    checkpoint_dir: str | Path | None = None,
    resume_from_backup: str | Path | None = None,
    distributed: bool = False,
    early_stopping_metric: str = "loss",
    checkpoint_interval: int = 5,
    precision: str = "float32",
    max_training_seconds: float | None = None,
    history_path: str | Path | None = None,
    gradient_clip_norm: float | None = 1.0,
    lr_scheduler_patience: int | None = 8,
    lr_scheduler_factor: float = 0.5,
    min_learning_rate: float = 1e-6,
):
    """
    Train model with optional early stopping.

    Checkpoints are saved under ``checkpoint_dir`` using ``experiment_name``.
    """
    distributed = distributed or is_distributed()
    main = is_main_process()

    checkpoint_dir = Path(checkpoint_dir or "artifacts/checkpoints")
    if main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"{experiment_name}_checkpoint.pt"
    backup_path = checkpoint_dir / f"{experiment_name}_backup.pt"

    model.to(device)
    if isinstance(criterion, torch.nn.Module):
        criterion.to(device)

    if gradient_clip_norm is not None and gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive or None")
    if scheduler is None and lr_scheduler_patience is not None:
        if lr_scheduler_patience < 0:
            raise ValueError("lr_scheduler_patience must be non-negative or None")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
            min_lr=min_learning_rate,
        )

    fit_started = time.perf_counter()
    prior_training_seconds = 0.0
    resumed_from_epoch = 0
    best_wall_seconds = None
    stop_reason = None
    history = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    grad_scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and precision == "fp16")
    )
    if patience > 0:
        patience_counter = 0
        best_metric = float("-inf") if mode == "max" else float("inf")
        best_epoch = 0

    if resume_from_backup:
        checkpoint = torch.load(resume_from_backup, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint["scheduler"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("grad_scaler") is not None:
            grad_scaler.load_state_dict(checkpoint["grad_scaler"])
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if (
            device.type == "cuda"
            and not distributed
            and checkpoint.get("cuda_rng_state_all") is not None
        ):
            states = [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
            torch.cuda.set_rng_state_all(states)
        start_epoch = checkpoint["epoch"] + 1
        resumed_from_epoch = int(checkpoint["epoch"])
        training_state = checkpoint.get("training_state", {})
        prior_training_seconds = float(
            training_state.get("cumulative_training_seconds", 0.0)
        )
        best_wall_seconds = training_state.get("best_wall_seconds")
        history = list(training_state.get("history", []))
        if patience > 0:
            patience_counter = int(
                training_state.get("patience_counter", 0)
            )
            best_metric = float(
                training_state.get("best_metric", best_metric)
            )
            best_epoch = int(training_state.get("best_epoch", 0))
    else:
        start_epoch = 1

    if distributed:
        model = wrap_ddp(model, device)

    train_sampler = (
        train_loader.sampler
        if isinstance(train_loader.sampler, DistributedSampler)
        else None
    )

    if main:
        print(f"Training {epochs} epochs...")

    should_stop = torch.zeros(1, device=device)

    for epoch in range(start_epoch, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        log_epoch = main and (epoch == 1 or (verbose > 0 and epoch % verbose == 0))
        pbar = None
        if _use_progress_bar():
            pbar = tqdm(
                total=len(train_loader) + len(val_loader),
                desc=f"Epoch {epoch:3d}/{epochs}",
                unit="batch",
                leave=log_epoch,
            )

        _synchronize(device)
        epoch_started = time.perf_counter()
        train_started = epoch_started
        train_result = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            pbar=pbar,
            distributed=distributed,
            precision=precision,
            return_profile=True,
            gradient_clip_norm=gradient_clip_norm,
            grad_scaler=grad_scaler if grad_scaler.is_enabled() else None,
        )
        if isinstance(train_result, tuple):
            train_loss, train_profile = train_result
        else:  # Keep simple mocked/custom loops compatible.
            train_loss, train_profile = train_result, {}
        _synchronize(device)
        train_seconds = time.perf_counter() - train_started
        validation_started = time.perf_counter()
        val_metrics = validate_one_epoch(
            model, val_loader, criterion, device,
            pbar=pbar, distributed=distributed, precision=precision,
        )
        if main:
            scheduler_metric = val_metrics[early_stopping_metric]
            if isinstance(scheduler_metric, torch.Tensor):
                scheduler_metric = scheduler_metric.mean().item()
            scheduler_metric = float(scheduler_metric)
        else:
            scheduler_metric = 0.0
        if distributed:
            scheduler_metric_tensor = torch.tensor(
                scheduler_metric, device=device, dtype=torch.float64
            )
            dist.broadcast(scheduler_metric_tensor, src=0)
            scheduler_metric = float(scheduler_metric_tensor.item())
        learning_rate = float(optimizer.param_groups[0]["lr"])
        if scheduler is not None:
            if isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step(scheduler_metric)
            else:
                scheduler.step()
        next_learning_rate = float(optimizer.param_groups[0]["lr"])
        _synchronize(device)
        validation_seconds = time.perf_counter() - validation_started
        epoch_seconds = time.perf_counter() - epoch_started
        cumulative_seconds = (
            prior_training_seconds + time.perf_counter() - fit_started
        )
        if main:
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    **_history_validation_metrics(val_metrics),
                    "train_seconds": train_seconds,
                    "validation_seconds": validation_seconds,
                    "epoch_seconds": epoch_seconds,
                    "cumulative_training_seconds": cumulative_seconds,
                    "learning_rate": learning_rate,
                    "next_learning_rate": next_learning_rate,
                    "train_seconds_per_sample": train_seconds
                    / max(len(train_loader.dataset), 1),
                    **train_profile,
                    "peak_gpu_memory_mb": (
                        torch.cuda.max_memory_allocated(device) / (1024**2)
                        if device.type == "cuda"
                        else 0.0
                    ),
                }
            )
            if history_path is not None:
                _write_history(history_path, history)

        if pbar is not None:
            pbar.set_postfix({
                "train": f"{train_loss:.4f}",
                "val": f"{val_metrics['loss']:.4f}"
            })
            pbar.close()
        elif log_epoch:
            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"train={train_loss:.4f}  val={val_metrics['loss']:.4f}  "
                f"lr={next_learning_rate:.2e}",
                flush=True,
            )

        if main and writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch) 
            writer.add_scalar("loss/val", val_metrics['loss'], epoch) 
            writer.add_scalar("learning_rate", next_learning_rate, epoch)
            for name, value in val_metrics.items(): 
                if name != 'loss':
                    for i, lbl in enumerate(LABEL_KEYS):
                        writer.add_scalar(f"{name}/{lbl}", value[i].item(), epoch)

        should_stop.zero_()

        if patience > 0 and main:
            current_metric = val_metrics[early_stopping_metric]
            if isinstance(current_metric, torch.Tensor):
                current_metric = current_metric.mean().item()
            is_improvement = (
                current_metric > best_metric if mode == "max" else current_metric < best_metric
            )

            if is_improvement:
                best_metric = current_metric
                best_epoch = epoch
                best_wall_seconds = cumulative_seconds
                patience_counter = 0
                _save_checkpoint(
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    ckpt_path,
                    training_state={
                        "history": history,
                        "best_metric": best_metric,
                        "best_epoch": best_epoch,
                        "patience_counter": patience_counter,
                        "cumulative_training_seconds": (
                            prior_training_seconds
                            + time.perf_counter()
                            - fit_started
                        ),
                        "best_wall_seconds": best_wall_seconds,
                        "peak_gpu_memory_mb": history[-1][
                            "peak_gpu_memory_mb"
                        ],
                    },
                    grad_scaler=grad_scaler if grad_scaler.is_enabled() else None,
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    stop_reason = "early_stopping"
                    should_stop[0] = 1

        if main and checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
            _save_checkpoint(
                epoch,
                model,
                optimizer,
                scheduler,
                backup_path,
                training_state={
                    "history": history,
                    "best_metric": (
                        best_metric if patience > 0 else None
                    ),
                    "best_epoch": (
                        best_epoch if patience > 0 else epoch
                    ),
                    "patience_counter": (
                        patience_counter if patience > 0 else 0
                    ),
                    "cumulative_training_seconds": (
                        prior_training_seconds
                        + time.perf_counter()
                        - fit_started
                    ),
                    "best_wall_seconds": best_wall_seconds,
                    "peak_gpu_memory_mb": history[-1]["peak_gpu_memory_mb"],
                },
                grad_scaler=grad_scaler if grad_scaler.is_enabled() else None,
            )
            print(f"Model saved to {backup_path}")

        if (
            main
            and max_training_seconds is not None
            and cumulative_seconds >= max_training_seconds
        ):
            stop_reason = "wall_time_budget"
            should_stop[0] = 1
            print(
                f"Wall-time budget reached after epoch {epoch} "
                f"({cumulative_seconds:.1f}s)."
            )

        if distributed:
            dist.broadcast(should_stop, src=0)

        if patience > 0 and should_stop.item() == 1:
            break

    if restore_best_weights and patience > 0 and ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        unwrap_model(model).load_state_dict(checkpoint["model"])
        if main:
            print(
                f"Best model restored from epoch {best_epoch} "
                f"with validation {early_stopping_metric} "
                f"{best_metric:.4f}"
            )

    if patience == 0 and main:
        _save_checkpoint(
            epoch,
            model,
            optimizer,
            scheduler,
            ckpt_path,
            training_state={
                "history": history,
                "best_metric": None,
                "best_epoch": epoch,
                "patience_counter": 0,
                "cumulative_training_seconds": (
                    prior_training_seconds
                    + time.perf_counter()
                    - fit_started
                ),
                "best_wall_seconds": None,
                "peak_gpu_memory_mb": history[-1]["peak_gpu_memory_mb"],
            },
            grad_scaler=grad_scaler if grad_scaler.is_enabled() else None,
        )

    if distributed:
        dist.barrier()

    base_model = unwrap_model(model)
    if stop_reason is None:
        stop_reason = "epochs_complete"
    base_model.training_history = history
    base_model.best_epoch = best_epoch if patience > 0 else epoch
    base_model.best_metric = best_metric if patience > 0 else None
    base_model.cumulative_training_seconds = (
        prior_training_seconds + time.perf_counter() - fit_started
    )
    base_model.resumed_from_epoch = resumed_from_epoch
    base_model.validation_cadence_epochs = 1
    base_model.best_wall_seconds = best_wall_seconds
    base_model.peak_gpu_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else 0.0
    )
    base_model.stop_reason = stop_reason
    return model


def transductive_vs_inductive_fit(
    original_model,
    max_per_kernel_type: dict[str, int],
    tensor_dir: str | Path,
    epochs,
    batch_size,
    criterion,
    optimizer,
    device,
    patience=50,
    evaluation_metric="val_loss",
    mode="min",
    restore_best_weights=True,
    writer=None,
    verbose=10,
    experiment_name="",
    checkpoint_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
    distributed: bool = False,
):
    """
    Fits N models, one for each kernel type, where that kernel type is used as the inductive set,
    and the rest of the kernel types are used as the transductive set.
    Validation/early stopping is done on the transductive set's loss.

    Returns:
        A dictionary of metrics, keyed by inductive kernel type, containing:
            train_loss: Train loss
            val_loss: Validation loss
            r_per_target: Pearson correlation coefficient per target
            std_ratio_per_target: Ratio of predicted standard deviation to target standard deviation per target
            best_val_preds: Best validation predictions
            best_val_targets: Best validation targets
            best_val_loss: Best validation loss during training
            best_epoch: Best epoch during training
            test_loss: Test loss
            test_preds: Test predictions
            test_targets: Test targets
            y_means: Per-target means used for normalization (CPU numpy)
            y_stds: Per-target stds used for normalization (CPU numpy)

        All array/tensor fields are stored as CPU numpy arrays.
        Under DDP, only rank 0 returns the full dictionary; other ranks return {}.
    """
    distributed = distributed or is_distributed()
    main = is_main_process()

    checkpoint_dir = Path(checkpoint_dir or "artifacts/checkpoints")
    results_dir = Path(results_dir) if results_dir is not None else checkpoint_dir.parent
    if main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

    kernel_types = list(max_per_kernel_type.keys())
    device_type = device.type if isinstance(device, torch.device) else str(device)
    use_cuda = device_type.startswith("cuda") and torch.cuda.is_available()

    training_histories = {}
    for kernel_type in kernel_types:
        model = None
        loop_optimizer = None
        train_loader = val_loader = test_loader = None
        train_ds = val_ds = test_ds = train_val_ds = None
        y_means = y_stds = None

        try:
            if use_cuda:
                _release_cuda_memory()

            train_val_ds = HeteroGraphDataset(
                tensor_dir,
                types=[t for t in kernel_types if t != kernel_type],
                max_per_type={
                    k: v for k, v in max_per_kernel_type.items() if k != kernel_type
                },
            )
            test_ds = HeteroGraphDataset(
                tensor_dir,
                types=[kernel_type],
                max_per_type=max_per_kernel_type[kernel_type],
            )

            train_ds, val_ds, _unused_test = random_train_val_test_split(
                train_val_ds,
                val_fraction=0.2,
                test_fraction=0.0,
            )

            train_loader = make_loader(
                train_ds, batch_size=batch_size, shuffle=True, distributed=distributed,
            )
            val_loader = make_loader(
                val_ds, batch_size=batch_size, shuffle=False, distributed=False,
            )
            test_loader = make_loader(
                test_ds, batch_size=batch_size, shuffle=False, distributed=False,
            )

            if main:
                print(
                    f"*********** \"{kernel_type}\" as inductive set ***********\n"
                    f"  Train size: {len(train_ds)}, "
                    f"  Val size: {len(val_ds)}, "
                    f"  Test size: {len(test_ds)}"
                )

            model = copy.deepcopy(original_model).to(device)
            y_means, y_stds = compute_target_z_stats(train_ds)
            model.y_means.copy_(y_means)
            model.y_stds.copy_(y_stds)
            loop_optimizer = _optimizer_for_model(optimizer, model)
            model = fit(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                criterion=criterion,
                optimizer=loop_optimizer,
                scheduler=None,
                device=device,
                patience=patience,
                mode=mode,
                restore_best_weights=restore_best_weights,
                writer=writer,
                verbose=verbose,
                experiment_name=experiment_name + f"_{kernel_type}",
                checkpoint_dir=checkpoint_dir,
                distributed=distributed,
            )

            if main:
                print("Evaluating on test set...")
                test_metrics = validate_one_epoch(
                    model, test_loader, criterion, device
                )

                training_histories[kernel_type] = _persist_on_cpu(
                    {
                        "y_means": y_means,
                        "y_stds": y_stds,
                        "test_metrics": test_metrics,
                    }
                )

                th_path = results_dir / f"training_histories_{kernel_type}.json"
                with open(th_path, "w") as f:
                    json.dump(training_histories[kernel_type], f, default=_json_converter)

                print(f"Test loss: {test_metrics['loss']:.4f}")
        finally:
            _release_cuda_memory(
                model,
                loop_optimizer,
                train_loader,
                val_loader,
                test_loader,
                train_ds,
                val_ds,
                test_ds,
                train_val_ds,
                y_means,
                y_stds,
            )
        if main:
            print("***********************************************\n\n")

        if distributed:
            dist.barrier()

    return training_histories
