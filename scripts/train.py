#!/usr/bin/env python3
"""Config-driven training entry point for portable local/Colab/Kaggle runs."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import Subset, WeightedRandomSampler

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hls-surrogate-lab-matplotlib")

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from ll_hls4ml.data.dataset import HeteroGraphDataset
from ll_hls4ml.data.augmentations import resolve as resolve_transform
from ll_hls4ml.data.splits import (
    benchmark_train_val_test_split,
    limit_subset_archives,
    nested_group_train_subset,
    random_train_val_test_split,
    saved_manifest_split,
)
from ll_hls4ml.data.vocab import load_vocab
from ll_hls4ml.models.registry import build
from ll_hls4ml.reporting.accounting import (
    cohort_membership,
    format_cohort_table,
    format_hurdle_table,
    graph_structure_rows,
    hurdle_calibration_rows,
    hurdle_confusion_rows,
    macro_metric_rows,
    paired_delta_rows,
    read_prediction_rows,
    split_sha256,
    structural_error_rows,
)
from ll_hls4ml.training import compute_target_z_stats, make_loader
from ll_hls4ml.io.schema import LABEL_KEYS
from ll_hls4ml.training.distributed import (
    cleanup_ddp,
    is_main_process,
    setup_from_env,
    unwrap_model,
)
from ll_hls4ml.training.loops import _autocast, _json_converter, fit
from ll_hls4ml.training.telemetry import NvidiaSmiMonitor
from ll_hls4ml.training.targets import (
    apply_hurdle_prediction,
    LogHuberLoss,
    LogHuberHurdleLoss,
    denormalize_target,
    relative_percentage_error,
    wahls4ml_metrics_raw,
)
from ll_hls4ml.viz.training import (
    hurdle_calibration_figure,
    learning_curve_figure,
    prediction_scatter_plots,
    rpe_box_plots,
)


DISPLAY_LABELS = ["LUT", "FF", "DSP", "BRAM", "Cycles", "II"]
PAPER_ORDER = ["BRAM", "DSP", "FF", "LUT", "Cycles", "II"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_path(value: str, config_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_dir / path).resolve()


def _model_from_config(config: dict, vocab_size: int, max_pos: int, train_ds):
    target_log_shift = config.get("target_log_shift")
    y_means, y_stds = compute_target_z_stats(
        train_ds,
        log_shift=target_log_shift,
        unbiased=config.get("target_std_unbiased", True),
    )
    common = {
        "instruction_vocab_size": vocab_size,
        "y_means": y_means,
        "y_stds": y_stds,
        "hidden_dim": config.get("hidden_dim", 128),
        "num_layers": config.get("num_layers", 3),
        "dropout": config.get("dropout", 0.1),
        "use_global_features": config.get("use_global_features", False),
        "use_context": config.get("use_context", False),
        "split_heads": config.get("split_heads", False),
        "context_mode": config.get("context_mode", "core"),
        "hurdle_heads": config.get("hurdle_heads", False),
        "hurdle_prediction_mode": config.get(
            "hurdle_prediction_mode",
            "expected",
        ),
    }
    model_name = config.get("model", "hetero_gat")
    high_level_models = {
        "high_level_layer_gnn",
        "paper_high_level_gatv2",
        "paper_transformer",
    }
    hierarchical_models = {
        "hierarchical",
        "hierarchical_operator",
        "hierarchical_no_local_message",
        "hierarchical_no_block_cfg",
        "hierarchical_no_callee",
        "hierarchical_orderless",
        "hierarchical_topology_destroyed",
        "hierarchical_high_level_fusion",
        "hierarchical_sequence",
        "hierarchical_block_attention",
        "hierarchical_memory_dual",
    }
    if model_name not in (hierarchical_models | high_level_models):
        common["pool"] = config.get("pool", "mean")
    if model_name == "paper_high_level_gatv2":
        from ll_hls4ml.data.high_level import PROCESSED_FEATURE_DIM

        return build(
            model_name,
            input_dim=PROCESSED_FEATURE_DIM,
            y_means=y_means,
            y_stds=y_stds,
            hidden_dim=config.get("hidden_dim", 512),
            num_layers=config.get("num_layers", 5),
            heads=config.get("heads", 5),
            mlp_hidden_dim=config.get("mlp_hidden_dim", 512),
            dropout=config.get("dropout", 0.3),
            target_log_shift=target_log_shift,
        )
    if model_name == "paper_transformer":
        from ll_hls4ml.data.high_level import PROCESSED_FEATURE_DIM

        return build(
            model_name,
            y_means=y_means,
            y_stds=y_stds,
            feature_dim=config.get("feature_dim", PROCESSED_FEATURE_DIM),
            embed_dim=config.get("embed_dim", 512),
            num_heads=config.get("num_heads", 8),
            ff_dim=config.get("ff_dim", 512),
            num_layers=config.get("num_layers", 2),
            max_layers=config.get("max_layers", 51),
            dropout=config.get("dropout", 0.1),
            target_log_shift=target_log_shift,
        )
    if model_name == "high_level_layer_gnn":
        from ll_hls4ml.data.high_level import PROCESSED_FEATURE_DIM

        return build(
            model_name,
            input_dim=PROCESSED_FEATURE_DIM,
            y_means=y_means,
            y_stds=y_stds,
            hidden_dim=config.get("hidden_dim", 64),
            num_layers=config.get("num_layers", 3),
            heads=config.get("heads", 1),
            dropout=config.get("dropout", 0.15),
            encoder=config.get("high_level_encoder", "gatv2"),
            hurdle_heads=config.get("hurdle_heads", True),
            hurdle_prediction_mode=config.get(
                "hurdle_prediction_mode", "threshold"
            ),
        )
    if model_name in {
        "hetero_gat",
        "hetero_relational",
        "hierarchical",
        "hierarchical_operator",
        "hierarchical_no_local_message",
        "hierarchical_no_block_cfg",
        "hierarchical_no_callee",
        "hierarchical_orderless",
        "hierarchical_topology_destroyed",
        "hierarchical_high_level_fusion",
        "hierarchical_sequence",
        "hierarchical_block_attention",
        "hierarchical_memory_dual",
        "rgcn",
    }:
        common["edge_pos_vocab_size"] = max_pos
        if model_name == "hierarchical_no_local_message":
            common["use_local_messages"] = False
        if model_name not in hierarchical_models:
            common["aggr"] = config.get("aggr", "sum")
        if model_name == "hetero_relational":
            common["message_aggr"] = config.get("message_aggr", "mean")
        elif model_name not in hierarchical_models:
            common["heads"] = config.get("heads", 1)
        elif model_name == "hierarchical_high_level_fusion":
            from ll_hls4ml.data.high_level import PROCESSED_FEATURE_DIM

            common["heads"] = config.get("heads", 1)
            common["high_level_input_dim"] = PROCESSED_FEATURE_DIM
            common["high_level_encoder"] = config.get(
                "high_level_encoder", "gatv2"
            )
        else:
            common["instruction_num_layers"] = config.get(
                "instruction_num_layers"
            )
            common["block_num_layers"] = config.get("block_num_layers")
            if model_name == "hierarchical_operator":
                common["operator_profile"] = config.get("operator_profile")
            if model_name in {
                "hierarchical_sequence",
                "hierarchical_block_attention",
                "hierarchical_memory_dual",
            }:
                common.update(
                    {
                        "attention_heads": config.get("attention_heads", 4),
                        "attention_layers": config.get("attention_layers", 2),
                        "cfg_recurrent_steps": config.get(
                            "cfg_recurrent_steps", 8
                        ),
                        "sequence_token_budget": config.get(
                            "sequence_token_budget", 16_384
                        ),
                        "attention_pair_budget": config.get(
                            "attention_pair_budget", 131_072
                        ),
                    }
                )
    elif model_name in {"mlp", "pooled_control"}:
        common["num_var_embed_layers"] = config.get("num_var_embed_layers", 2)
        common["node_aggr"] = config.get("node_aggr", "concat")
    return build(model_name, **common)


def _git_state(repository: Path) -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    return {"commit": commit, "dirty": dirty}


def _split_manifest(dataset, splits: dict[str, object], tensor_dir: Path) -> dict:
    manifest = {}
    for split_name, subset in splits.items():
        manifest[split_name] = [
            {
                "dataset_index": int(index),
                "kernel_family": dataset.type_of(index),
                "tensor_path": str(dataset.paths[index].relative_to(tensor_dir)),
            }
            for index in subset.indices
        ]
    return manifest


def _predict(model, loader, device):
    base_model = unwrap_model(model)
    base_model.eval()
    predictions = []
    targets = []
    presence_probabilities = []
    structures = None
    elapsed = 0.0
    with torch.no_grad():
        for batch in loader:
            if hasattr(batch, "node_types") and "instruction" in batch.node_types:
                if structures is None:
                    structures = []
                structures.extend(graph_structure_rows(batch))
            batch = batch.to(device, non_blocking=device.type == "cuda")
            precision = getattr(base_model, "inference_precision", "float32")
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            with _autocast(device, precision):
                raw_prediction = base_model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - started
            if raw_prediction.shape[-1] == len(LABEL_KEYS) + 2:
                presence_probabilities.append(
                    raw_prediction[:, len(LABEL_KEYS):].sigmoid().float().cpu()
                )
            normalized_prediction = apply_hurdle_prediction(
                raw_prediction,
                base_model.y_means,
                base_model.y_stds,
                mode=getattr(
                    base_model,
                    "hurdle_prediction_mode",
                    "expected",
                ),
            )
            prediction = denormalize_target(
                normalized_prediction,
                base_model.y_means,
                base_model.y_stds,
                getattr(base_model, "target_log_shift", None),
            )
            predictions.append(prediction.cpu())
            targets.append(batch.y.view(-1, len(LABEL_KEYS)).cpu())
    prediction_array = torch.cat(predictions).numpy()
    target_array = torch.cat(targets).numpy()
    return (
        np.clip(prediction_array, 0, None),
        target_array,
        (
            torch.cat(presence_probabilities).numpy()
            if presence_probabilities
            else None
        ),
        structures,
        elapsed / len(target_array),
    )


def _metric_rows(
    split_name: str,
    predictions: np.ndarray,
    targets: np.ndarray,
    inference_seconds_per_sample: float,
    kernel_family: str = "all",
) -> list[dict]:
    metrics = wahls4ml_metrics_raw(
        torch.from_numpy(np.array(predictions, copy=True)),
        torch.from_numpy(np.array(targets, copy=True)),
    )
    rpe = relative_percentage_error(
        torch.from_numpy(np.array(predictions, copy=True)),
        torch.from_numpy(np.array(targets, copy=True)),
    ).numpy()
    rows = []
    for index, target_name in enumerate(LABEL_KEYS):
        rows.append(
            {
                "split": split_name,
                "kernel_family": kernel_family,
                "target": target_name,
                "r2": float(metrics["r2"][index]),
                "smape": float(metrics["smape"][index]),
                "rmse": float(metrics["rmse"][index]),
                "rpe_median": float(np.median(rpe[:, index])),
                "rpe_mean": float(np.mean(rpe[:, index])),
                "rpe_q25": float(np.percentile(rpe[:, index], 25)),
                "rpe_q75": float(np.percentile(rpe[:, index], 75)),
                "inference_seconds_per_sample": inference_seconds_per_sample,
                "n_samples": len(targets),
            }
        )
    return rows


def _prediction_rows(
    split_name: str,
    manifest: list[dict],
    predictions: np.ndarray,
    targets: np.ndarray,
    presence_probabilities: np.ndarray | None = None,
    structures: list[dict] | None = None,
) -> list[dict]:
    rows = []
    for row_index, sample in enumerate(manifest):
        row = {"split": split_name, **sample}
        if structures is not None:
            row.update(structures[row_index])
        for target_index, target_name in enumerate(LABEL_KEYS):
            row[f"target_{target_name}"] = float(targets[row_index, target_index])
            row[f"prediction_{target_name}"] = float(
                predictions[row_index, target_index]
            )
        if presence_probabilities is not None:
            row["presence_probability_dsp"] = float(
                presence_probabilities[row_index, 0]
            )
            row["presence_probability_bram"] = float(
                presence_probabilities[row_index, 1]
            )
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_plots(
    run_dir: Path,
    experiment_name: str,
    split_name: str,
    predictions: np.ndarray,
    targets: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    title = f"{experiment_name}: {split_name}"
    figure, _ = rpe_box_plots(
        predictions,
        targets,
        DISPLAY_LABELS,
        ordering=PAPER_ORDER,
        title=title,
        show=False,
    )
    figure.savefig(
        figures_dir / f"{split_name}__rpe.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)
    figure, _ = prediction_scatter_plots(
        predictions,
        targets,
        DISPLAY_LABELS,
        title=title,
        show=False,
    )
    figure.savefig(
        figures_dir / f"{split_name}__scatter.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _save_training_plots(run_dir: Path, experiment_name: str, history: list[dict]):
    if not history:
        return
    import matplotlib.pyplot as plt

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for key, filename, suffix in (
        ("epoch", "training_per_epoch.png", "per epoch"),
        (
            "cumulative_training_seconds",
            "training_per_wall_time.png",
            "per wall time",
        ),
    ):
        figure, _axes = learning_curve_figure(
            history, key, f"{experiment_name}: {suffix}"
        )
        figure.savefig(figures_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(figure)


def _save_calibration_plots(run_dir: Path, rows: list[dict]):
    if not rows:
        return
    import matplotlib.pyplot as plt

    figures_dir = run_dir / "figures"
    for split in sorted({row["split"] for row in rows}):
        figure, _axes = hurdle_calibration_figure(rows, split)
        figure.savefig(
            figures_dir / f"{split}__hurdle_calibration.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)


def _write_report(
    run_dir: Path,
    experiment_name: str,
    resolved_config: dict,
    sizes: dict[str, int],
    metric_rows: list[dict],
    accounting: dict | None = None,
) -> None:
    accounting = accounting or {}
    cumulative_seconds = resolved_config.get("cumulative_training_seconds")
    cumulative_text = (
        f"{float(cumulative_seconds):.1f} seconds"
        if cumulative_seconds is not None
        else "not recoverable for this legacy resumed run"
    )
    gpu_telemetry = resolved_config.get("gpu_telemetry") or {}
    gpu_telemetry_text = (
        f"{gpu_telemetry.get('gpu_utilization_mean_percent', 0):.1f}% mean, "
        f"{gpu_telemetry.get('gpu_utilization_p10_percent', 0):.1f}/"
        f"{gpu_telemetry.get('gpu_utilization_p50_percent', 0):.1f}/"
        f"{gpu_telemetry.get('gpu_utilization_p90_percent', 0):.1f}% p10/p50/p90, "
        f"{gpu_telemetry.get('gpu_zero_utilization_fraction', 0):.1%} zero samples, "
        f"{gpu_telemetry.get('gpu_longest_below_25_seconds', 0):.1f}s longest low-utilization streak"
        if gpu_telemetry
        else "not sampled"
    )
    system_telemetry_text = (
        f"{gpu_telemetry.get('system_cpu_mean_percent', 0):.1f}% system CPU, "
        f"{gpu_telemetry.get('training_tree_cpu_mean_percent', 0):.1f}% training-tree CPU, "
        f"{gpu_telemetry.get('system_disk_read_mean_mb_s', 0):.2f} MiB/s disk reads"
        if gpu_telemetry.get("system_cpu_mean_percent") is not None
        else "not sampled"
    )
    evaluation_rows = [
        row
        for row in metric_rows
        if row["split"] in {"test", "exemplar"}
        and row["kernel_family"] == "all"
    ]
    table = [
        "| split | target | R² | SMAPE (%) | RMSE | median RPE (%) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in evaluation_rows:
        table.append(
            f"| {row['split']} | {row['target']} | {row['r2']:.3f} | "
            f"{row['smape']:.2f} | "
            f"{row['rmse']:.2f} | {row['rpe_median']:.2f} |"
        )
    macro_table = [
        "| split | scope | macro R² | macro SMAPE (%) | macro RMSE |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in macro_metric_rows(metric_rows):
        if row["kernel_family"] == "all":
            macro_table.append(
                f"| {row['split']} | {row['scope']} | {row['r2']:.3f} | "
                f"{row['smape']:.2f} | {row['rmse']:.2f} |"
            )
    family_table = [
        "| test family | samples | macro R² | macro SMAPE (%) |",
        "| --- | ---: | ---: | ---: |",
    ]
    test_families = sorted(
        {
            row["kernel_family"]
            for row in metric_rows
            if row["split"] == "test" and row["kernel_family"] != "all"
        }
    )
    for family in test_families:
        rows = [
            row
            for row in metric_rows
            if row["split"] == "test" and row["kernel_family"] == family
        ]
        family_table.append(
            f"| {family} | {rows[0]['n_samples']} | "
            f"{np.mean([row['r2'] for row in rows]):.3f} | "
            f"{np.mean([row['smape'] for row in rows]):.2f} |"
        )
    report = f"""# {experiment_name}

Single-model wa-hls4ml-style evaluation generated from persisted predictions.

- Model: `{resolved_config["model"]}`
- Device: `{resolved_config["device"]}`
- Tensor source revision: `{resolved_config.get("tensor_source_revision") or "not recorded"}`
- Seed: {resolved_config["seed"]}
- Current invocation wall time: {resolved_config["wall_seconds"]:.1f} seconds
- Cumulative training wall time: {cumulative_text}
- Wall time to best validation: {resolved_config.get("best_wall_seconds") or "not recorded"} seconds
- Parameters: {resolved_config.get("parameter_count", "not recorded")}
- Peak allocated GPU memory: {resolved_config.get("peak_gpu_memory_mb") or 0:.1f} MiB
- GPU utilization trace: {gpu_telemetry_text}
- Host utilization trace: {system_telemetry_text}
- Mean train seconds/sample/epoch: {resolved_config.get("mean_train_seconds_per_sample") or 0:.6f}
- Stop reason: `{resolved_config.get("stop_reason") or "not recorded"}`
- Validation cadence: every {resolved_config.get("validation_cadence_epochs", 1)} epoch(s)
- Checkpoint cadence: every {resolved_config.get("checkpoint_cadence_epochs", 5)} epoch(s)
- Split SHA-256: `{resolved_config.get("split_sha256", "not recorded")}`
- Split sizes: `{json.dumps(sizes, sort_keys=True)}`
- hls-surrogate-lab state: `{json.dumps(resolved_config["ll_hls4ml_git"])}`

## Evaluation metrics

{chr(10).join(macro_table)}

{chr(10).join(table)}

## Test metrics by kernel family

{chr(10).join(family_table)}

## Cohort membership

{format_cohort_table(accounting.get("cohort_membership", {}))}

## Hurdle confusion matrices

{format_hurdle_table(accounting.get("hurdle_confusion", []))}

Full per-family confusion matrices and per-archive membership are persisted in
`experiment_accounting.json` and `hurdle_confusion.csv`. Presence reliability
data and figures are in `hurdle_calibration.csv` and `figures/`.

## Efficiency and structural diagnostics

Learning curves are persisted in `learning_curves.csv` and plotted against both
epoch and cumulative training wall time. `macro_metrics.csv` separates resource
and timing quality; `structural_error_slices.csv` bins error by graph size, block
length, loop/SCC burden, call depth, and memory burden. If an H0 prediction path
was configured, exact per-sample paired deltas are in `paired_deltas_vs_h0.csv`.

Per-target test and exemplar metrics are in `metrics.csv`. Exact split
membership is in `split_manifest.json`, per-sample predictions are in
`predictions.csv`, and RPE/scatter figures are in `figures/`.

This is only directly comparable with wa-hls4ml when dataset membership,
compiler/graph provenance, targets, and evaluation splits are aligned.
"""
    (run_dir / "REPORT.md").write_text(report)


def main() -> None:
    run_started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Train an HLS surrogate model")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML or JSON training config",
    )
    parser.add_argument(
        "--evaluate-checkpoint",
        type=Path,
        default=None,
        help="Skip fitting and evaluate this checkpoint with the current config",
    )
    args = parser.parse_args()

    config_file = Path(args.config).resolve()
    with config_file.open() as handle:
        config = (
            json.load(handle)
            if config_file.suffix == ".json"
            else yaml.safe_load(handle)
        )
    config_dir = config_file.parent
    if config.get("worker_tmpdir"):
        os.environ["TMPDIR"] = str(config["worker_tmpdir"])
        tempfile.tempdir = None

    _rank, world_size, local_rank = setup_from_env()
    distributed = world_size > 1
    main_process = is_main_process()

    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if distributed else "cuda")
    else:
        device = torch.device("cpu")

    tensor_dir = _config_path(config["tensor_dir"], config_dir)
    vocab_path = _config_path(
        config.get("vocab_path", str(tensor_dir / "vocab.json")),
        config_dir,
    )
    vocab, max_pos, _counts = load_vocab(vocab_path)

    model_name = config.get("model", "hetero_gat")
    paper_model = model_name in {
        "high_level_layer_gnn",
        "paper_high_level_gatv2",
        "paper_transformer",
    }
    split_manifest_path = config.get("split_manifest_path")
    saved_manifest = None
    if split_manifest_path:
        split_manifest_path = _config_path(split_manifest_path, config_dir)
        saved_manifest = json.loads(split_manifest_path.read_text())
    if paper_model and saved_manifest is None:
        raise ValueError("Paper models require split_manifest_path")

    max_per_type = config.get("max_per_kernel_type")
    configured_kernel_types = config.get("kernel_types")
    if configured_kernel_types is None and isinstance(max_per_type, dict):
        configured_kernel_types = list(max_per_type)
    if configured_kernel_types is None:
        kernel_types = [
            path.name
            for path in sorted(tensor_dir.iterdir())
            if path.is_dir() and path.name != "exemplar"
        ]
    else:
        kernel_types = [
            kernel_type
            for kernel_type in configured_kernel_types
            if kernel_type != "exemplar"
        ]
    if not kernel_types:
        raise ValueError("No non-exemplar kernel types found in the tensor directory")
    main_relative_paths = None
    exemplar_relative_paths = None
    if saved_manifest is not None:
        main_relative_paths = [
            row["tensor_path"]
            for split in ("train", "validation", "test")
            for row in saved_manifest[split]
        ]
        exemplar_relative_paths = [
            row["tensor_path"] for row in saved_manifest["exemplar"]
        ]
    transform_name = config.get("graph_transform")
    if model_name == "hierarchical_topology_destroyed":
        transform_name = transform_name or "permute_destinations_within_function_v2"
    graph_transform = (
        resolve_transform(transform_name, seed=int(config.get("corruption_seed", seed)))
        if transform_name else None
    )
    dataset = HeteroGraphDataset(
        tensor_dir,
        types=kernel_types,
        max_per_type=max_per_type,
        silent=not main_process,
        relative_paths=main_relative_paths,
        transform=graph_transform,
    )
    exemplar_dataset = HeteroGraphDataset(
        tensor_dir,
        types=["exemplar"],
        max_per_type=None,
        silent=not main_process,
        relative_paths=exemplar_relative_paths,
        transform=graph_transform,
    )
    split_coverage = None
    if saved_manifest is not None:
        strict_manifest = config.get("require_complete_split_manifest", True)
        train_ds, val_ds, test_ds, split_coverage = saved_manifest_split(
            dataset,
            saved_manifest,
            tensor_dir,
            ("train", "validation", "test"),
            require_all=strict_manifest,
        )
        exemplar_ds, exemplar_coverage = saved_manifest_split(
            exemplar_dataset,
            saved_manifest,
            tensor_dir,
            ("exemplar",),
            require_all=strict_manifest,
        )
        split_coverage.update(exemplar_coverage)
    else:
        exemplar_ds = Subset(exemplar_dataset, range(len(exemplar_dataset)))
        split_fn = (
            random_train_val_test_split
            if config.get("split_strategy") == "random"
            else benchmark_train_val_test_split
        )
        train_ds, val_ds, test_ds = split_fn(
            dataset,
            val_fraction=config.get("val_fraction", 0.15),
            test_fraction=config.get("test_fraction", 0.15),
            seed=seed,
        )
    data_scale_manifest = None
    if config.get("train_scale") is not None:
        train_scale = float(config["train_scale"])
        baseline_archives = int(
            config.get("baseline_archives_per_family", 4)
        )
        evaluation_archives = int(
            config.get(
                "evaluation_archives_per_family",
                baseline_archives,
            )
        )
        subset_seed = int(config.get("train_subset_seed", seed))
        strict_archives = bool(config.get("strict_archive_counts", True))
        train_ds, train_scale_report = nested_group_train_subset(
            dataset,
            train_ds,
            train_scale,
            baseline_archives_per_family=baseline_archives,
            seed=subset_seed,
            strict=strict_archives,
        )
        val_ds, validation_archives = limit_subset_archives(
            dataset,
            val_ds,
            evaluation_archives,
            strict=strict_archives,
        )
        test_ds, test_archives = limit_subset_archives(
            dataset,
            test_ds,
            evaluation_archives,
            strict=strict_archives,
        )
        exemplar_ds, exemplar_archives = limit_subset_archives(
            exemplar_dataset,
            exemplar_ds,
            evaluation_archives,
            strict=strict_archives,
        )
        data_scale_manifest = {
            **train_scale_report,
            "evaluation_archives_per_family": evaluation_archives,
            "validation_archive_cohorts": validation_archives,
            "test_archive_cohorts": test_archives,
            "exemplar_archive_cohorts": exemplar_archives,
        }
    if not train_ds or not val_ds or not test_ds:
        raise ValueError(
            "Train/validation/test split is empty; increase the dataset or fractions"
        )
    if not exemplar_ds:
        raise ValueError("No exemplar tensors found in the tensor directory")

    if config.get("model") in {
        "hierarchical_high_level_fusion",
        "high_level_layer_gnn",
        "paper_high_level_gatv2",
        "paper_transformer",
    }:
        from ll_hls4ml.data.high_level import (
            CDFGHighLevelDataset,
            HighLevelLayerDataset,
            PaperTransformerDataset,
            feature_statistics,
            paper_feature_statistics,
        )

        cache_path = _config_path(config["high_level_cache"], config_dir)
        high_level_cache = torch.load(cache_path, weights_only=False)
        train_paths = [
            dataset.paths[index].relative_to(tensor_dir).as_posix()
            for index in train_ds.indices
        ]
        statistics_fn = (
            paper_feature_statistics
            if config.get("model") in {"paper_high_level_gatv2", "paper_transformer"}
            else feature_statistics
        )
        high_level_means, high_level_stds = statistics_fn(
            high_level_cache, train_paths
        )
        if config.get("model") in {
            "high_level_layer_gnn",
            "paper_high_level_gatv2",
            "paper_transformer",
        }:
            def high_level_split(source, subset):
                indices = list(subset.indices)
                paths = [
                    source.paths[index].relative_to(tensor_dir).as_posix()
                    for index in indices
                ]
                if config.get("model") == "paper_transformer":
                    split = PaperTransformerDataset(
                        high_level_cache, paths, high_level_means, high_level_stds,
                        max_layers=config.get("max_layers", 51),
                    )
                else:
                    split = HighLevelLayerDataset(
                        high_level_cache, paths, high_level_means, high_level_stds
                    )
                split.indices = indices
                return split

            train_ds = high_level_split(dataset, train_ds)
            val_ds = high_level_split(dataset, val_ds)
            test_ds = high_level_split(dataset, test_ds)
            exemplar_ds = high_level_split(exemplar_dataset, exemplar_ds)
        else:
            train_ds = CDFGHighLevelDataset(
                dataset, train_ds.indices, high_level_cache,
                high_level_means, high_level_stds,
            )
            val_ds = CDFGHighLevelDataset(
                dataset, val_ds.indices, high_level_cache,
                high_level_means, high_level_stds,
            )
            test_ds = CDFGHighLevelDataset(
                dataset, test_ds.indices, high_level_cache,
                high_level_means, high_level_stds,
            )
            exemplar_ds = CDFGHighLevelDataset(
                exemplar_dataset, exemplar_ds.indices, high_level_cache,
                high_level_means, high_level_stds,
            )

    batch_size = config.get("batch_size", 4)
    num_workers = config.get("num_workers")
    pin_memory = config.get("pin_memory")
    prefetch_factor = config.get("prefetch_factor", 2)
    thread_prefetch = config.get("thread_prefetch", False)
    train_sampler = None
    if config.get("family_balanced_sampling", False):
        family_counts = Counter(
            dataset.type_of(index) for index in train_ds.indices
        )
        weights = torch.tensor(
            [
                1.0 / family_counts[dataset.type_of(index)]
                for index in train_ds.indices
            ],
            dtype=torch.double,
        )
        train_sampler = WeightedRandomSampler(
            weights,
            num_samples=len(weights),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
    train_loader = make_loader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        distributed=distributed,
        sampler=train_sampler,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        thread_prefetch=thread_prefetch,
    )
    val_loader = make_loader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory, prefetch_factor=prefetch_factor,
        thread_prefetch=thread_prefetch,
    )
    test_loader = make_loader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory, prefetch_factor=prefetch_factor,
        thread_prefetch=thread_prefetch,
    )
    exemplar_loader = make_loader(
        exemplar_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory, prefetch_factor=prefetch_factor,
        thread_prefetch=thread_prefetch,
    )

    model = _model_from_config(config, len(vocab), max_pos, train_ds)
    precision = config.get("precision", "float32")
    if precision not in {"float32", "fp16", "bf16"}:
        raise ValueError("precision must be 'float32', 'fp16', or 'bf16'")
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bfloat16 training")
    model.inference_precision = precision
    optimizer_class = config.get("optimizer", "adamw")
    optimizer_kwargs = {
        "lr": config.get("learning_rate", 1e-3),
        "weight_decay": config.get("weight_decay", 0.0),
    }
    if optimizer_class == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    elif optimizer_class == "adam":
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
    else:
        raise ValueError("optimizer must be 'adamw' or 'adam'")
    loss_name = config.get("loss", "log_huber")
    if loss_name == "log_huber_hurdle":
        criterion = LogHuberHurdleLoss(
            unwrap_model(model).y_means,
            unwrap_model(model).y_stds,
            delta=config.get("log_huber_delta", 0.35),
            classification_weight=config.get(
                "hurdle_classification_weight", 0.25
            ),
        )
    elif loss_name == "log_huber":
        criterion = LogHuberLoss(
            unwrap_model(model).y_stds,
            delta=config.get("log_huber_delta", 0.35),
        )
    elif loss_name == "mse":
        criterion = nn.MSELoss()
    else:
        criterion = nn.HuberLoss(delta=config.get("huber_delta", 1.0))

    checkpoint_dir = _config_path(
        config.get("checkpoint_dir", "../artifacts/checkpoints"),
        config_dir,
    )
    results_dir = _config_path(
        config.get("results_dir", "../artifacts/results"),
        config_dir,
    )
    experiment_name = config.get("experiment_name", "baseline")
    run_dir = results_dir / experiment_name
    resume_checkpoint_path = config.get("resume_checkpoint_path")
    if resume_checkpoint_path:
        resume_checkpoint_path = _config_path(resume_checkpoint_path, config_dir)
    if (
        main_process and run_dir.exists() and any(run_dir.iterdir())
        and resume_checkpoint_path is None and args.evaluate_checkpoint is None
    ):
        raise FileExistsError(
            f"Refusing to overwrite existing run directory without resume/evaluation: {run_dir}"
        )
    if main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
    if saved_manifest is not None and config.get("train_scale") is None:
        # Preserve release/signature provenance rather than reducing saved rows
        # back to path-only membership records.
        split_manifest = {
            name: [dict(row) for row in saved_manifest[name]]
            for name in ("train", "validation", "test", "exemplar")
        }
    else:
        split_manifest = _split_manifest(
            dataset,
            {"train": train_ds, "validation": val_ds, "test": test_ds},
            tensor_dir,
        )
        split_manifest.update(
            _split_manifest(
                exemplar_dataset,
                {"exemplar": exemplar_ds},
                tensor_dir,
            )
        )
    membership = cohort_membership(split_manifest)
    manifest_hash = split_sha256(split_manifest)
    checkpoint_cadence = int(config.get("checkpoint_interval", 5))
    resolved_config = {
        **config,
        "model": config.get("model", "hetero_gat"),
        "kernel_types": kernel_types,
        "tensor_dir": str(tensor_dir),
        "vocab_path": str(vocab_path),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_path": str(
            checkpoint_dir / f"{experiment_name}_checkpoint.pt"
        ),
        "backup_checkpoint_path": str(
            checkpoint_dir / f"{experiment_name}_backup.pt"
        ),
        "resume_checkpoint_path": (
            str(resume_checkpoint_path) if resume_checkpoint_path else None
        ),
        "evaluation_checkpoint_path": (
            str(args.evaluate_checkpoint.resolve())
            if args.evaluate_checkpoint
            else None
        ),
        "results_dir": str(results_dir),
        "run_dir": str(run_dir),
        "device": str(device),
        "torch_version": torch.__version__,
        "seed": seed,
        "split_manifest_path": (
            str(split_manifest_path) if split_manifest_path else None
        ),
        "split_manifest_coverage": split_coverage,
        "split_sha256": manifest_hash,
        "cohort_membership": membership,
        "validation_cadence_epochs": 1,
        "checkpoint_cadence_epochs": checkpoint_cadence,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "gradient_clip_norm": config.get("gradient_clip_norm", 1.0),
        "lr_scheduler": "reduce_on_plateau",
        "lr_scheduler_patience": config.get("lr_scheduler_patience", 8),
        "lr_scheduler_factor": config.get("lr_scheduler_factor", 0.5),
        "min_learning_rate": config.get("min_learning_rate", 1e-6),
        "ll_hls4ml_git": _git_state(_REPO_ROOT),
    }
    if main_process:
        (run_dir / "resolved_config.json").write_text(
            json.dumps(resolved_config, indent=2)
        )
        (run_dir / "split_manifest.json").write_text(
            json.dumps(split_manifest, indent=2)
        )
        if data_scale_manifest is not None:
            (run_dir / "data_scale_manifest.json").write_text(
                json.dumps(data_scale_manifest, indent=2)
            )

    try:
        gpu_monitor = None
        if args.evaluate_checkpoint is not None:
            if distributed:
                raise ValueError(
                    "--evaluate-checkpoint is only supported in one process"
                )
            checkpoint = torch.load(
                args.evaluate_checkpoint.resolve(),
                map_location=device,
                weights_only=True,
            )
            model.load_state_dict(checkpoint["model"])
            model.to(device)
            training_state = checkpoint.get("training_state", {})
            base_model = unwrap_model(model)
            base_model.training_history = list(
                training_state.get("history", [])
            )
            base_model.best_epoch = training_state.get(
                "best_epoch",
                checkpoint.get("epoch"),
            )
            base_model.best_metric = training_state.get("best_metric")
            base_model.cumulative_training_seconds = training_state.get(
                "cumulative_training_seconds"
            )
            base_model.resumed_from_epoch = training_state.get(
                "resumed_from_epoch"
            )
            base_model.best_wall_seconds = training_state.get(
                "best_wall_seconds"
            )
            base_model.peak_gpu_memory_mb = training_state.get(
                "peak_gpu_memory_mb"
            )
            base_model.stop_reason = "checkpoint_evaluation"
        else:
            if main_process:
                # One writer samples rank 0. Starting one monitor per DDP rank
                # would corrupt the shared telemetry files on Kaggle.
                gpu_monitor = NvidiaSmiMonitor(
                    run_dir / "gpu_telemetry.csv",
                    interval_ms=int(config.get("gpu_telemetry_interval_ms", 0)),
                    gpu=local_rank if distributed else 0,
                )
                gpu_monitor.start()
            try:
                model = fit(
                    model,
                    train_loader,
                    val_loader,
                    epochs=config.get("epochs", 200),
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=None,
                    device=device,
                    patience=config.get("patience", 30),
                    mode="min",
                    restore_best_weights=True,
                    verbose=config.get("verbose", 5),
                    experiment_name=experiment_name,
                    checkpoint_dir=checkpoint_dir,
                    resume_from_backup=resume_checkpoint_path,
                    distributed=distributed,
                    early_stopping_metric=config.get(
                        "early_stopping_metric", "smape"
                    ),
                    precision=precision,
                    checkpoint_interval=checkpoint_cadence,
                    max_training_seconds=config.get("max_training_seconds"),
                    history_path=run_dir / "learning_curves.csv",
                    gradient_clip_norm=config.get("gradient_clip_norm", 1.0),
                    lr_scheduler_patience=config.get(
                        "lr_scheduler_patience", 8
                    ),
                    lr_scheduler_factor=config.get(
                        "lr_scheduler_factor", 0.5
                    ),
                    min_learning_rate=config.get(
                        "min_learning_rate", 1e-6
                    ),
                )
            finally:
                if gpu_monitor is not None:
                    gpu_monitor.stop()

        if main_process:
            base_model = unwrap_model(model)
            sizes = {
                "train": len(train_ds),
                "validation": len(val_ds),
                "test": len(test_ds),
                "exemplar": len(exemplar_ds),
            }
            if config.get("fit_only") and args.evaluate_checkpoint is None:
                history = getattr(base_model, "training_history", [])
                _write_csv(run_dir / "learning_curves.csv", history)
                _save_training_plots(run_dir, experiment_name, history)
                fit_accounting = {
                    "best_epoch": getattr(base_model, "best_epoch", None),
                    "best_validation_smape": getattr(
                        base_model, "best_metric", None
                    ),
                    "cumulative_training_seconds": getattr(
                        base_model, "cumulative_training_seconds", None
                    ),
                    "best_wall_seconds": getattr(
                        base_model, "best_wall_seconds", None
                    ),
                    "peak_gpu_memory_mb": getattr(
                        base_model, "peak_gpu_memory_mb", None
                    ),
                    "stop_reason": getattr(base_model, "stop_reason", None),
                    "gpu_telemetry": (
                        gpu_monitor.summary() if gpu_monitor is not None else None
                    ),
                    "sizes": sizes,
                    "checkpoint_sha256": _sha256(
                        checkpoint_dir / f"{experiment_name}_checkpoint.pt"
                    ),
                }
                resolved_config.update(fit_accounting)
                resolved_config["wall_seconds"] = time.perf_counter() - run_started
                (run_dir / "resolved_config.json").write_text(
                    json.dumps(resolved_config, indent=2)
                )
                (run_dir / "fit_summary.json").write_text(
                    json.dumps(fit_accounting, indent=2, default=_json_converter)
                )
                print(f"Wrote fit-only result bundle to {run_dir}")
                return
            metric_rows = []
            prediction_rows = []
            evaluation_loaders = {
                "validation": val_loader,
                "test": test_loader,
                "exemplar": exemplar_loader,
            }
            evaluation_splits = tuple(
                config.get("evaluation_splits", ("test", "exemplar"))
            )
            unknown_splits = set(evaluation_splits) - set(evaluation_loaders)
            if unknown_splits or len(evaluation_splits) != len(set(evaluation_splits)):
                raise ValueError(
                    f"Invalid evaluation_splits: {list(evaluation_splits)}"
                )
            for split_name in evaluation_splits:
                loader = evaluation_loaders[split_name]
                (
                    predictions,
                    targets,
                    presence_probabilities,
                    structures,
                    inference_latency,
                ) = _predict(
                    model, loader, device
                )
                metric_rows.extend(
                    _metric_rows(
                        split_name,
                        predictions,
                        targets,
                        inference_latency,
                    )
                )
                families = np.asarray(
                    [
                        sample["kernel_family"]
                        for sample in split_manifest[split_name]
                    ]
                )
                for family in sorted(set(families)):
                    family_mask = families == family
                    metric_rows.extend(
                        _metric_rows(
                            split_name,
                            predictions[family_mask],
                            targets[family_mask],
                            inference_latency,
                            kernel_family=family,
                        )
                    )
                prediction_rows.extend(
                    _prediction_rows(
                        split_name,
                        split_manifest[split_name],
                        predictions,
                        targets,
                        presence_probabilities,
                        structures,
                    )
                )
                _save_plots(
                    run_dir,
                    experiment_name,
                    split_name,
                    predictions,
                    targets,
                )
            _write_csv(run_dir / "metrics.csv", metric_rows)
            _write_csv(run_dir / "predictions.csv", prediction_rows)
            macro_rows = macro_metric_rows(metric_rows)
            structure_rows = structural_error_rows(prediction_rows)
            hurdle_rows = hurdle_confusion_rows(prediction_rows)
            calibration_rows = hurdle_calibration_rows(prediction_rows)
            _write_csv(run_dir / "macro_metrics.csv", macro_rows)
            _write_csv(run_dir / "structural_error_slices.csv", structure_rows)
            _write_csv(run_dir / "hurdle_confusion.csv", hurdle_rows)
            _write_csv(run_dir / "hurdle_calibration.csv", calibration_rows)
            history = getattr(base_model, "training_history", [])
            _write_csv(run_dir / "learning_curves.csv", history)
            _save_training_plots(run_dir, experiment_name, history)
            _save_calibration_plots(run_dir, calibration_rows)
            paired_summary = []
            baseline_predictions_path = config.get("baseline_predictions_path")
            if baseline_predictions_path:
                baseline_predictions_path = _config_path(
                    baseline_predictions_path, config_dir
                )
                paired_rows, paired_summary = paired_delta_rows(
                    prediction_rows,
                    read_prediction_rows(baseline_predictions_path),
                )
                _write_csv(run_dir / "paired_deltas_vs_h0.csv", paired_rows)
                _write_csv(
                    run_dir / "paired_delta_summary_vs_h0.csv", paired_summary
                )
            accounting = {
                "split_sha256": manifest_hash,
                "cohort_membership": membership,
                "hurdle_confusion": hurdle_rows,
                "validation_cadence_epochs": 1,
                "checkpoint_cadence_epochs": checkpoint_cadence,
                "cumulative_training_seconds": getattr(
                    base_model, "cumulative_training_seconds", None
                ),
                "best_wall_seconds": getattr(
                    base_model, "best_wall_seconds", None
                ),
                "peak_gpu_memory_mb": getattr(
                    base_model, "peak_gpu_memory_mb", None
                ),
                "mean_train_seconds_per_sample": (
                    float(
                        np.mean(
                            [
                                row["train_seconds_per_sample"]
                                for row in history
                            ]
                        )
                    )
                    if history
                    else None
                ),
                "stop_reason": getattr(base_model, "stop_reason", None),
                "gpu_telemetry": (
                    gpu_monitor.summary() if gpu_monitor is not None else None
                ),
                "resumed_from_epoch": getattr(
                    base_model, "resumed_from_epoch", None
                ),
                "macro_metrics": macro_rows,
                "paired_delta_summary_vs_h0": paired_summary,
            }
            (run_dir / "experiment_accounting.json").write_text(
                json.dumps(accounting, indent=2)
            )
            result = {
                "resolved_config": resolved_config,
                "sizes": sizes,
                "target_log_mean": base_model.y_means.detach().cpu(),
                "target_log_std": base_model.y_stds.detach().cpu(),
                "training_history": getattr(
                    base_model,
                    "training_history",
                    [],
                ),
                "best_epoch": getattr(base_model, "best_epoch", None),
                "best_metric": getattr(base_model, "best_metric", None),
                "metrics": metric_rows,
            }
            resolved_config["wall_seconds"] = time.perf_counter() - run_started
            resolved_config["cumulative_training_seconds"] = accounting[
                "cumulative_training_seconds"
            ]
            for key in (
                "best_wall_seconds",
                "peak_gpu_memory_mb",
                "mean_train_seconds_per_sample",
                "stop_reason",
                "gpu_telemetry",
            ):
                resolved_config[key] = accounting[key]
            result["resolved_config"] = resolved_config
            (run_dir / "resolved_config.json").write_text(
                json.dumps(resolved_config, indent=2)
            )
            output = run_dir / "summary.json"
            with output.open("w") as handle:
                json.dump(result, handle, indent=2, default=_json_converter)
            _write_report(
                run_dir,
                experiment_name,
                resolved_config,
                sizes,
                metric_rows,
                accounting,
            )
            print(f"Wrote result bundle to {run_dir}")
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
