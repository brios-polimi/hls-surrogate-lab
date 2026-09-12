#!/usr/bin/env python3
"""Run the frozen E5b matched-scratch attribution control."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hls-surrogate-lab-matplotlib")

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from ll_hls4ml.data.high_level import PROCESSED_FEATURE_DIM
from ll_hls4ml.data.vocab import load_vocab
from ll_hls4ml.io.schema import LABEL_KEYS
from ll_hls4ml.models.registry import build
from ll_hls4ml.training.loaders import make_loader
from ll_hls4ml.training.loops import fit
from ll_hls4ml.training.targets import LogHuberHurdleLoss
from ll_hls4ml.training.telemetry import NvidiaSmiMonitor
from scripts.run_e5 import (
    SCOPES,
    _archive_telemetry_segment,
    _derived_seed,
    _fusion_dataset,
    _git_revision,
    _gpu_preflight,
    _hierarchical_interval,
    _markdown_table,
    _metric_record,
    _predict_model,
    _prediction_rows,
    _rows_by_path,
    _set_seed,
    _sha256,
    _write_frame_atomic,
    _write_json_atomic,
    _write_stable_json,
    _write_stable_text,
    _write_text_atomic,
    _validate_source_manifest,
    audit_partitions,
)


STUDY_ID = "e5b_matched_scratch_v1"
PROTOCOL_ID = "e5b_late_added_matched_scratch_2026_09_11"
DEFAULT_E5_ROOT = _REPO_ROOT / "artifacts/results/e5_adaptation_v1"
DEFAULT_OUTPUT = _REPO_ROOT / "artifacts/results/e5b_matched_scratch_v1"
PRIMARY_BUDGETS = (16, 32)
PRIMARY_DRAW_SEEDS = (7, 42, 137)
ALLOWED_BUDGETS = (16, 32)
ALLOWED_DRAW_SEEDS = (7, 42, 137)
EXPECTED_E5_METADATA_SHA256 = (
    "5b5d18e9575b2dfd6d8c78404e2518cc67e9468b639636c4dfabd8f1296e48c6"
)
EXPECTED_PARTITIONS_SHA256 = (
    "8614dfd22de1365f9dd40663e9a265ede25a2e51543714a992848eedf454b5cf"
)
EXPECTED_PREPROCESSING_SHA256 = (
    "72befc5733f5fb1484dcb370d1b59cd5537848964ce3ad2b297bd58b0ee45ad3"
)
EXPECTED_SOURCE_CHECKPOINT_SHA256 = (
    "dbcf278abe0c95d3c7e1ccd564277e04bb38dff4b9fda5c9ddd9048ca10d8c48"
)

# Frozen before E5b training. These are the source E2 optimizer/loss settings,
# with a longer early-stopping patience for the much smaller validation sets.
SCRATCH_EPOCHS = 400
SCRATCH_PATIENCE = 50
SCRATCH_BATCH_SIZE = 8
SCRATCH_LEARNING_RATE = 1e-3
SCRATCH_WEIGHT_DECAY = 1e-4
LOG_HUBER_DELTA = 0.35
HURDLE_CLASSIFICATION_WEIGHT = 0.25
GRADIENT_CLIP_NORM = 1.0
LR_SCHEDULER_PATIENCE = 8
LR_SCHEDULER_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6
PRECISION = "bf16"
INITIALIZATION_SEED_BASE = 20260911


@dataclass
class Job:
    architecture_id: str
    draw_seed: int
    stage: str
    log_path: str
    device: str
    status: str = "PENDING"
    returncode: int | None = None


def _existing_or(recorded: str | Path, fallback: Path) -> Path:
    recorded = Path(recorded).expanduser()
    return recorded.resolve() if recorded.exists() else fallback.resolve()


def _run_dir(metadata: dict, architecture_id: str, draw_seed: int, budget: int) -> Path:
    return (
        Path(metadata["output_dir"]) / "runs" / architecture_id
        / f"draw{draw_seed}" / f"scratch_k{budget}"
    )


def _experiment(architecture_id: str, draw_seed: int, budget: int) -> str:
    return f"e5b_scratch_{architecture_id}_draw{draw_seed}_k{budget}"


def _run_seeds(architecture_id: str, draw_seed: int, budget: int) -> tuple[int, int]:
    initialization_seed = _derived_seed(
        STUDY_ID, INITIALIZATION_SEED_BASE, architecture_id, draw_seed, "init"
    )
    training_seed = _derived_seed(
        STUDY_ID, INITIALIZATION_SEED_BASE, architecture_id, draw_seed, budget, "fit"
    )
    return initialization_seed, training_seed


def _build_scratch_model(metadata: dict) -> torch.nn.Module:
    resolved = json.loads(Path(metadata["source_resolved_config"]).read_text())
    preprocessing = torch.load(
        metadata["preprocessing_path"], map_location="cpu", weights_only=True
    )
    vocabulary, max_pos, _ = load_vocab(metadata["vocab"])
    return build(
        "hierarchical_high_level_fusion",
        instruction_vocab_size=len(vocabulary),
        edge_pos_vocab_size=max_pos,
        high_level_input_dim=PROCESSED_FEATURE_DIM,
        y_means=preprocessing["y_means"],
        y_stds=preprocessing["y_stds"],
        hidden_dim=int(resolved.get("hidden_dim", 64)),
        num_layers=int(resolved.get("num_layers", 3)),
        heads=int(resolved.get("heads", 1)),
        dropout=float(resolved.get("dropout", 0.15)),
        high_level_encoder=resolved.get("high_level_encoder", "gatv2"),
        use_global_features=bool(resolved.get("use_global_features", True)),
        use_context=bool(resolved.get("use_context", True)),
        context_mode=resolved.get("context_mode", "core"),
        split_heads=True,
        hurdle_heads=True,
        hurdle_prediction_mode=resolved.get("hurdle_prediction_mode", "threshold"),
    )


def _validate_matched_architecture(metadata: dict) -> int:
    _set_seed(INITIALIZATION_SEED_BASE)
    model = _build_scratch_model(metadata)
    checkpoint = torch.load(
        metadata["source_checkpoint"], map_location="cpu", weights_only=True
    )["model"]
    current = model.state_dict()
    if current.keys() != checkpoint.keys():
        raise ValueError("Scratch and E2 source state dictionaries differ")
    for name in current:
        if current[name].shape != checkpoint[name].shape:
            raise ValueError(f"Scratch/source shape mismatch for {name}")
    if not torch.equal(current["y_means"], checkpoint["y_means"]) or not torch.equal(
        current["y_stds"], checkpoint["y_stds"]
    ):
        raise ValueError("Scratch/source target normalization differs")
    learned_names = [name for name in current if name not in {"y_means", "y_stds"}]
    if all(torch.equal(current[name], checkpoint[name]) for name in learned_names):
        raise ValueError("Scratch model unexpectedly has source checkpoint weights")
    return sum(parameter.numel() for parameter in model.parameters())


def _fit_complete(metadata: dict, architecture_id: str, draw_seed: int, budgets) -> bool:
    for budget in budgets:
        run_dir = _run_dir(metadata, architecture_id, draw_seed, budget)
        checkpoint = (
            run_dir / "checkpoints"
            / f"{_experiment(architecture_id, draw_seed, budget)}_checkpoint.pt"
        )
        marker = run_dir / "fit_summary.json"
        if not marker.is_file() or not checkpoint.is_file():
            return False
        try:
            summary = json.loads(marker.read_text())
        except (json.JSONDecodeError, OSError):
            return False
        if summary.get("checkpoint_sha256") != _sha256(checkpoint):
            return False
    return True


def _evaluation_complete(
    metadata: dict, architecture_id: str, draw_seed: int, budgets
) -> bool:
    for budget in budgets:
        run_dir = _run_dir(metadata, architecture_id, draw_seed, budget)
        predictions = run_dir / "predictions.csv"
        marker = run_dir / "evaluation_summary.json"
        if not predictions.is_file() or not marker.is_file():
            return False
        try:
            summary = json.loads(marker.read_text())
        except (json.JSONDecodeError, OSError):
            return False
        if summary.get("predictions_sha256") != _sha256(predictions):
            return False
    return True


def _matrix_complete(
    metadata: dict,
    stage: str,
    architectures,
    draw_seeds,
    budgets,
) -> bool:
    check = _fit_complete if stage == "fit" else _evaluation_complete
    return all(
        check(metadata, architecture, draw, budgets)
        for architecture in architectures
        for draw in draw_seeds
    )


def _partial_status(
    metadata: dict, architecture_id: str, draw_seed: int, budgets, resume: bool
) -> str | None:
    for budget in budgets:
        run_dir = _run_dir(metadata, architecture_id, draw_seed, budget)
        experiment = _experiment(architecture_id, draw_seed, budget)
        checkpoint_dir = run_dir / "checkpoints"
        marker = run_dir / "fit_summary.json"
        best = checkpoint_dir / f"{experiment}_checkpoint.pt"
        backup = checkpoint_dir / f"{experiment}_backup.pt"
        if marker.exists() != best.exists():
            return "BLOCKED_INCONSISTENT_COMPLETION"
        if marker.is_file():
            continue
        entries = (
            [path for path in run_dir.iterdir() if path.name != "run_config.json"]
            if run_dir.is_dir() else []
        )
        if entries and not resume:
            return "BLOCKED_PARTIAL_USE_RESUME"
        if entries and not backup.is_file():
            return "BLOCKED_NO_BACKUP_CHECKPOINT"
    return None


def _fit_one(
    metadata: dict,
    architecture_id: str,
    draw_seed: int,
    budget: int,
    support_paths: list[str],
    validation_paths: list[str],
    high_level_cache: dict,
    resume: bool,
) -> None:
    run_dir = _run_dir(metadata, architecture_id, draw_seed, budget)
    experiment = _experiment(architecture_id, draw_seed, budget)
    checkpoint_dir = run_dir / "checkpoints"
    best_checkpoint = checkpoint_dir / f"{experiment}_checkpoint.pt"
    backup = checkpoint_dir / f"{experiment}_backup.pt"
    marker = run_dir / "fit_summary.json"
    if marker.is_file() and best_checkpoint.is_file():
        summary = json.loads(marker.read_text())
        if summary.get("checkpoint_sha256") != _sha256(best_checkpoint):
            raise RuntimeError(f"Checkpoint hash mismatch: {run_dir}")
        return
    if marker.exists() or best_checkpoint.exists():
        raise RuntimeError(f"Inconsistent completion artifacts: {run_dir}")
    entries = (
        [path for path in run_dir.iterdir() if path.name != "run_config.json"]
        if run_dir.is_dir() else []
    )
    if entries and (not resume or not backup.is_file()):
        raise RuntimeError(f"Unsafe partial scratch run: {run_dir}; use --resume")

    initialization_seed, training_seed = _run_seeds(
        architecture_id, draw_seed, budget
    )
    config = {
        "study_id": STUDY_ID,
        "protocol_id": PROTOCOL_ID,
        "architecture_id": architecture_id,
        "draw_seed": draw_seed,
        "budget": budget,
        "method": "scratch",
        "initialization_seed": initialization_seed,
        "training_seed": training_seed,
        "epochs": SCRATCH_EPOCHS,
        "patience": SCRATCH_PATIENCE,
        "batch_size": SCRATCH_BATCH_SIZE,
        "learning_rate": SCRATCH_LEARNING_RATE,
        "weight_decay": SCRATCH_WEIGHT_DECAY,
        "selection_split": "adaptation_validation",
        "query_used_for_selection": False,
        "normalization": "frozen E5 source preprocessing",
        "partitions_sha256": metadata["partitions_sha256"],
    }
    _write_stable_json(run_dir / "run_config.json", config)

    _set_seed(initialization_seed)
    model = _build_scratch_model(metadata)
    _set_seed(training_seed)
    preprocessing = torch.load(
        metadata["preprocessing_path"], map_location="cpu", weights_only=True
    )
    train_dataset = _fusion_dataset(
        metadata, support_paths, high_level_cache,
        preprocessing["high_level_means"], preprocessing["high_level_stds"],
    )
    validation_dataset = _fusion_dataset(
        metadata, validation_paths, high_level_cache,
        preprocessing["high_level_means"], preprocessing["high_level_stds"],
    )
    train_loader = make_loader(
        train_dataset,
        batch_size=min(SCRATCH_BATCH_SIZE, budget),
        shuffle=True,
        num_workers=metadata["num_workers"],
        pin_memory=True,
        prefetch_factor=metadata["prefetch_factor"],
        thread_prefetch=metadata["thread_prefetch"],
    )
    validation_loader = make_loader(
        validation_dataset,
        batch_size=metadata["evaluation_batch_size"],
        shuffle=False,
        num_workers=metadata["num_workers"],
        pin_memory=True,
        prefetch_factor=metadata["prefetch_factor"],
        thread_prefetch=metadata["thread_prefetch"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=SCRATCH_LEARNING_RATE, weight_decay=SCRATCH_WEIGHT_DECAY
    )
    criterion = LogHuberHurdleLoss(
        model.y_means,
        model.y_stds,
        delta=LOG_HUBER_DELTA,
        classification_weight=HURDLE_CLASSIFICATION_WEIGHT,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if resume and backup.is_file():
        _archive_telemetry_segment(run_dir)
    monitor = NvidiaSmiMonitor(
        run_dir / "gpu_telemetry.csv",
        interval_ms=metadata["gpu_telemetry_interval_ms"],
        gpu=os.environ.get("E5B_MONITOR_GPU", "0"),
    )
    started = time.perf_counter()
    monitor.start()
    try:
        model = fit(
            model,
            train_loader,
            validation_loader,
            epochs=SCRATCH_EPOCHS,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,
            device=device,
            patience=SCRATCH_PATIENCE,
            mode="min",
            restore_best_weights=True,
            verbose=metadata["verbose"],
            experiment_name=experiment,
            checkpoint_dir=checkpoint_dir,
            resume_from_backup=backup if resume and backup.is_file() else None,
            early_stopping_metric="smape",
            precision=PRECISION,
            checkpoint_interval=1,
            history_path=run_dir / "learning_curves.csv",
            gradient_clip_norm=GRADIENT_CLIP_NORM,
            lr_scheduler_patience=LR_SCHEDULER_PATIENCE,
            lr_scheduler_factor=LR_SCHEDULER_FACTOR,
            min_learning_rate=MIN_LEARNING_RATE,
        )
    finally:
        monitor.stop()
    _write_json_atomic(marker, {
        "status": "complete",
        "method": "scratch",
        "architecture_id": architecture_id,
        "draw_seed": draw_seed,
        "budget": budget,
        "initialization_seed": initialization_seed,
        "training_seed": training_seed,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": model.best_epoch,
        "best_validation_smape": model.best_metric,
        "stop_reason": model.stop_reason,
        "wall_seconds": time.perf_counter() - started,
        "gpu_telemetry": monitor.summary(),
        "checkpoint_sha256": _sha256(best_checkpoint),
    })


def _fit_bundle(
    metadata: dict, architecture_id: str, draw_seed: int, budgets, resume: bool
) -> None:
    partitions = json.loads(Path(metadata["partitions_path"]).read_text())
    record = partitions["architectures"][architecture_id]
    validation_paths = record["validation"]
    draw = record["draws"][str(draw_seed)]
    high_level_cache = torch.load(
        metadata["high_level_cache"], map_location="cpu", weights_only=False
    )
    for budget in budgets:
        print(
            f"FIT E5b architecture={architecture_id} draw={draw_seed} k={budget}",
            flush=True,
        )
        _fit_one(
            metadata, architecture_id, draw_seed, budget,
            draw["support"][str(budget)], validation_paths,
            high_level_cache, resume,
        )


def _evaluate_bundle(metadata: dict, architecture_id: str, draw_seed: int, budgets) -> None:
    manifest = json.loads(Path(metadata["source_manifest"]).read_text())
    rows_by_path = _rows_by_path(manifest)
    partitions = json.loads(Path(metadata["partitions_path"]).read_text())
    record = partitions["architectures"][architecture_id]
    validation_paths = record["validation"]
    query_paths = record["query"]
    preprocessing = torch.load(
        metadata["preprocessing_path"], map_location="cpu", weights_only=True
    )
    high_level_cache = torch.load(
        metadata["high_level_cache"], map_location="cpu", weights_only=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for budget in budgets:
        run_dir = _run_dir(metadata, architecture_id, draw_seed, budget)
        marker = run_dir / "evaluation_summary.json"
        prediction_path = run_dir / "predictions.csv"
        if marker.is_file() and prediction_path.is_file():
            summary = json.loads(marker.read_text())
            if summary.get("predictions_sha256") != _sha256(prediction_path):
                raise RuntimeError(f"Prediction hash mismatch: {run_dir}")
            continue
        if not (run_dir / "fit_summary.json").is_file():
            raise FileNotFoundError(f"Incomplete E5b fit: {run_dir}")
        support_paths = record["draws"][str(draw_seed)]["support"][str(budget)]
        paths = [*support_paths, *validation_paths, *query_paths]
        model = _build_scratch_model(metadata)
        checkpoint_path = (
            run_dir / "checkpoints"
            / f"{_experiment(architecture_id, draw_seed, budget)}_checkpoint.pt"
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=True)
        dataset = _fusion_dataset(
            metadata, paths, high_level_cache,
            preprocessing["high_level_means"], preprocessing["high_level_stds"],
        )
        loader = make_loader(
            dataset,
            batch_size=metadata["evaluation_batch_size"],
            shuffle=False,
            num_workers=metadata["num_workers"],
            pin_memory=True,
            prefetch_factor=metadata["prefetch_factor"],
            thread_prefetch=metadata["thread_prefetch"],
        )
        predictions, _ = _predict_model(model, loader, device, PRECISION)
        support_end = len(support_paths)
        validation_end = support_end + len(validation_paths)
        rows = _prediction_rows(
            support_paths, "support", predictions[:support_end], rows_by_path,
            architecture_id, draw_seed, "scratch", budget,
        )
        rows.extend(_prediction_rows(
            validation_paths, "adaptation_validation",
            predictions[support_end:validation_end], rows_by_path,
            architecture_id, draw_seed, "scratch", budget,
        ))
        rows.extend(_prediction_rows(
            query_paths, "query", predictions[validation_end:], rows_by_path,
            architecture_id, draw_seed, "scratch", budget,
        ))
        from scripts.run_e5 import _write_csv_atomic

        _write_csv_atomic(prediction_path, rows)
        _write_json_atomic(marker, {
            "status": "complete",
            "architecture_id": architecture_id,
            "draw_seed": draw_seed,
            "budget": budget,
            "support_samples": len(support_paths),
            "adaptation_validation_samples": len(validation_paths),
            "query_samples": len(query_paths),
            "predictions_sha256": _sha256(prediction_path),
        })


def _verified_predictions(path: Path) -> pd.DataFrame:
    marker = path.with_name("evaluation_summary.json")
    if not path.is_file() or not marker.is_file():
        raise FileNotFoundError(path)
    if json.loads(marker.read_text()).get("predictions_sha256") != _sha256(path):
        raise RuntimeError(f"Prediction hash mismatch: {path}")
    return pd.read_csv(path)


def _analysis_dir(metadata: dict, draw_seeds, budgets) -> Path:
    if tuple(draw_seeds) == PRIMARY_DRAW_SEEDS and tuple(budgets) == PRIMARY_BUDGETS:
        name = "analysis"
    else:
        draws = "-".join(map(str, draw_seeds))
        sizes = "-".join(map(str, budgets))
        name = f"analysis_selection_draws{draws}_k{sizes}"
    return Path(metadata["output_dir"]) / name


def _analyze(metadata: dict, architectures, draw_seeds, budgets) -> None:
    partitions = json.loads(Path(metadata["partitions_path"]).read_text())
    architectures = tuple(architectures)
    metric_rows = []
    workload = []
    zero = _verified_predictions(
        Path(metadata["e5_root"]) / "runs/zero_shot/predictions.csv"
    )
    for architecture in architectures:
        frame = zero.query(
            "split == 'query' and adapted_for_architecture == @architecture"
        )
        metric_rows.append({
            "method": "zero_shot", "budget": 0,
            "architecture_id": architecture, "draw_seed": np.nan,
            "n_samples": len(frame), **_metric_record(frame),
        })
        for draw_seed in draw_seeds:
            for budget in budgets:
                scratch_dir = _run_dir(metadata, architecture, draw_seed, budget)
                scratch = _verified_predictions(scratch_dir / "predictions.csv")
                scratch_query = scratch[scratch["split"] == "query"]
                metric_rows.append({
                    "method": "scratch", "budget": budget,
                    "architecture_id": architecture, "draw_seed": draw_seed,
                    "n_samples": len(scratch_query), **_metric_record(scratch_query),
                })
                head_dir = (
                    Path(metadata["e5_root"]) / "runs" / architecture
                    / f"draw{draw_seed}" / f"head_k{budget}"
                )
                head = _verified_predictions(head_dir / "predictions.csv")
                head_query = head[head["split"] == "query"]
                metric_rows.append({
                    "method": "pretrained_head", "budget": budget,
                    "architecture_id": architecture, "draw_seed": draw_seed,
                    "n_samples": len(head_query), **_metric_record(head_query),
                })
                summary = json.loads((scratch_dir / "fit_summary.json").read_text())
                workload.append({
                    "architecture_id": architecture,
                    "draw_seed": draw_seed,
                    "budget": budget,
                    **{key: summary.get(key) for key in (
                        "trainable_parameters", "wall_seconds", "best_epoch",
                        "best_validation_smape", "stop_reason",
                    )},
                })
    metrics = pd.DataFrame(metric_rows)
    analysis_dir = _analysis_dir(metadata, draw_seeds, budgets)
    _write_frame_atomic(analysis_dir / "per_run_metrics.csv", metrics)
    _write_frame_atomic(analysis_dir / "workload.csv", pd.DataFrame(workload))

    summary_rows = []
    for (method, budget), frame in metrics.groupby(["method", "budget"]):
        for scope in SCOPES:
            value = f"smape_{scope}"
            means = frame.groupby("architecture_id")[value].mean()
            low, high = _hierarchical_interval(
                frame, value, metadata["bootstrap_replicates"],
                _derived_seed(STUDY_ID, "summary", method, budget, scope),
            )
            summary_rows.append({
                "method": method,
                "budget": int(budget),
                "scope": scope,
                "equal_architecture_smape": float(means.mean()),
                "ci95_low": low,
                "ci95_high": high,
            })
    summary = pd.DataFrame(summary_rows)
    _write_frame_atomic(analysis_dir / "equal_architecture_summary.csv", summary)

    paired_rows = []
    for budget in budgets:
        scratch = metrics.query("method == 'scratch' and budget == @budget")
        head = metrics.query("method == 'pretrained_head' and budget == @budget")
        keys = ["architecture_id", "draw_seed"]
        paired = scratch.merge(head, on=keys, suffixes=("_scratch", "_head"))
        for scope in SCOPES:
            paired[f"head_advantage_{scope}"] = (
                paired[f"smape_{scope}_scratch"] - paired[f"smape_{scope}_head"]
            )
            value = f"head_advantage_{scope}"
            low, high = _hierarchical_interval(
                paired, value, metadata["bootstrap_replicates"],
                _derived_seed(STUDY_ID, "paired", budget, scope),
            )
            architecture_means = paired.groupby("architecture_id")[value].mean()
            paired_rows.append({
                "budget": budget,
                "scope": scope,
                "pretrained_head_advantage_smape": float(architecture_means.mean()),
                "ci95_low": low,
                "ci95_high": high,
                "architecture_win_fraction": float((architecture_means > 0).mean()),
            })
    paired_summary = pd.DataFrame(paired_rows)
    _write_frame_atomic(
        analysis_dir / "paired_pretrained_head_vs_scratch.csv", paired_summary
    )
    _write_figures(analysis_dir, summary, metrics, paired_summary, budgets)
    _write_text_atomic(
        analysis_dir / "REPORT.md", _report(summary, paired_summary, workload)
    )
    _write_json_atomic(analysis_dir / "analysis_provenance.json", {
        "study_id": STUDY_ID,
        "protocol_id": PROTOCOL_ID,
        "late_added_after_query_inspection": True,
        "query_used_for_selection": False,
        "outer_unit": "seven fixed exemplar architectures",
        "support_draws_are_repeated_measures": True,
        "bootstrap_replicates": metadata["bootstrap_replicates"],
        "draw_seeds": list(draw_seeds),
        "budgets": list(budgets),
        "study_metadata_sha256": _sha256(Path(metadata["metadata_path"])),
        "partitions_sha256": metadata["partitions_sha256"],
    })
    inventory_path = analysis_dir / "artifact_inventory.csv"
    inventory = []
    output_root = Path(metadata["output_dir"])
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path != inventory_path and not path.name.endswith(".tmp"):
            inventory.append({
                "relative_path": str(path.relative_to(output_root)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
    _write_frame_atomic(inventory_path, pd.DataFrame(inventory))


def _write_figures(
    output: Path,
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    paired: pd.DataFrame,
    budgets,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    for axis, scope in zip(axes, ("overall", "resource", "timing")):
        selected = summary[summary["scope"] == scope]
        for method in ("zero_shot", "scratch", "pretrained_head"):
            rows = selected[selected["method"] == method].sort_values("budget")
            if rows.empty:
                continue
            x = rows["budget"].to_numpy(float)
            axis.plot(
                x, rows["equal_architecture_smape"].to_numpy(float),
                marker="o", label=method,
            )
            axis.fill_between(
                x, rows["ci95_low"].to_numpy(float),
                rows["ci95_high"].to_numpy(float), alpha=0.15,
            )
        axis.set_title(scope)
        axis.set_xlabel("support labels k")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("equal-architecture query SMAPE")
    axes[-1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "scratch_vs_pretrained.png", dpi=180)
    plt.close(figure)

    overall = metrics[metrics["method"].isin(["scratch", "pretrained_head"])]
    figure, axes = plt.subplots(1, len(budgets), figsize=(6 * len(budgets), 5), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, budget in zip(axes, budgets):
        selected = overall[overall["budget"] == budget]
        pivot = selected.groupby(
            ["architecture_id", "method"]
        )["smape_overall"].mean().unstack()
        axis.scatter(pivot["scratch"], pivot["pretrained_head"])
        limit = float(max(pivot.max().max(), 1))
        axis.plot([0, limit], [0, limit], color="black", linewidth=0.8)
        for architecture, row in pivot.iterrows():
            axis.annotate(architecture[:6], (row["scratch"], row["pretrained_head"]))
        axis.set_title(f"k={budget}")
        axis.set_xlabel("scratch SMAPE")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("pretrained-head SMAPE")
    figure.tight_layout()
    figure.savefig(output / "per_architecture_attribution.png", dpi=180)
    plt.close(figure)


def _report(summary: pd.DataFrame, paired: pd.DataFrame, workload: list[dict]) -> str:
    overall = summary[summary["scope"] == "overall"].copy()
    overall["result"] = overall.apply(
        lambda row: (
            f"{row['equal_architecture_smape']:.2f} "
            f"[{row['ci95_low']:.2f}, {row['ci95_high']:.2f}]"
        ), axis=1,
    )
    paired_overall = paired[paired["scope"] == "overall"].copy()
    decisions = []
    for _, row in paired_overall.iterrows():
        if row["ci95_low"] > 0:
            decision = "pretrained head better"
        elif row["ci95_high"] < 0:
            decision = "scratch better"
        else:
            decision = "statistically unresolved"
        decisions.append(decision)
    paired_overall["decision"] = decisions
    total_wall = sum(float(row.get("wall_seconds") or 0) for row in workload)
    return f"""# E5b matched-scratch attribution control

This is a late-added control frozen after the original E5 query was inspected.
No E5b hyperparameter or checkpoint decision used query labels.
The primary analysis uses three fixed support draws at k=16 and k=32 across
seven outer architecture units. Draws are treated as repeated measures within
architecture, not as independent outer replicates.

## Equal-architecture query results

{_markdown_table(overall[["method", "budget", "result"]])}

## Pretrained-head advantage over scratch

Positive SMAPE values favor the pretrained frozen-encoder head.

{_markdown_table(paired_overall)}

## Compute

Summed scratch fitting time: {total_wall / 3600:.2f} hours. Concurrent wall time
is lower. Per-run checkpoints, learning curves, telemetry, raw predictions,
resolved settings, and hashes are retained with this report.
"""


def _worker_command(stage: str, metadata_path: Path, job: Job, budgets, resume) -> list[str]:
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker-stage", stage,
        "--study-metadata", str(metadata_path),
        "--architecture", job.architecture_id,
        "--draw-seed", str(job.draw_seed),
        "--budgets", *map(str, budgets),
    ]
    if resume:
        command.append("--resume")
    return command


def _dispatch(
    stage: str,
    metadata: dict,
    metadata_path: Path,
    architectures,
    draw_seeds,
    budgets,
    devices,
    jobs_per_device: int,
    resume: bool,
    fail_fast: bool,
    dry_run: bool,
) -> None:
    check = _fit_complete if stage == "fit" else _evaluation_complete
    slots = [device for device in devices for _ in range(jobs_per_device)]
    jobs = []
    for position, (architecture, draw) in enumerate(
        (architecture, draw) for architecture in architectures for draw in draw_seeds
    ):
        status = (
            "SKIPPED_COMPLETE" if check(metadata, architecture, draw, budgets)
            else "PENDING"
        )
        if stage == "fit" and status == "PENDING":
            status = _partial_status(
                metadata, architecture, draw, budgets, resume
            ) or status
        jobs.append(Job(
            architecture_id=architecture,
            draw_seed=draw,
            stage=stage,
            log_path=str(
                Path(metadata["output_dir"]) / "logs" / stage
                / f"{architecture}_draw{draw}.log"
            ),
            device=slots[position % len(slots)],
            status=status,
        ))
    output = Path(metadata["output_dir"])
    latest_index = output / f"{stage}_index.json"
    invocation = (
        output / "logs/invocations"
        / f"{time.strftime('%Y%m%dT%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}_{stage}.json"
    )

    def persist() -> None:
        payload = {
            "stage": stage, "budgets": list(budgets),
            "draw_seeds": list(draw_seeds), "architectures": list(architectures),
            "devices": list(devices), "jobs_per_device": jobs_per_device,
            "resume": resume, "dry_run": dry_run,
            "jobs": [asdict(job) for job in jobs],
        }
        _write_json_atomic(latest_index, payload)
        _write_json_atomic(invocation, payload)

    persist()
    blocked = [job for job in jobs if job.status.startswith("BLOCKED")]
    if blocked:
        for job in blocked:
            print(f"{job.status}: {job.architecture_id} draw {job.draw_seed}")
        raise SystemExit(2)
    pending = [job for job in jobs if job.status == "PENDING"]
    if dry_run:
        for job in pending:
            print(
                f"CUDA_VISIBLE_DEVICES={shlex.quote(job.device)} "
                + shlex.join(_worker_command(stage, metadata_path, job, budgets, resume))
            )
        print(f"E5b {stage}: runnable={len(pending)} index={latest_index}")
        return
    if not pending:
        print(f"All selected E5b {stage} bundles are complete.")
        return
    for device in dict.fromkeys(job.device for job in pending):
        _gpu_preflight(device)
    available = list(slots)
    active = []
    failed = False
    try:
        while pending or active:
            while pending and available and not (failed and fail_fast):
                job = pending.pop(0)
                device = available.pop(0)
                job.device = device
                command = _worker_command(stage, metadata_path, job, budgets, resume)
                log_path = Path(job.log_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                handle = log_path.open("a", buffering=1)
                handle.write(
                    f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"{shlex.join(command)} =====\n"
                )
                environment = os.environ.copy()
                environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                environment["CUDA_VISIBLE_DEVICES"] = device
                environment["E5B_MONITOR_GPU"] = device
                environment["LL_HLS4ML_TQDM"] = "0"
                process = subprocess.Popen(
                    command, cwd=_REPO_ROOT, env=environment,
                    stdout=handle, stderr=subprocess.STDOUT, text=True,
                )
                job.status = "RUNNING"
                active.append((process, job, handle))
                print(
                    f"Started E5b {stage} {job.architecture_id} draw "
                    f"{job.draw_seed} on GPU {device}; log={job.log_path}",
                    flush=True,
                )
                persist()
            if not active:
                break
            time.sleep(2)
            remaining = []
            for process, job, handle in active:
                returncode = process.poll()
                if returncode is None:
                    remaining.append((process, job, handle))
                    continue
                handle.close()
                available.append(job.device)
                job.returncode = returncode
                valid = returncode == 0 and check(
                    metadata, job.architecture_id, job.draw_seed, budgets
                )
                job.status = "COMPLETE" if valid else "FAILED"
                failed = failed or not valid
                print(
                    f"{job.status}: E5b {stage} {job.architecture_id} "
                    f"draw {job.draw_seed}", flush=True,
                )
                persist()
            active = remaining
        if failed and fail_fast:
            for job in pending:
                job.status = "SKIPPED_FAIL_FAST"
            persist()
    except KeyboardInterrupt:
        for process, job, handle in active:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            handle.close()
            job.status = "INTERRUPTED"
            job.returncode = process.returncode
        persist()
        raise
    if failed:
        raise SystemExit(1)


def _resolve_inputs(args) -> tuple[dict, Path]:
    e5_root = args.e5_root.resolve()
    e5_metadata_path = e5_root / "study_metadata.json"
    partitions_path = e5_root / "protocol/e5_partitions.json"
    preprocessing_path = e5_root / "cache/preprocessing.pt"
    for path in (e5_metadata_path, partitions_path, preprocessing_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if _sha256(e5_metadata_path) != EXPECTED_E5_METADATA_SHA256:
        raise ValueError("E5 metadata hash differs from the frozen completed study")
    if _sha256(partitions_path) != EXPECTED_PARTITIONS_SHA256:
        raise ValueError("E5 partition hash differs from the frozen completed study")
    if _sha256(preprocessing_path) != EXPECTED_PREPROCESSING_SHA256:
        raise ValueError("E5 preprocessing hash differs from the frozen completed study")
    e5 = json.loads(e5_metadata_path.read_text())
    partitions = json.loads(partitions_path.read_text())
    audit_partitions(partitions)

    source_manifest = args.source_manifest or _existing_or(
        e5["source_manifest"],
        _REPO_ROOT / "artifacts/releases/vitis-a31-coarsearch1-hierarchy2-vocab-2026-09-04/e2_structural_v1/architecture_grouped_structural_v1.json",
    )
    source_config = args.source_resolved_config or _existing_or(
        e5["source_resolved_config"],
        _REPO_ROOT / "artifacts/results/e2_replication_v1/seed42/e2_fusion_structural_seed42/resolved_config.json",
    )
    source_checkpoint = args.source_checkpoint or _existing_or(
        e5["source_checkpoint"],
        _REPO_ROOT / "artifacts/results/e2_replication_v1/seed42/e2_fusion_structural_seed42/checkpoints/e2_fusion_structural_seed42_checkpoint.pt",
    )
    vocab = args.vocab or _existing_or(e5["vocab"], _REPO_ROOT / "artifacts/vocab/vocab.json")
    high_level_cache = args.high_level_cache or _existing_or(
        e5["high_level_cache"], _REPO_ROOT / "artifacts/cache/wa_high_level_archives1_32.pt"
    )
    tensor_dir = (args.tensor_dir or Path(e5["tensor_dir"])).expanduser().resolve()
    tensor_index = (
        args.tensor_index or tensor_dir / "labels.json"
    ).expanduser().resolve()
    inputs = {
        "source_manifest": Path(source_manifest).resolve(),
        "source_resolved_config": Path(source_config).resolve(),
        "source_checkpoint": Path(source_checkpoint).resolve(),
        "vocab": Path(vocab).resolve(),
        "high_level_cache": Path(high_level_cache).resolve(),
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    expected_hashes = {
        "source_manifest": e5["source_manifest_sha256"],
        "source_resolved_config": e5["source_resolved_config_sha256"],
        "source_checkpoint": EXPECTED_SOURCE_CHECKPOINT_SHA256,
        "vocab": e5["vocab_sha256"],
        "high_level_cache": e5["high_level_cache_sha256"],
    }
    for name, path in inputs.items():
        if _sha256(path) != expected_hashes[name]:
            raise ValueError(f"Frozen input hash mismatch: {name} ({path})")
    _, source_split_hash = _validate_source_manifest(
        inputs["source_manifest"], e5["source_split_sha256"]
    )
    resolved = json.loads(inputs["source_resolved_config"].read_text())
    if resolved.get("model") != "hierarchical_high_level_fusion":
        raise ValueError("E5b source must be the E2 fusion model")
    if resolved.get("seed") != 42 or resolved.get("split_sha256") != source_split_hash:
        raise ValueError("E5b source checkpoint identity does not match E2 seed 42")
    checkpoint = torch.load(
        inputs["source_checkpoint"], map_location="cpu", weights_only=True
    )
    if "model" not in checkpoint or "y_means" not in checkpoint["model"]:
        raise ValueError("Source checkpoint is incomplete")
    if not args.dry_run:
        if not tensor_dir.is_dir() or not tensor_index.is_file():
            raise FileNotFoundError(tensor_index)
        if _sha256(tensor_index) != e5["tensor_index_sha256"]:
            raise ValueError("Frozen tensor-index hash mismatch")

    output = args.output_dir.resolve()
    provenance = output / "provenance"
    script_snapshot = provenance / "run_e5b.py"
    e5_snapshot = provenance / "run_e5.py"
    fusion_snapshot = provenance / "fusion.py"
    _write_stable_text(script_snapshot, Path(__file__).read_text())
    _write_stable_text(e5_snapshot, (_REPO_ROOT / "scripts/run_e5.py").read_text())
    _write_stable_text(
        fusion_snapshot, (_REPO_ROOT / "src/ll_hls4ml/models/fusion.py").read_text()
    )
    environment_path = provenance / "environment.json"
    _write_stable_json(environment_path, {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    })
    protocol_path = output / "protocol/e5b_protocol.json"
    protocol = {
        "study_id": STUDY_ID,
        "protocol_id": PROTOCOL_ID,
        "registered_on": "2026-09-11",
        "status": "late-added control after E5 query inspection",
        "question": "Does source-pretrained head adaptation beat matched random initialization?",
        "changed_variable": "learned fusion-model initialization",
        "held_fixed": [
            "fusion architecture", "source feature and target normalization",
            "E5 support/validation/query partitions", "loss and optimizer family",
            "validation-only checkpoint selection",
        ],
        "primary_budgets": list(PRIMARY_BUDGETS),
        "primary_draw_seeds": list(PRIMARY_DRAW_SEEDS),
        "primary_new_fits": 42,
        "optional_expansion_budgets": list(ALLOWED_BUDGETS),
        "optional_expansion_draw_seeds": list(ALLOWED_DRAW_SEEDS),
        "scope_rationale": (
            "Matched scratch fits at k=16 and k=32 across all three frozen "
            "support draws close the attribution gap while quantifying "
            "support-set sensitivity."
        ),
        "epochs": SCRATCH_EPOCHS,
        "patience": SCRATCH_PATIENCE,
        "batch_size": SCRATCH_BATCH_SIZE,
        "learning_rate": SCRATCH_LEARNING_RATE,
        "weight_decay": SCRATCH_WEIGHT_DECAY,
        "precision": PRECISION,
        "initialization_seed_base": INITIALIZATION_SEED_BASE,
        "query_used_for_selection": False,
        "parallelism": "independent single-GPU processes; no DDP",
        "partitions_sha256": EXPECTED_PARTITIONS_SHA256,
    }
    _write_stable_json(protocol_path, protocol)
    metadata_path = output / "study_metadata.json"
    metadata = {
        **{name: str(path) for name, path in inputs.items()},
        "study_id": STUDY_ID,
        "protocol_id": PROTOCOL_ID,
        "output_dir": str(output),
        "metadata_path": str(metadata_path),
        "protocol_path": str(protocol_path),
        "e5_root": str(e5_root),
        "e5_metadata_path": str(e5_metadata_path),
        "e5_metadata_sha256": EXPECTED_E5_METADATA_SHA256,
        "partitions_path": str(partitions_path),
        "partitions_sha256": EXPECTED_PARTITIONS_SHA256,
        "preprocessing_path": str(preprocessing_path),
        "preprocessing_sha256": EXPECTED_PREPROCESSING_SHA256,
        "tensor_dir": str(tensor_dir),
        "tensor_index": str(tensor_index),
        "tensor_index_sha256": e5["tensor_index_sha256"],
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "thread_prefetch": args.thread_prefetch,
        "evaluation_batch_size": args.evaluation_batch_size,
        "gpu_telemetry_interval_ms": args.gpu_telemetry_interval_ms,
        "verbose": args.verbose,
        "bootstrap_replicates": args.bootstrap_replicates,
        "git": _git_revision(),
        "script_snapshot": str(script_snapshot),
        "script_snapshot_sha256": _sha256(script_snapshot),
        "run_e5_snapshot_sha256": _sha256(e5_snapshot),
        "fusion_snapshot_sha256": _sha256(fusion_snapshot),
        "environment_path": str(environment_path),
        "environment_sha256": _sha256(environment_path),
        "input_hashes": expected_hashes,
    }
    metadata["trainable_parameters"] = _validate_matched_architecture(metadata)
    _write_stable_json(metadata_path, metadata)
    return metadata, metadata_path


def _load_metadata(path: Path) -> dict:
    metadata = json.loads(path.read_text())
    if metadata.get("study_id") != STUDY_ID:
        raise ValueError(f"Not E5b metadata: {path}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Review the frozen 21-bundle / 42-fit schedule.
  python scripts/run_e5b.py --dry-run

  # Run the complete control on the RTX PRO 6000 and resume interruptions.
  python scripts/run_e5b.py --resume --jobs-per-device 2

  # Fill a fit subset without exposing query evaluation.
  python scripts/run_e5b.py --stage fit --budgets 16 --draw-seeds 7 --resume
""",
    )
    parser.add_argument(
        "--stage", choices=("fit", "evaluate", "analyze", "all"), default="all"
    )
    parser.add_argument("--e5-root", type=Path, default=DEFAULT_E5_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tensor-dir", type=Path)
    parser.add_argument("--tensor-index", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--source-resolved-config", type=Path)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--vocab", type=Path)
    parser.add_argument("--high-level-cache", type=Path)
    parser.add_argument("--budgets", nargs="+", type=int, default=list(PRIMARY_BUDGETS))
    parser.add_argument(
        "--draw-seeds", nargs="+", type=int, default=list(PRIMARY_DRAW_SEEDS)
    )
    parser.add_argument("--architectures", nargs="+")
    parser.add_argument("--devices", nargs="+", default=["0"])
    parser.add_argument("--jobs-per-device", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--thread-prefetch", action="store_true")
    parser.add_argument("--evaluation-batch-size", type=int, default=16)
    parser.add_argument("--gpu-telemetry-interval-ms", type=int, default=1000)
    parser.add_argument("--verbose", type=int, default=5)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--worker-stage", choices=("fit", "evaluate"), help=argparse.SUPPRESS
    )
    parser.add_argument("--study-metadata", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--architecture", help=argparse.SUPPRESS)
    parser.add_argument("--draw-seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    budgets = tuple(sorted(set(args.budgets)))
    draw_seeds = tuple(args.draw_seeds)
    if args.worker_stage:
        if not args.study_metadata or not args.architecture or args.draw_seed is None:
            parser.error("worker mode requires metadata, architecture, and draw seed")
        metadata = _load_metadata(args.study_metadata.resolve())
        if args.worker_stage == "fit":
            _fit_bundle(metadata, args.architecture, args.draw_seed, budgets, args.resume)
        else:
            _evaluate_bundle(metadata, args.architecture, args.draw_seed, budgets)
        return
    if budgets != tuple(args.budgets) or not set(budgets) <= set(ALLOWED_BUDGETS):
        parser.error("--budgets must be a unique increasing subset of 16 32")
    if not budgets:
        parser.error("--budgets must not be empty")
    if len(draw_seeds) != len(set(draw_seeds)) or not set(draw_seeds) <= set(
        ALLOWED_DRAW_SEEDS
    ):
        parser.error("--draw-seeds must be a unique subset of 7 42 137")
    if args.jobs_per_device < 1 or args.num_workers < 0 or args.prefetch_factor < 1:
        parser.error("invalid worker/concurrency settings")
    if len(args.devices) != len(set(args.devices)):
        parser.error("--devices must not contain duplicates")
    metadata, metadata_path = _resolve_inputs(args)
    partitions = json.loads(Path(metadata["partitions_path"]).read_text())
    all_architectures = sorted(partitions["architectures"])
    architectures = args.architectures or all_architectures
    if not set(architectures) <= set(all_architectures):
        parser.error("unknown E5 architecture ID")
    if args.stage == "all" and set(architectures) != set(all_architectures):
        parser.error("--stage all requires all seven architectures")
    if args.stage == "analyze" and set(architectures) != set(all_architectures):
        parser.error("analysis requires all seven architectures")
    stages = ("fit", "evaluate", "analyze") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage in {"fit", "evaluate"}:
            if stage == "evaluate" and not args.dry_run and not _matrix_complete(
                metadata,
                "fit",
                all_architectures,
                PRIMARY_DRAW_SEEDS,
                PRIMARY_BUDGETS,
            ):
                raise RuntimeError(
                    "E5b query evaluation is sealed until all 42 primary "
                    "scratch fits complete"
                )
            _dispatch(
                stage, metadata, metadata_path, architectures, draw_seeds, budgets,
                args.devices, args.jobs_per_device, args.resume,
                args.fail_fast, args.dry_run,
            )
        elif args.dry_run:
            print(f"Analyze complete E5b evaluations under {metadata['output_dir']}")
        else:
            if not _matrix_complete(
                metadata, "evaluate", architectures, draw_seeds, budgets
            ):
                raise RuntimeError("E5b analysis requires all selected evaluations")
            _analyze(metadata, architectures, draw_seeds, budgets)


if __name__ == "__main__":
    main()
