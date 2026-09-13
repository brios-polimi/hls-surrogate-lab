#!/usr/bin/env python3
"""Run E6 nested exact-architecture-group sample-efficiency curves."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
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
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from hierarchical_cpu_baseline import (
    HurdleRegressor,
    arrays,
    feature_frame,
    feature_sets,
    macro_smape,
    metric_rows,
)
from ll_hls4ml.data.dataset import HeteroGraphDataset
from ll_hls4ml.data.high_level import PROCESSED_FEATURE_DIM
from ll_hls4ml.data.vocab import load_vocab
from ll_hls4ml.io.schema import LABEL_KEYS
from ll_hls4ml.models.high_level import HighLevelLayerGNN
from ll_hls4ml.reporting.accounting import split_sha256
from scripts.run_e2 import (
    FAMILIES,
    OFFICIAL_HIGH_LEVEL_SHA256,
    OFFICIAL_SPLIT_SHA256,
    OFFICIAL_TENSOR_REVISION,
    OFFICIAL_VOCAB_SHA256,
    _gpu_preflight,
    _validate_manifest,
)
from scripts.run_e5 import (
    SCOPES,
    _derived_seed,
    _git_revision,
    _markdown_table,
    _sha256,
    _write_frame_atomic,
    _write_json_atomic,
    _write_stable_json,
    _write_stable_text,
    _write_text_atomic,
)


STUDY_ID = "e6_architecture_group_scaling_v1"
PROTOCOL_ID = "e6_nested_architecture_group_scaling_2026_09_13"
DEFAULT_OUTPUT = _REPO_ROOT / "artifacts/results/e6_architecture_group_scaling_v1"
BUDGETS = (10, 25, 50, 100)
SEEDS = (7, 42, 137)
MODELS = ("h0", "high_level", "extra_trees")
NEURAL_MODELS = ("h0", "high_level")
EXPECTED_TENSOR_INDEX_SHA256 = (
    "1e0e3dc80edc71098e0fd9a05c4683b304e6574944928c804570e795412db226"
)

# Frozen from the E2 optimization contract. E6 changes only training membership
# and the representation; every checkpoint is selected on fixed validation data.
TRAINING = {
    "batch_size": 16,
    "epochs": 400,
    "patience": 20,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "hidden_dim": 64,
    "num_layers": 3,
    "heads": 1,
    "dropout": 0.15,
    "log_huber_delta": 0.35,
    "hurdle_classification_weight": 0.25,
    "gradient_clip_norm": 1.0,
    "lr_scheduler_patience": 8,
    "lr_scheduler_factor": 0.5,
    "min_learning_rate": 1e-6,
    "precision": "bf16",
}


@dataclass
class Job:
    model: str
    seed: int
    budget: int
    stage: str
    run_dir: str
    log_path: str
    device: str | None = None
    status: str = "PENDING"
    returncode: int | None = None


def _write_csv_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    else:
        temporary.write_text("")
    temporary.replace(path)


def _pickle_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _compact_row(row: dict) -> dict:
    fields = (
        "tensor_path",
        "kernel_family",
        "architecture_id",
        "topology_id",
        "archive",
        "architecture_signature_version",
        "topology_signature_version",
    )
    return {field: row[field] for field in fields if field in row}


def _split_membership_sha256(rows: list[dict]) -> str:
    canonical = json.dumps(
        sorted(
            (row["kernel_family"], row["architecture_id"], row["tensor_path"])
            for row in rows
        ),
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _manifest_path(metadata: dict, seed: int, budget: int) -> Path:
    return (
        Path(metadata["output_dir"])
        / "protocol/manifests"
        / f"seed{seed}"
        / f"train_{budget:03d}.json"
    )


def _run_dir(metadata: dict, model: str, seed: int, budget: int) -> Path:
    return Path(metadata["output_dir"]) / "runs" / model / f"seed{seed}" / f"p{budget:03d}"


def _experiment(model: str, seed: int, budget: int) -> str:
    return f"e6_{model}_seed{seed}_p{budget:03d}"


def build_nested_manifests(
    source: dict,
    output_dir: Path,
    seeds=SEEDS,
    budgets=BUDGETS,
) -> tuple[dict, list[dict]]:
    """Create family-stratified, group-safe, nested train memberships."""
    by_family: dict[str, dict[str, list[dict]]] = {}
    architecture_families: dict[str, set[str]] = {}
    for row in source["train"]:
        family = row["kernel_family"]
        architecture = row["architecture_id"]
        by_family.setdefault(family, {}).setdefault(architecture, []).append(row)
        architecture_families.setdefault(architecture, set()).add(family)
    if set(by_family) != FAMILIES:
        raise ValueError("E6 source train split does not contain all seven families")
    cross_family = {
        architecture: families
        for architecture, families in architecture_families.items()
        if len(families) != 1
    }
    if cross_family:
        raise ValueError(
            "E6 cannot family-stratify architecture IDs spanning multiple families"
        )

    fixed_splits = {
        split: [_compact_row(row) for row in source[split]]
        for split in ("validation", "test", "exemplar")
    }
    manifest_index: dict[str, dict[str, dict]] = {}
    audit_rows: list[dict] = []
    for seed in seeds:
        family_orders = {}
        for family, groups in sorted(by_family.items()):
            identifiers = sorted(groups)
            rng = np.random.default_rng(
                _derived_seed(STUDY_ID, "subset", seed, family)
            )
            family_orders[family] = [identifiers[index] for index in rng.permutation(len(identifiers))]
        prior: set[str] = set()
        manifest_index[str(seed)] = {}
        for budget in budgets:
            selected: set[str] = set()
            train_rows = []
            family_counts = {}
            for family, order in sorted(family_orders.items()):
                count = len(order) if budget == 100 else max(1, math.ceil(len(order) * budget / 100))
                chosen = order[:count]
                selected.update(chosen)
                rows = [row for architecture in chosen for row in by_family[family][architecture]]
                train_rows.extend(rows)
                family_counts[family] = {
                    "architectures": len(chosen),
                    "designs": len(rows),
                }
            if not prior <= selected:
                raise AssertionError("E6 training memberships are not nested")
            prior = selected
            train_rows = sorted(train_rows, key=lambda row: row["tensor_path"])
            manifest = {
                "train": [_compact_row(row) for row in train_rows],
                **fixed_splits,
            }
            path = output_dir / "protocol/manifests" / f"seed{seed}" / f"train_{budget:03d}.json"
            _write_stable_json(path, manifest)
            entry = {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "split_sha256": split_sha256(manifest),
                "train_membership_sha256": _split_membership_sha256(manifest["train"]),
                "train_designs": len(manifest["train"]),
                "train_architectures": len(selected),
                "family_counts": family_counts,
            }
            manifest_index[str(seed)][str(budget)] = entry
            for family, counts in family_counts.items():
                audit_rows.append({
                    "seed": seed,
                    "budget_percent": budget,
                    "kernel_family": family,
                    **counts,
                })
        if len(prior) != sum(len(groups) for groups in by_family.values()):
            raise AssertionError("E6 100% membership is incomplete")
    return manifest_index, audit_rows


def _find_run(roots: list[Path], experiment: str) -> Path | None:
    matches = []
    for root in roots:
        matches.extend(
            path.parent
            for path in root.rglob(f"{experiment}/resolved_config.json")
        )
    unique = sorted(set(path.resolve() for path in matches))
    if len(unique) > 1:
        raise ValueError(
            f"Multiple {experiment} runs found; pass narrower --e2-results-root values: {unique}"
        )
    return unique[0] if unique else None


def _validate_endpoint(
    run_dir: Path,
    model: str,
    seed: int,
    source_manifest: dict,
    require_checkpoint: bool,
) -> dict:
    resolved_path = run_dir / "resolved_config.json"
    predictions_path = run_dir / "predictions.csv"
    summary_path = run_dir / ("summary.json" if model == "h0" else "summary.csv")
    experiment = (
        f"e2_h0_structural_seed{seed}"
        if model == "h0"
        else f"e2_extra_trees_structural_seed{seed}"
    )
    for path in (resolved_path, predictions_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    resolved = json.loads(resolved_path.read_text())
    expected_model = "hierarchical" if model == "h0" else "extra_trees"
    allowed_experiments = {experiment}
    if model == "extra_trees":
        allowed_experiments.add(f"e2_extra_trees_seed{seed}")
    expected = {
        "model": expected_model,
        "seed": seed,
        "split_sha256": OFFICIAL_SPLIT_SHA256,
    }
    mismatch = {
        key: (resolved.get(key), value)
        for key, value in expected.items()
        if resolved.get(key) != value
    }
    if mismatch:
        raise ValueError(f"Incompatible E2 endpoint {run_dir}: {mismatch}")
    if resolved.get("experiment_name") not in allowed_experiments:
        raise ValueError(
            f"Incompatible E2 endpoint experiment identity: {run_dir}"
        )
    if model == "h0":
        training_identity = {
            "batch_size": TRAINING["batch_size"],
            "epochs": TRAINING["epochs"],
            "patience": TRAINING["patience"],
            "learning_rate": TRAINING["learning_rate"],
            "weight_decay": TRAINING["weight_decay"],
            "hidden_dim": TRAINING["hidden_dim"],
            "num_layers": TRAINING["num_layers"],
            "dropout": TRAINING["dropout"],
            "loss": "log_huber_hurdle",
            "log_huber_delta": TRAINING["log_huber_delta"],
            "hurdle_classification_weight": TRAINING[
                "hurdle_classification_weight"
            ],
            "precision": TRAINING["precision"],
            "use_global_features": True,
            "use_context": True,
            "context_mode": "core",
            "split_heads": True,
            "hurdle_heads": True,
            "hurdle_prediction_mode": "threshold",
        }
        training_mismatch = {
            key: (resolved.get(key), value)
            for key, value in training_identity.items()
            if resolved.get(key) != value
        }
        if training_mismatch:
            raise ValueError(
                f"Full-data H0 endpoint is not optimization-matched: {training_mismatch}"
            )
    elif resolved.get("feature_set") != "core_context":
        raise ValueError(f"E2 ExtraTrees endpoint is not core_context: {run_dir}")

    predictions = pd.read_csv(predictions_path)
    test = predictions[predictions["split"] == "test"]
    expected_paths = [row["tensor_path"] for row in source_manifest["test"]]
    if len(test) != len(expected_paths) or set(test["tensor_path"]) != set(expected_paths):
        raise ValueError(f"E2 endpoint test membership differs: {run_dir}")
    if test["tensor_path"].duplicated().any():
        raise ValueError(f"Duplicate E2 endpoint predictions: {run_dir}")
    expected_targets = {
        row["tensor_path"]: np.asarray(row["labels"], dtype=float)
        for row in source_manifest["test"]
    }
    ordered = test.set_index("tensor_path").loc[expected_paths]
    observed_targets = ordered[
        [f"target_{target}" for target in LABEL_KEYS]
    ].to_numpy(float)
    wanted_targets = np.stack([expected_targets[path] for path in expected_paths])
    if not np.allclose(observed_targets, wanted_targets, rtol=0, atol=1e-6):
        raise ValueError(f"E2 endpoint targets differ from frozen manifest: {run_dir}")

    checkpoint = None
    if require_checkpoint:
        checkpoint = run_dir / "checkpoints" / f"{experiment}_checkpoint.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if "model" not in payload:
            raise ValueError(f"Incomplete E2 H0 checkpoint: {checkpoint}")
    return {
        "run_dir": str(run_dir),
        "resolved_config": str(resolved_path),
        "resolved_config_sha256": _sha256(resolved_path),
        "predictions": str(predictions_path),
        "predictions_sha256": _sha256(predictions_path),
        "summary": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": _sha256(checkpoint) if checkpoint else None,
    }


def _discover_endpoints(roots: list[Path], source_manifest: dict) -> dict:
    endpoints = {"h0": {}, "extra_trees": {}}
    for seed in SEEDS:
        h0_experiment = f"e2_h0_structural_seed{seed}"
        h0_run = _find_run(roots, h0_experiment)
        if h0_run is None:
            raise FileNotFoundError(
                f"Could not discover required full-data H0 run {h0_experiment}"
            )
        endpoints["h0"][str(seed)] = _validate_endpoint(
            h0_run, "h0", seed, source_manifest, require_checkpoint=True
        )
        extra_experiment = f"e2_extra_trees_structural_seed{seed}"
        extra_run = _find_run(roots, extra_experiment)
        if extra_run is not None:
            endpoints["extra_trees"][str(seed)] = _validate_endpoint(
                extra_run,
                "extra_trees",
                seed,
                source_manifest,
                require_checkpoint=False,
            )
    return endpoints


def _feature_cache_fingerprint(manifest: dict) -> str:
    paths = {
        row["tensor_path"]
        for rows in manifest.values()
        for row in rows
    }
    return hashlib.sha256("\n".join(sorted(paths)).encode()).hexdigest()


def _choose_feature_cache(args, source_manifest: dict) -> Path:
    expected = _feature_cache_fingerprint(source_manifest)
    if args.extra_trees_feature_cache:
        candidates = [args.extra_trees_feature_cache.resolve()]
    else:
        candidates = [
            path.resolve()
            for root in args.e2_results_root
            for path in root.rglob("feature_cache/e2_core_context.pkl")
        ]
    compatible = []
    for path in candidates:
        fingerprint_path = path.with_suffix(".sha256")
        if path.is_file() and fingerprint_path.is_file():
            if fingerprint_path.read_text().strip() == expected:
                compatible.append(path)
    if len(set(compatible)) > 1:
        raise ValueError(f"Multiple compatible ExtraTrees caches found: {compatible}")
    if compatible:
        return compatible[0]
    if args.extra_trees_feature_cache:
        return args.extra_trees_feature_cache.resolve()
    return (args.output_dir / "cache/e6_core_context.pkl").resolve()


def _validate_model_scaffolds(endpoints: dict) -> dict:
    y_means = torch.zeros(len(LABEL_KEYS))
    y_stds = torch.ones(len(LABEL_KEYS))
    h0_counts = {
        json.loads(Path(record["resolved_config"]).read_text()).get("parameter_count")
        for record in endpoints["h0"].values()
    }
    if len(h0_counts) != 1 or None in h0_counts:
        raise ValueError("Full-data E2 H0 endpoints disagree on parameter count")
    h0_parameters = h0_counts.pop()
    high_level = HighLevelLayerGNN(
        input_dim=PROCESSED_FEATURE_DIM,
        y_means=y_means,
        y_stds=y_stds,
        hidden_dim=TRAINING["hidden_dim"],
        num_layers=TRAINING["num_layers"],
        heads=TRAINING["heads"],
        dropout=TRAINING["dropout"],
        encoder="gatv2",
        hurdle_heads=True,
        hurdle_prediction_mode="threshold",
    )
    high_level_parameters = sum(p.numel() for p in high_level.parameters())
    return {
        "h0_parameters": h0_parameters,
        "high_level_parameters": high_level_parameters,
        "h0_to_high_level_parameter_ratio": h0_parameters / high_level_parameters,
        "parameter_matching": False,
    }


def _resolve_study(args) -> tuple[dict, Path]:
    args.tensor_dir = args.tensor_dir.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.high_level_cache = args.high_level_cache.expanduser().resolve()
    args.e2_results_root = [path.expanduser().resolve() for path in args.e2_results_root]
    args.vocab = (args.vocab or args.tensor_dir / "vocab.json").expanduser().resolve()
    args.tensor_index = (
        args.tensor_index or args.tensor_dir / "labels.json"
    ).expanduser().resolve()
    for path in (
        args.tensor_dir,
        args.manifest,
        args.high_level_cache,
        args.vocab,
        args.tensor_index,
        *args.e2_results_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if _sha256(args.vocab) != args.expected_vocab_sha256:
        raise ValueError("Frozen vocabulary hash mismatch")
    if _sha256(args.high_level_cache) != args.expected_high_level_sha256:
        raise ValueError("Frozen high-level cache hash mismatch")
    if _sha256(args.tensor_index) != args.expected_tensor_index_sha256:
        raise ValueError("Frozen tensor index hash mismatch")
    source_manifest, _, source_split_hash = _validate_manifest(
        args.manifest, args.expected_split_sha256
    )
    endpoints = _discover_endpoints(args.e2_results_root, source_manifest)
    feature_cache_path = _choose_feature_cache(args, source_manifest)

    provenance = args.output_dir / "provenance"
    snapshots = {
        "run_e6.py": Path(__file__).resolve(),
        "train.py": _REPO_ROOT / "scripts/train.py",
        "run_e2.py": _REPO_ROOT / "scripts/run_e2.py",
        "high_level.py": _REPO_ROOT / "src/ll_hls4ml/models/high_level.py",
        "data_high_level.py": _REPO_ROOT / "src/ll_hls4ml/data/high_level.py",
        "hierarchical.py": _REPO_ROOT / "src/ll_hls4ml/models/hierarchical.py",
        "training_loops.py": _REPO_ROOT / "src/ll_hls4ml/training/loops.py",
    }
    snapshot_hashes = {}
    for name, source in snapshots.items():
        target = provenance / name
        _write_stable_text(target, source.read_text())
        snapshot_hashes[name] = _sha256(target)
    environment_path = provenance / "environment.json"
    _write_stable_json(environment_path, {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    })

    manifests, audit_rows = build_nested_manifests(
        source_manifest, args.output_dir
    )
    _write_frame_atomic(
        args.output_dir / "protocol/subset_family_audit.csv",
        pd.DataFrame(audit_rows),
    )
    protocol_path = args.output_dir / "protocol/e6_protocol.json"
    protocol = {
        "study_id": STUDY_ID,
        "protocol_id": PROTOCOL_ID,
        "frozen_on": "2026-09-13",
        "status": "late-added after E2 test inspection",
        "estimand": (
            "performance on unchanged unseen exact architectures as labeled "
            "source architecture-group coverage grows"
        ),
        "budgets_percent_of_groups_within_family": list(BUDGETS),
        "replication_seeds": list(SEEDS),
        "seed_role": (
            "each seed jointly fixes a nested group draw and model/ExtraTrees randomness"
        ),
        "models": list(MODELS),
        "primary_metric": "equal-test-architecture mean of per-design six-target SMAPE",
        "outer_unit": "exact test architecture_id",
        "selection": "fixed validation split only",
        "evaluation": "unchanged E2 test split after all primary fits complete",
        "preprocessing": "same pipelines, statistics fit only on selected training rows",
        "h0_100_percent": "reuse verified E2 checkpoints and predictions",
        "subset_sampling": "nested and family-stratified at exact architecture_id level",
        "x_axis": "exact labeled design count; log scale",
        "parameter_matched": False,
        "claim_limits": [
            "not configuration density at fixed architecture diversity",
            "not an intrinsic data-hunger law",
            "representation, side-information, and parameter count are not causally separated",
            "three whole-pipeline seeds do not separate subset and optimizer variance",
        ],
        "source_split_sha256": source_split_hash,
        "training": TRAINING,
    }
    _write_stable_json(protocol_path, protocol)
    metadata_path = args.output_dir / "study_metadata.json"
    metadata = {
        "study_id": STUDY_ID,
        "protocol_id": PROTOCOL_ID,
        "output_dir": str(args.output_dir),
        "metadata_path": str(metadata_path),
        "protocol_path": str(protocol_path),
        "tensor_dir": str(args.tensor_dir),
        "tensor_index": str(args.tensor_index),
        "tensor_index_sha256": _sha256(args.tensor_index),
        "tensor_source_revision": OFFICIAL_TENSOR_REVISION,
        "vocab": str(args.vocab),
        "vocab_sha256": _sha256(args.vocab),
        "high_level_cache": str(args.high_level_cache),
        "high_level_cache_sha256": _sha256(args.high_level_cache),
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": _sha256(args.manifest),
        "source_split_sha256": source_split_hash,
        "validation_membership_sha256": _split_membership_sha256(source_manifest["validation"]),
        "test_membership_sha256": _split_membership_sha256(source_manifest["test"]),
        "e2_results_roots": [str(path) for path in args.e2_results_root],
        "full_data_endpoints": endpoints,
        "extra_trees_feature_cache": str(feature_cache_path),
        "extra_trees_feature_fingerprint": _feature_cache_fingerprint(source_manifest),
        "manifests": manifests,
        "models": _validate_model_scaffolds(endpoints),
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "thread_prefetch": args.thread_prefetch,
        "gpu_telemetry_interval_ms": args.gpu_telemetry_interval_ms,
        "bootstrap_replicates": args.bootstrap_replicates,
        "git": _git_revision(),
        "snapshots": snapshot_hashes,
        "environment_path": str(environment_path),
        "environment_sha256": _sha256(environment_path),
    }
    _write_stable_json(metadata_path, metadata)
    return metadata, metadata_path


def _load_metadata(path: Path) -> dict:
    metadata = json.loads(path.read_text())
    if metadata.get("study_id") != STUDY_ID:
        raise ValueError(f"Not E6 metadata: {path}")
    return metadata


def _neural_config(metadata: dict, model: str, seed: int, budget: int) -> dict:
    run_dir = _run_dir(metadata, model, seed, budget)
    experiment = _experiment(model, seed, budget)
    model_name = "hierarchical" if model == "h0" else "high_level_layer_gnn"
    config = {
        "study_id": STUDY_ID,
        "protocol_id": PROTOCOL_ID,
        "experiment_name": experiment,
        "model": model_name,
        "representation": (
            "hierarchical LLVM-CDFG H0 with global/context features"
            if model == "h0"
            else "wa-hls4ml layer/config graph"
        ),
        "budget_percent": budget,
        "replication_seed": seed,
        "train_subset_seed_scheme": (
            "sha256(study_id, 'subset', replication_seed, kernel_family)"
        ),
        "seed": seed,
        "tensor_dir": metadata["tensor_dir"],
        "tensor_source_revision": metadata["tensor_source_revision"],
        "vocab_path": metadata["vocab"],
        "split_manifest_path": str(_manifest_path(metadata, seed, budget)),
        "require_complete_split_manifest": True,
        "results_dir": str(run_dir.parent),
        "checkpoint_dir": str(run_dir / "checkpoints"),
        "kernel_types": sorted(FAMILIES),
        "family_balanced_sampling": False,
        "fit_only": True,
        "evaluation_splits": ["validation", "test"],
        "optimizer": "adamw",
        "early_stopping_metric": "smape",
        "loss": "log_huber_hurdle",
        "split_heads": True,
        "hurdle_heads": True,
        "hurdle_prediction_mode": "threshold",
        "high_level_encoder": "gatv2",
        "num_workers": metadata["num_workers"],
        "pin_memory": True,
        "prefetch_factor": metadata["prefetch_factor"],
        "thread_prefetch": metadata["thread_prefetch"],
        "checkpoint_interval": 1,
        "gpu_telemetry_interval_ms": metadata["gpu_telemetry_interval_ms"],
        "verbose": 5,
        **TRAINING,
    }
    if model == "h0":
        config.update(
            use_global_features=True,
            use_context=True,
            context_mode="core",
        )
    else:
        config["high_level_cache"] = metadata["high_level_cache"]
    return config


def _config_path(metadata: dict, model: str, seed: int, budget: int) -> Path:
    return (
        Path(metadata["output_dir"])
        / "configs"
        / model
        / f"seed{seed}_p{budget:03d}.json"
    )


def _materialize_configs(metadata: dict) -> None:
    for model in NEURAL_MODELS:
        for seed in SEEDS:
            for budget in BUDGETS:
                if model == "h0" and budget == 100:
                    continue
                _write_stable_json(
                    _config_path(metadata, model, seed, budget),
                    _neural_config(metadata, model, seed, budget),
                )


def _neural_fit_complete(metadata: dict, model: str, seed: int, budget: int) -> bool:
    run_dir = _run_dir(metadata, model, seed, budget)
    experiment = _experiment(model, seed, budget)
    summary_path = run_dir / "fit_summary.json"
    checkpoint = run_dir / "checkpoints" / f"{experiment}_checkpoint.pt"
    resolved_path = run_dir / "resolved_config.json"
    if not all(path.is_file() for path in (summary_path, checkpoint, resolved_path)):
        return False
    try:
        summary = json.loads(summary_path.read_text())
        resolved = json.loads(resolved_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    expected_model = "hierarchical" if model == "h0" else "high_level_layer_gnn"
    expected = {
        "study_id": STUDY_ID,
        "protocol_id": PROTOCOL_ID,
        "model": expected_model,
        "seed": seed,
        "budget_percent": budget,
        "split_sha256": metadata["manifests"][str(seed)][str(budget)]["split_sha256"],
    }
    if any(resolved.get(key) != value for key, value in expected.items()):
        return False
    return summary.get("checkpoint_sha256") == _sha256(checkpoint)


def _neural_evaluation_complete(
    metadata: dict, model: str, seed: int, budget: int
) -> bool:
    run_dir = _run_dir(metadata, model, seed, budget)
    marker = run_dir / "evaluation_completion.json"
    predictions = run_dir / "predictions.csv"
    summary = run_dir / "summary.json"
    if not all(path.is_file() for path in (marker, predictions, summary)):
        return False
    try:
        record = json.loads(marker.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return (
        record.get("predictions_sha256") == _sha256(predictions)
        and record.get("summary_sha256") == _sha256(summary)
    )


def _finalize_neural_evaluation(metadata: dict, model: str, seed: int, budget: int) -> None:
    run_dir = _run_dir(metadata, model, seed, budget)
    predictions = run_dir / "predictions.csv"
    summary = run_dir / "summary.json"
    resolved = run_dir / "resolved_config.json"
    if not all(path.is_file() for path in (predictions, summary, resolved)):
        raise RuntimeError(f"Incomplete neural evaluation: {run_dir}")
    frame = pd.read_csv(predictions)
    expected_test = json.loads(_manifest_path(metadata, seed, budget).read_text())["test"]
    test = frame[frame["split"] == "test"]
    if len(test) != len(expected_test) or set(test["tensor_path"]) != {
        row["tensor_path"] for row in expected_test
    }:
        raise RuntimeError(f"Neural evaluation has wrong test membership: {run_dir}")
    _write_json_atomic(run_dir / "evaluation_completion.json", {
        "status": "complete",
        "predictions_sha256": _sha256(predictions),
        "summary_sha256": _sha256(summary),
        "resolved_config_sha256": _sha256(resolved),
    })


def _extra_fit_complete(metadata: dict, seed: int, budget: int) -> bool:
    run_dir = _run_dir(metadata, "extra_trees", seed, budget)
    model_path = run_dir / "model.pkl"
    marker = run_dir / "fit_summary.json"
    if not model_path.is_file() or not marker.is_file():
        return False
    try:
        summary = json.loads(marker.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return summary.get("model_sha256") == _sha256(model_path)


def _extra_evaluation_complete(metadata: dict, seed: int, budget: int) -> bool:
    run_dir = _run_dir(metadata, "extra_trees", seed, budget)
    predictions = run_dir / "predictions.csv"
    marker = run_dir / "evaluation_summary.json"
    if not predictions.is_file() or not marker.is_file():
        return False
    try:
        summary = json.loads(marker.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return summary.get("predictions_sha256") == _sha256(predictions)


def _endpoint_available(metadata: dict, model: str, seed: int) -> bool:
    return str(seed) in metadata["full_data_endpoints"].get(model, {})


def _fit_complete(metadata: dict, model: str, seed: int, budget: int) -> bool:
    if budget == 100 and _endpoint_available(metadata, model, seed):
        return True
    if model in NEURAL_MODELS:
        return _neural_fit_complete(metadata, model, seed, budget)
    return _extra_fit_complete(metadata, seed, budget)


def _evaluation_complete(metadata: dict, model: str, seed: int, budget: int) -> bool:
    if budget == 100 and _endpoint_available(metadata, model, seed):
        return True
    if model in NEURAL_MODELS:
        return _neural_evaluation_complete(metadata, model, seed, budget)
    return _extra_evaluation_complete(metadata, seed, budget)


def _ensure_feature_cache(metadata: dict) -> None:
    cache_path = Path(metadata["extra_trees_feature_cache"])
    fingerprint_path = cache_path.with_suffix(".sha256")
    expected = metadata["extra_trees_feature_fingerprint"]
    if cache_path.is_file() and fingerprint_path.is_file():
        if fingerprint_path.read_text().strip() == expected:
            print(f"Using verified ExtraTrees feature cache: {cache_path}")
            return
        raise ValueError(f"ExtraTrees feature-cache fingerprint mismatch: {cache_path}")
    manifest = json.loads(Path(metadata["source_manifest"]).read_text())
    needed = {
        row["tensor_path"]
        for rows in manifest.values()
        for row in rows
    }
    dataset = HeteroGraphDataset(
        metadata["tensor_dir"],
        types=sorted(FAMILIES | {"exemplar"}),
        silent=False,
        relative_paths=sorted(needed),
    )
    vocabulary, _, _ = load_vocab(metadata["vocab"])
    feature_frame(dataset, needed, len(vocabulary), cache_path)
    if not cache_path.is_file() or fingerprint_path.read_text().strip() != expected:
        raise RuntimeError("ExtraTrees feature-cache creation did not verify")


def _extra_fit(metadata: dict, seed: int, budget: int) -> None:
    if _extra_fit_complete(metadata, seed, budget):
        return
    run_dir = _run_dir(metadata, "extra_trees", seed, budget)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _manifest_path(metadata, seed, budget)
    manifest = json.loads(manifest_path.read_text())
    frame = pd.read_pickle(metadata["extra_trees_feature_cache"])
    indexed = frame.set_index("tensor_path")
    columns = feature_sets(frame)["core_context"]
    train_paths = [row["tensor_path"] for row in manifest["train"]]
    validation_paths = [row["tensor_path"] for row in manifest["validation"]]
    train_x, train_y = arrays(indexed, train_paths, columns)
    validation_x, validation_y = arrays(indexed, validation_paths, columns)
    started = time.perf_counter()
    model = HurdleRegressor("extra_trees", seed).fit(train_x, train_y)
    validation_predictions = model.predict_modes(validation_x)
    validation_scores = {
        mode: macro_smape(prediction, validation_y)
        for mode, prediction in validation_predictions.items()
    }
    selected_mode = min(validation_scores, key=validation_scores.get)
    model_path = run_dir / "model.pkl"
    _pickle_atomic(model_path, {
        "model": model,
        "selected_hurdle_mode": selected_mode,
        "feature_columns": columns,
    })
    _write_json_atomic(run_dir / "fit_summary.json", {
        "status": "complete",
        "study_id": STUDY_ID,
        "protocol_id": PROTOCOL_ID,
        "model": "extra_trees",
        "seed": seed,
        "budget_percent": budget,
        "train_designs": len(train_paths),
        "train_architectures": metadata["manifests"][str(seed)][str(budget)][
            "train_architectures"
        ],
        "feature_count": len(columns),
        "validation_hurdle_mode_smape": validation_scores,
        "selected_hurdle_mode": selected_mode,
        "wall_seconds": time.perf_counter() - started,
        "model_sha256": _sha256(model_path),
        "manifest_sha256": _sha256(manifest_path),
    })


def _extra_evaluate(metadata: dict, seed: int, budget: int) -> None:
    if _extra_evaluation_complete(metadata, seed, budget):
        return
    run_dir = _run_dir(metadata, "extra_trees", seed, budget)
    if not _extra_fit_complete(metadata, seed, budget):
        raise RuntimeError(f"Incomplete ExtraTrees fit: {run_dir}")
    with (run_dir / "model.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    model = payload["model"]
    mode = payload["selected_hurdle_mode"]
    columns = payload["feature_columns"]
    manifest = json.loads(_manifest_path(metadata, seed, budget).read_text())
    frame = pd.read_pickle(metadata["extra_trees_feature_cache"])
    indexed = frame.set_index("tensor_path")
    rows = []
    metrics = []
    started = time.perf_counter()
    for split in ("validation", "test"):
        split_rows = manifest[split]
        paths = [row["tensor_path"] for row in split_rows]
        x, target = arrays(indexed, paths, columns)
        prediction = model.predict_modes(x)[mode]
        families = indexed.loc[paths, "kernel_family"].to_numpy(str)
        metrics.extend(metric_rows(
            _experiment("extra_trees", seed, budget),
            "extra_trees",
            "core_context",
            len(manifest["train"]),
            split,
            families,
            prediction,
            target,
        ))
        for position, source in enumerate(split_rows):
            row = {
                "study_id": STUDY_ID,
                "model": "extra_trees",
                "seed": seed,
                "budget_percent": budget,
                "selected_hurdle_mode": mode,
                "split": split,
                **source,
            }
            for target_index, name in enumerate(LABEL_KEYS):
                row[f"target_{name}"] = float(target[position, target_index])
                row[f"prediction_{name}"] = float(prediction[position, target_index])
            rows.append(row)
    prediction_path = run_dir / "predictions.csv"
    _write_csv_atomic(prediction_path, rows)
    _write_csv_atomic(run_dir / "metrics.csv", metrics)
    _write_json_atomic(run_dir / "evaluation_summary.json", {
        "status": "complete",
        "seed": seed,
        "budget_percent": budget,
        "selected_hurdle_mode": mode,
        "validation_samples": len(manifest["validation"]),
        "test_samples": len(manifest["test"]),
        "wall_seconds": time.perf_counter() - started,
        "predictions_sha256": _sha256(prediction_path),
    })


def _worker_command(
    metadata: dict,
    metadata_path: Path,
    job: Job,
    resume: bool,
) -> list[str]:
    if job.model == "extra_trees":
        return [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-stage",
            "extra-fit" if job.stage == "fit" else "extra-evaluate",
            "--study-metadata",
            str(metadata_path),
            "--worker-seed",
            str(job.seed),
            "--worker-budget",
            str(job.budget),
        ]
    config_path = _config_path(metadata, job.model, job.seed, job.budget)
    command = [
        sys.executable,
        str(_REPO_ROOT / "scripts/train.py"),
        "--config",
        str(config_path),
    ]
    if job.stage == "evaluate":
        checkpoint = (
            Path(job.run_dir)
            / "checkpoints"
            / f"{_experiment(job.model, job.seed, job.budget)}_checkpoint.pt"
        )
        command.extend(["--evaluate-checkpoint", str(checkpoint)])
    elif resume:
        backup = (
            Path(job.run_dir)
            / "checkpoints"
            / f"{_experiment(job.model, job.seed, job.budget)}_backup.pt"
        )
        if backup.is_file():
            resume_config = json.loads(config_path.read_text())
            resume_config["resume_checkpoint_path"] = str(backup)
            invocation = (
                Path(job.run_dir).parents[3]
                / "configs/resume"
                / f"{job.model}_seed{job.seed}_p{job.budget:03d}_{time.time_ns()}.json"
            )
            _write_json_atomic(invocation, resume_config)
            command[-1] = str(invocation)
    return command


def _selected_jobs(
    metadata: dict,
    stage: str,
    models,
    seeds,
    budgets,
    devices,
    jobs_per_device: int,
    resume: bool,
    dry_run: bool,
) -> list[Job]:
    slots = [device for device in devices for _ in range(jobs_per_device)]
    neural_position = 0
    jobs = []
    for model in models:
        for seed in seeds:
            for budget in budgets:
                run_dir = _run_dir(metadata, model, seed, budget)
                device = None
                if model in NEURAL_MODELS:
                    device = slots[neural_position % len(slots)]
                    neural_position += 1
                job = Job(
                    model=model,
                    seed=seed,
                    budget=budget,
                    stage=stage,
                    run_dir=str(run_dir),
                    log_path=str(
                        Path(metadata["output_dir"])
                        / "logs"
                        / stage
                        / f"{model}_seed{seed}_p{budget:03d}.log"
                    ),
                    device=device,
                )
                complete = (
                    _fit_complete(metadata, model, seed, budget)
                    if stage == "fit"
                    else _evaluation_complete(metadata, model, seed, budget)
                )
                if complete:
                    job.status = (
                        "REUSED_FULL_ENDPOINT"
                        if budget == 100 and _endpoint_available(metadata, model, seed)
                        else "SKIPPED_COMPLETE"
                    )
                elif stage == "evaluate" and not dry_run and not _fit_complete(
                    metadata, model, seed, budget
                ):
                    job.status = "BLOCKED_INCOMPLETE_FIT"
                elif stage == "fit" and run_dir.is_dir() and any(run_dir.iterdir()):
                    if model == "extra_trees":
                        job.status = "PENDING" if resume else "BLOCKED_PARTIAL_USE_RESUME"
                    else:
                        backup = (
                            run_dir
                            / "checkpoints"
                            / f"{_experiment(model, seed, budget)}_backup.pt"
                        )
                        if not resume:
                            job.status = "BLOCKED_PARTIAL_USE_RESUME"
                        elif not backup.is_file():
                            job.status = "BLOCKED_NO_BACKUP_CHECKPOINT"
                jobs.append(job)
    return jobs


def _persist_index(path: Path, payload: dict, jobs: list[Job]) -> None:
    _write_json_atomic(path, {**payload, "jobs": [asdict(job) for job in jobs]})


def _finish_job(metadata: dict, process, job: Job) -> bool:
    job.returncode = process.returncode
    valid = False
    if process.returncode == 0:
        if job.stage == "evaluate" and job.model in NEURAL_MODELS:
            try:
                _finalize_neural_evaluation(
                    metadata, job.model, job.seed, job.budget
                )
            except Exception as error:
                print(f"Evaluation finalization failed for {job.run_dir}: {error}", file=sys.stderr)
            valid = _neural_evaluation_complete(
                metadata, job.model, job.seed, job.budget
            )
        else:
            valid = (
                _fit_complete(metadata, job.model, job.seed, job.budget)
                if job.stage == "fit"
                else _evaluation_complete(metadata, job.model, job.seed, job.budget)
            )
    job.status = "COMPLETE" if valid else "FAILED"
    return not valid


def _dispatch(
    metadata: dict,
    metadata_path: Path,
    stage: str,
    models,
    seeds,
    budgets,
    devices,
    jobs_per_device: int,
    resume: bool,
    fail_fast: bool,
    dry_run: bool,
) -> None:
    jobs = _selected_jobs(
        metadata,
        stage,
        models,
        seeds,
        budgets,
        devices,
        jobs_per_device,
        resume,
        dry_run,
    )
    output = Path(metadata["output_dir"])
    index_path = output / f"{stage}_index.json"
    invocation_path = (
        output
        / "logs/invocations"
        / f"{time.strftime('%Y%m%dT%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}_{stage}.json"
    )
    payload = {
        "stage": stage,
        "models": list(models),
        "seeds": list(seeds),
        "budgets": list(budgets),
        "devices": list(devices),
        "jobs_per_device": jobs_per_device,
        "resume": resume,
        "dry_run": dry_run,
    }

    def persist() -> None:
        _persist_index(index_path, payload, jobs)
        _persist_index(invocation_path, payload, jobs)

    persist()
    blocked = [job for job in jobs if job.status.startswith("BLOCKED")]
    if blocked:
        for job in blocked:
            print(f"{job.status}: {job.model} seed={job.seed} p={job.budget}")
        raise SystemExit(2)
    pending = [job for job in jobs if job.status == "PENDING"]
    if dry_run:
        for job in pending:
            prefix = (
                f"CUDA_VISIBLE_DEVICES={shlex.quote(str(job.device))} "
                if job.model in NEURAL_MODELS
                else ""
            )
            print(
                prefix
                + shlex.join(
                    _worker_command(metadata, metadata_path, job, resume)
                )
            )
        reused = sum(job.status == "REUSED_FULL_ENDPOINT" for job in jobs)
        print(
            f"E6 {stage}: runnable={len(pending)} reused={reused} index={index_path}"
        )
        return
    if not pending:
        print(f"All selected E6 {stage} jobs are complete.")
        return

    neural = [job for job in pending if job.model in NEURAL_MODELS]
    extra = [job for job in pending if job.model == "extra_trees"]
    for device in dict.fromkeys(job.device for job in neural):
        _gpu_preflight(str(device))
    available = [device for device in devices for _ in range(jobs_per_device)]
    active = []
    failed = False
    try:
        while neural or active:
            while neural and available and not (failed and fail_fast):
                job = neural.pop(0)
                device = available.pop(0)
                job.device = device
                command = _worker_command(metadata, metadata_path, job, resume)
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
                environment["LL_HLS4ML_TQDM"] = "0"
                process = subprocess.Popen(
                    command,
                    cwd=_REPO_ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                job.status = "RUNNING"
                active.append((process, job, handle))
                print(
                    f"Started E6 {stage} {job.model} seed={job.seed} "
                    f"p={job.budget} on GPU {device}; log={job.log_path}",
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
                available.append(str(job.device))
                process.returncode = returncode
                failed = _finish_job(metadata, process, job) or failed
                print(f"{job.status}: {job.model} seed={job.seed} p={job.budget}")
                persist()
            active = remaining

        if failed and fail_fast:
            for job in neural:
                job.status = "SKIPPED_FAIL_FAST"
            persist()
        for job in extra:
            if failed and fail_fast:
                job.status = "SKIPPED_FAIL_FAST"
                continue
            command = _worker_command(metadata, metadata_path, job, resume)
            log_path = Path(job.log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a", buffering=1)
            job.status = "RUNNING"
            persist()
            process = subprocess.Popen(
                command,
                cwd=_REPO_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            print(
                f"Started E6 {stage} ExtraTrees seed={job.seed} p={job.budget}; "
                f"log={job.log_path}",
                flush=True,
            )
            try:
                returncode = process.wait()
            except KeyboardInterrupt:
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
            handle.close()
            process.returncode = returncode
            failed = _finish_job(metadata, process, job) or failed
            print(f"{job.status}: extra_trees seed={job.seed} p={job.budget}")
            persist()
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


def _prediction_path(metadata: dict, model: str, seed: int, budget: int) -> Path:
    if budget == 100 and _endpoint_available(metadata, model, seed):
        return Path(
            metadata["full_data_endpoints"][model][str(seed)]["predictions"]
        )
    return _run_dir(metadata, model, seed, budget) / "predictions.csv"


def _verified_predictions(
    metadata: dict, model: str, seed: int, budget: int
) -> pd.DataFrame:
    path = _prediction_path(metadata, model, seed, budget)
    if budget == 100 and _endpoint_available(metadata, model, seed):
        expected = metadata["full_data_endpoints"][model][str(seed)][
            "predictions_sha256"
        ]
        if _sha256(path) != expected:
            raise RuntimeError(f"Reused endpoint prediction hash changed: {path}")
    elif not _evaluation_complete(metadata, model, seed, budget):
        raise RuntimeError(f"Incomplete E6 predictions: {path}")
    frame = pd.read_csv(path)
    frame = frame[frame["split"].isin(["validation", "test"])].copy()
    if frame.empty or frame["tensor_path"].duplicated().any():
        raise ValueError(f"Invalid prediction rows: {path}")
    return frame


def _error_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame[["split", "tensor_path", "kernel_family", "architecture_id"]].copy()
    errors = []
    for target in LABEL_KEYS:
        truth = frame[f"target_{target}"].to_numpy(float)
        prediction = frame[f"prediction_{target}"].to_numpy(float)
        error = 200 * np.abs(truth - prediction) / (
            np.abs(truth) + np.abs(prediction) + 1.0
        )
        output[f"smape_{target}"] = error
        errors.append(error)
    matrix = np.asarray(errors).T
    for scope, positions in SCOPES.items():
        output[f"smape_{scope}"] = matrix[:, positions].mean(axis=1)
    return output


def _cluster_interval(
    values: pd.Series,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    array = values.to_numpy(float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates)
    for start in range(0, replicates, 1000):
        stop = min(start + 1000, replicates)
        indices = rng.integers(0, len(array), size=(stop - start, len(array)))
        estimates[start:stop] = array[indices].mean(axis=1)
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def _run_metrics(
    frame: pd.DataFrame,
    architecture_errors: pd.DataFrame,
) -> dict:
    result = {}
    for scope in SCOPES:
        result[f"smape_{scope}"] = float(
            architecture_errors[f"smape_{scope}"].mean()
        )
    for target in LABEL_KEYS:
        truth = frame[f"target_{target}"].to_numpy(float)
        prediction = frame[f"prediction_{target}"].to_numpy(float)
        result[f"smape_{target}"] = float(
            architecture_errors[f"smape_{target}"].mean()
        )
        result[f"mae_{target}"] = float(np.abs(truth - prediction).mean())
        result[f"rmse_{target}"] = float(np.sqrt(np.square(truth - prediction).mean()))
        denominator = np.square(truth - truth.mean()).sum()
        result[f"r2_{target}"] = (
            float("nan")
            if denominator == 0
            else float(1 - np.square(truth - prediction).sum() / denominator)
        )
    return result


def _analyze(metadata: dict) -> None:
    source_manifest = json.loads(Path(metadata["source_manifest"]).read_text())
    expected_test = {row["tensor_path"] for row in source_manifest["test"]}
    run_rows = []
    architecture_frames = []
    validation_rows = []
    workload_rows = []
    for model in MODELS:
        for seed in SEEDS:
            for budget in BUDGETS:
                frame = _verified_predictions(metadata, model, seed, budget)
                test = frame[frame["split"] == "test"].copy()
                if set(test["tensor_path"]) != expected_test or len(test) != len(expected_test):
                    raise ValueError(
                        f"Test membership mismatch for {model} seed={seed} p={budget}"
                    )
                errors = _error_columns(test)
                by_architecture = (
                    errors.groupby(
                        ["architecture_id", "kernel_family"], as_index=False
                    )
                    .mean(numeric_only=True)
                )
                by_architecture.insert(0, "budget_percent", budget)
                by_architecture.insert(0, "seed", seed)
                by_architecture.insert(0, "model", model)
                architecture_frames.append(by_architecture)
                manifest_info = metadata["manifests"][str(seed)][str(budget)]
                record = {
                    "model": model,
                    "seed": seed,
                    "budget_percent": budget,
                    "train_designs": manifest_info["train_designs"],
                    "train_architectures": manifest_info["train_architectures"],
                    "test_designs": len(test),
                    "test_architectures": len(by_architecture),
                    **_run_metrics(test, by_architecture),
                }
                for scope in SCOPES:
                    low, high = _cluster_interval(
                        by_architecture[f"smape_{scope}"],
                        metadata["bootstrap_replicates"],
                        _derived_seed(STUDY_ID, "run", model, seed, budget, scope),
                    )
                    record[f"smape_{scope}_ci95_low"] = low
                    record[f"smape_{scope}_ci95_high"] = high
                run_rows.append(record)

                validation = frame[frame["split"] == "validation"]
                if not validation.empty:
                    validation_errors = _error_columns(validation)
                    validation_arch = validation_errors.groupby(
                        "architecture_id", as_index=False
                    ).mean(numeric_only=True)
                    validation_rows.append({
                        "model": model,
                        "seed": seed,
                        "budget_percent": budget,
                        "validation_designs": len(validation),
                        "validation_architectures": len(validation_arch),
                        **{
                            f"smape_{scope}": float(
                                validation_arch[f"smape_{scope}"].mean()
                            )
                            for scope in SCOPES
                        },
                    })

                if budget == 100 and _endpoint_available(metadata, model, seed):
                    endpoint = metadata["full_data_endpoints"][model][str(seed)]
                    resolved = json.loads(Path(endpoint["resolved_config"]).read_text())
                    workload_rows.append({
                        "model": model,
                        "seed": seed,
                        "budget_percent": budget,
                        "new_fit": False,
                        "reused_run_dir": endpoint["run_dir"],
                        "checkpoint_sha256": endpoint.get("checkpoint_sha256"),
                        **{
                            key: resolved.get(key)
                            for key in (
                                "wall_seconds",
                                "cumulative_training_seconds",
                                "best_wall_seconds",
                                "best_epoch",
                                "best_metric",
                                "stop_reason",
                                "peak_gpu_memory_mb",
                                "parameter_count",
                                "feature_count",
                            )
                        },
                    })
                else:
                    run_dir = _run_dir(metadata, model, seed, budget)
                    fit_summary = json.loads((run_dir / "fit_summary.json").read_text())
                    resolved_path = run_dir / "resolved_config.json"
                    resolved = (
                        json.loads(resolved_path.read_text())
                        if resolved_path.is_file()
                        else {}
                    )
                    workload_rows.append({
                        "model": model,
                        "seed": seed,
                        "budget_percent": budget,
                        "new_fit": True,
                        "parameter_count": resolved.get("parameter_count"),
                        **{
                            key: fit_summary.get(key)
                            for key in (
                                "wall_seconds",
                                "cumulative_training_seconds",
                                "best_wall_seconds",
                                "best_epoch",
                                "best_validation_smape",
                                "stop_reason",
                                "peak_gpu_memory_mb",
                                "feature_count",
                            )
                        },
                    })
    runs = pd.DataFrame(run_rows)
    architectures = pd.concat(architecture_frames, ignore_index=True)
    analysis = Path(metadata["output_dir"]) / "analysis"
    _write_frame_atomic(analysis / "per_run_metrics.csv", runs)
    _write_frame_atomic(analysis / "per_architecture_metrics.csv", architectures)
    _write_frame_atomic(analysis / "validation_metrics.csv", pd.DataFrame(validation_rows))
    _write_frame_atomic(analysis / "workload.csv", pd.DataFrame(workload_rows))

    scope_columns = [f"smape_{scope}" for scope in SCOPES]
    seed_summary = (
        runs.groupby(["model", "budget_percent"], as_index=False)
        .agg(
            train_designs_mean=("train_designs", "mean"),
            train_designs_min=("train_designs", "min"),
            train_designs_max=("train_designs", "max"),
            **{
                f"{column}_{stat}": (column, stat)
                for column in scope_columns
                for stat in ("mean", "min", "max")
            },
        )
    )
    _write_frame_atomic(analysis / "seed_summary.csv", seed_summary)

    paired_rows = []
    comparisons = (
        ("high_level", "h0"),
        ("extra_trees", "h0"),
        ("high_level", "extra_trees"),
    )
    for left, right in comparisons:
        for seed in SEEDS:
            for budget in BUDGETS:
                left_frame = architectures.query(
                    "model == @left and seed == @seed and budget_percent == @budget"
                )
                right_frame = architectures.query(
                    "model == @right and seed == @seed and budget_percent == @budget"
                )
                paired = left_frame.merge(
                    right_frame,
                    on=["architecture_id", "kernel_family"],
                    suffixes=("_left", "_right"),
                    validate="one_to_one",
                )
                for scope in SCOPES:
                    delta = paired[f"smape_{scope}_left"] - paired[f"smape_{scope}_right"]
                    low, high = _cluster_interval(
                        delta,
                        metadata["bootstrap_replicates"],
                        _derived_seed(STUDY_ID, "paired", left, right, seed, budget, scope),
                    )
                    paired_rows.append({
                        "left_model": left,
                        "right_model": right,
                        "seed": seed,
                        "budget_percent": budget,
                        "scope": scope,
                        "left_minus_right_smape": float(delta.mean()),
                        "ci95_low": low,
                        "ci95_high": high,
                        "architecture_win_fraction_left": float((delta < 0).mean()),
                        "architectures": len(delta),
                    })
    paired = pd.DataFrame(paired_rows)
    _write_frame_atomic(analysis / "paired_architecture_deltas.csv", paired)

    diagnostics = []
    for model in MODELS:
        for seed in SEEDS:
            selected = runs.query("model == @model and seed == @seed").sort_values(
                "train_designs"
            )
            for scope in SCOPES:
                x = np.log(selected["train_designs"].to_numpy(float))
                y = selected[f"smape_{scope}"].to_numpy(float)
                slope, intercept = np.polyfit(x, y, 1)
                diagnostics.append({
                    "model": model,
                    "seed": seed,
                    "scope": scope,
                    "descriptive_smape_per_log_design_slope": float(slope),
                    "intercept": float(intercept),
                    "smape_reduction_10_to_100": float(y[0] - y[-1]),
                    "designs_10": int(selected.iloc[0]["train_designs"]),
                    "designs_100": int(selected.iloc[-1]["train_designs"]),
                })
    diagnostics_frame = pd.DataFrame(diagnostics)
    _write_frame_atomic(analysis / "scaling_diagnostics.csv", diagnostics_frame)
    _write_figures(analysis, runs)
    _write_text_atomic(
        analysis / "REPORT.md",
        _report(seed_summary, paired, diagnostics_frame, workload_rows),
    )
    _write_json_atomic(analysis / "analysis_provenance.json", {
        "study_id": STUDY_ID,
        "protocol_id": PROTOCOL_ID,
        "primary_metric": "equal architecture mean of per-design six-target SMAPE",
        "outer_unit": "architecture_id",
        "bootstrap_replicates": metadata["bootstrap_replicates"],
        "source_split_sha256": metadata["source_split_sha256"],
        "test_membership_sha256": metadata["test_membership_sha256"],
        "study_metadata_sha256": _sha256(Path(metadata["metadata_path"])),
    })
    inventory_path = analysis / "artifact_inventory.csv"
    inventory = []
    output = Path(metadata["output_dir"])
    for path in sorted(output.rglob("*")):
        if path.is_file() and path != inventory_path and not path.name.endswith(".tmp"):
            inventory.append({
                "relative_path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
    _write_frame_atomic(inventory_path, pd.DataFrame(inventory))


def _write_figures(output: Path, runs: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "h0": "#1f77b4",
        "high_level": "#d62728",
        "extra_trees": "#2ca02c",
    }
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, scope in zip(axes, ("overall", "resource", "timing")):
        for model in MODELS:
            selected = runs[runs["model"] == model]
            for seed in SEEDS:
                trace = selected[selected["seed"] == seed].sort_values("train_designs")
                axis.plot(
                    trace["train_designs"],
                    trace[f"smape_{scope}"],
                    color=colors[model],
                    alpha=0.25,
                    linewidth=1,
                )
            mean = (
                selected.groupby("budget_percent", as_index=False)
                .agg(
                    train_designs=("train_designs", "mean"),
                    smape=(f"smape_{scope}", "mean"),
                )
                .sort_values("train_designs")
            )
            axis.plot(
                mean["train_designs"],
                mean["smape"],
                marker="o",
                linewidth=2,
                color=colors[model],
                label=model,
            )
        axis.set_xscale("log")
        axis.set_title(scope)
        axis.set_xlabel("labeled source designs")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("equal-architecture test SMAPE")
    axes[-1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "scaling_curves.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for axis, target in zip(axes.flat, LABEL_KEYS):
        for model in MODELS:
            selected = runs[runs["model"] == model]
            mean = (
                selected.groupby("budget_percent", as_index=False)
                .agg(
                    train_designs=("train_designs", "mean"),
                    smape=(f"smape_{target}", "mean"),
                )
                .sort_values("train_designs")
            )
            axis.plot(
                mean["train_designs"],
                mean["smape"],
                marker="o",
                color=colors[model],
                label=model,
            )
        axis.set_xscale("log")
        axis.set_title(target)
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("labeled source designs")
    for axis in axes[:, 0]:
        axis.set_ylabel("test SMAPE")
    axes[0, -1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "per_target_scaling_curves.png", dpi=180)
    plt.close(figure)


def _report(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    diagnostics: pd.DataFrame,
    workload: list[dict],
) -> str:
    overall = summary[[
        "model",
        "budget_percent",
        "train_designs_mean",
        "smape_overall_mean",
        "smape_overall_min",
        "smape_overall_max",
    ]].copy()
    overall.columns = (
        "model",
        "budget_percent",
        "mean_train_designs",
        "mean_smape",
        "seed_min",
        "seed_max",
    )
    comparison = paired[
        (paired["left_model"] == "high_level")
        & (paired["right_model"] == "h0")
        & (paired["scope"] == "overall")
    ][[
        "seed",
        "budget_percent",
        "left_minus_right_smape",
        "ci95_low",
        "ci95_high",
        "architecture_win_fraction_left",
    ]]
    total_wall = sum(
        float(row.get("wall_seconds") or 0)
        for row in workload
        if row.get("new_fit")
    )
    return f"""# E6 exact-architecture-group scaling study

E6 estimates performance on the unchanged E2 test architectures as labeled
source architecture-group coverage grows. Percentages select nested groups
within each family; plots use the exact resulting number of labeled designs.

## Overall scaling curve

{_markdown_table(overall)}

## High-level minus H0 paired architecture deltas

Negative SMAPE favors the high-level layer-graph model. Each interval resamples
the unchanged exact test architecture IDs; seeds are reported separately.

{_markdown_table(comparison)}

## Interpretation contract

This is a learning curve for total labeled architecture/design coverage. It is
not the cancelled E3 fixed-signature-density estimand and cannot establish an
intrinsic data-hunger law. The representations are paired on data and training
protocol but are not parameter matched and do not expose identical side
information. Each replication seed jointly changes the nested subset draw and
training randomness, so those variance sources are not separately identified.
The descriptive log-design slopes in `scaling_diagnostics.csv` are summaries,
not power-law exponents.

## Compute and retained evidence

Summed newly fitted model time: {total_wall / 3600:.2f} hours. Raw validation
and test predictions, per-target errors, per-architecture metrics, paired
deltas, exact manifests, learning histories, checkpoints, telemetry, resolved
configs, source-endpoint hashes, and an artifact inventory are retained.
"""


def _primary_complete(metadata: dict, stage: str) -> bool:
    check = _fit_complete if stage == "fit" else _evaluation_complete
    return all(
        check(metadata, model, seed, budget)
        for model in MODELS
        for seed in SEEDS
        for budget in BUDGETS
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Audit the frozen schedule and portable endpoint discovery.
  python scripts/run_e6.py --tensor-dir TENSORS --manifest E2_MANIFEST \\
    --high-level-cache HIGH_LEVEL.pt --e2-results-root E2_RESULTS --dry-run

  # Pilot fit only, without opening E2 test predictions for new models.
  python scripts/run_e6.py --tensor-dir TENSORS --manifest E2_MANIFEST \\
    --high-level-cache HIGH_LEVEL.pt --e2-results-root E2_RESULTS \\
    --stage fit --budgets 10 25 --resume

  # Run/resume the complete frozen study with two jobs on one GPU.
  python scripts/run_e6.py --tensor-dir TENSORS --manifest E2_MANIFEST \\
    --high-level-cache HIGH_LEVEL.pt --e2-results-root E2_RESULTS \\
    --resume --jobs-per-device 2
""",
    )
    parser.add_argument(
        "--stage",
        choices=("prepare", "fit", "evaluate", "analyze", "all"),
        default="all",
    )
    parser.add_argument("--tensor-dir", type=Path)
    parser.add_argument("--tensor-index", type=Path)
    parser.add_argument("--vocab", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--high-level-cache", type=Path)
    parser.add_argument(
        "--e2-results-root", type=Path, nargs="+"
    )
    parser.add_argument("--extra-trees-feature-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--budgets", nargs="+", type=int, default=list(BUDGETS))
    parser.add_argument("--devices", nargs="+", default=["0"])
    parser.add_argument("--jobs-per-device", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--thread-prefetch", action="store_true")
    parser.add_argument("--gpu-telemetry-interval-ms", type=int, default=1000)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--expected-split-sha256", default=OFFICIAL_SPLIT_SHA256)
    parser.add_argument("--expected-vocab-sha256", default=OFFICIAL_VOCAB_SHA256)
    parser.add_argument(
        "--expected-high-level-sha256", default=OFFICIAL_HIGH_LEVEL_SHA256
    )
    parser.add_argument(
        "--expected-tensor-index-sha256", default=EXPECTED_TENSOR_INDEX_SHA256
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--worker-stage",
        choices=("extra-fit", "extra-evaluate"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--study-metadata", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-budget", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_stage:
        if args.study_metadata is None or args.worker_seed is None or args.worker_budget is None:
            parser.error("E6 worker mode requires metadata, seed, and budget")
        metadata = _load_metadata(args.study_metadata.resolve())
        if args.worker_stage == "extra-fit":
            _extra_fit(metadata, args.worker_seed, args.worker_budget)
        else:
            _extra_evaluate(metadata, args.worker_seed, args.worker_budget)
        return

    if any(
        value is None
        for value in (
            args.tensor_dir,
            args.manifest,
            args.high_level_cache,
            args.e2_results_root,
        )
    ):
        parser.error(
            "main mode requires --tensor-dir, --manifest, --high-level-cache, "
            "and --e2-results-root"
        )

    models = tuple(args.models)
    seeds = tuple(args.seeds)
    budgets = tuple(args.budgets)
    if len(models) != len(set(models)):
        parser.error("--models must not contain duplicates")
    if len(seeds) != len(set(seeds)) or not set(seeds) <= set(SEEDS):
        parser.error("--seeds must be a unique subset of 7 42 137")
    if len(budgets) != len(set(budgets)) or not set(budgets) <= set(BUDGETS):
        parser.error("--budgets must be a unique subset of 10 25 50 100")
    if tuple(sorted(budgets)) != budgets:
        parser.error("--budgets must be increasing")
    if not models or not seeds or not budgets:
        parser.error("model, seed, and budget selections must not be empty")
    if args.jobs_per_device < 1 or args.num_workers < 0 or args.prefetch_factor < 1:
        parser.error("invalid worker/concurrency settings")
    if len(args.devices) != len(set(args.devices)):
        parser.error("--devices must not contain duplicates")
    metadata, metadata_path = _resolve_study(args)
    _materialize_configs(metadata)
    if args.stage == "all" and (
        models != MODELS or seeds != SEEDS or budgets != BUDGETS
    ):
        parser.error("--stage all requires the complete frozen E6 matrix")
    if args.stage in {"evaluate", "analyze"} and (
        models != MODELS or seeds != SEEDS or budgets != BUDGETS
    ):
        parser.error("E6 evaluation and analysis require the complete frozen matrix")

    if "extra_trees" in models and args.stage in {"prepare", "fit", "all"}:
        if args.dry_run:
            print(
                "Prepare/verify ExtraTrees features at "
                f"{metadata['extra_trees_feature_cache']}"
            )
        else:
            _ensure_feature_cache(metadata)
    stages = (
        ("prepare", "fit", "evaluate", "analyze")
        if args.stage == "all"
        else (args.stage,)
    )
    for stage in stages:
        if stage == "prepare":
            print(f"E6 protocol prepared at {metadata['output_dir']}")
        elif stage in {"fit", "evaluate"}:
            if stage == "evaluate" and not args.dry_run and not _primary_complete(
                metadata, "fit"
            ):
                raise RuntimeError(
                    "E6 test evaluation is sealed until every primary fit is complete"
                )
            _dispatch(
                metadata,
                metadata_path,
                stage,
                models,
                seeds,
                budgets,
                args.devices,
                args.jobs_per_device,
                args.resume,
                args.fail_fast,
                args.dry_run,
            )
        elif args.dry_run:
            print(f"Analyze complete E6 results under {metadata['output_dir']}")
        else:
            if not _primary_complete(metadata, "evaluate"):
                raise RuntimeError("E6 analysis requires every primary evaluation")
            _analyze(metadata)


if __name__ == "__main__":
    main()
