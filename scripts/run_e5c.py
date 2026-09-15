#!/usr/bin/env python3
"""Run E5c: honest total-label adaptation and representation attribution."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hls-surrogate-lab-matplotlib")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from ll_hls4ml.data.high_level import PROCESSED_FEATURE_DIM
from ll_hls4ml.data.vocab import load_vocab
from ll_hls4ml.io.schema import LABEL_KEYS
from ll_hls4ml.models.registry import build
from ll_hls4ml.training.loaders import make_loader
from ll_hls4ml.training.loops import _autocast, train_one_epoch
from ll_hls4ml.training.targets import LogHuberHurdleLoss
from scripts.run_e5 import (
    CachedFeatureDataset,
    SCOPES,
    _apply_affine,
    _build_source_model,
    _cache_rows,
    _derived_seed,
    _fusion_dataset,
    _git_revision,
    _gpu_preflight,
    _markdown_table,
    _metric_record,
    _predict_model,
    _prediction_rows,
    _rows_by_path,
    _set_seed,
    _sha256,
    _source_head,
    _targets,
    _torch_save_atomic,
    _validate_source_manifest,
    _write_frame_atomic,
    _write_json_atomic,
    _write_stable_json,
    _write_stable_text,
)


STUDY_ID = "e5c_honest_budget_representation_attribution_v1"
PROTOCOL_ID = "e5c_late_added_honest_budget_attribution_2026_09_15"
PARTITION_VERSION = "e5c_fixed_e5_query_nested_total_label_budget_v1"
DEFAULT_E5_ROOT = REPO / "artifacts/results/e5_adaptation_v1"
DEFAULT_OUTPUT = REPO / "artifacts/results/e5c_honest_budget_attribution_v1"

BUDGETS = (4, 8, 16, 32, 64)
PRIMARY_BUDGETS = BUDGETS
DRAW_SEEDS = (7, 42, 137, 271, 911)
HEAD_REPLICATES = (0, 1, 2, 3, 4)
RANDOM_ENCODER_SEEDS = (101, 202, 303, 404, 505)
EPOCHS = {4: 40, 8: 60, 16: 80, 32: 120, 64: 160}
GRAPH_BUDGETS = (32, 64)
SCRATCH_EPOCHS = 200
FULL_TUNE_EPOCHS = 200
FULL_TUNE_LEARNING_RATE = 1e-5
FEATURE_STD_FLOOR = 1e-3

DETERMINISTIC_METHODS = (
    "identity_affine",
    "pretrained_residual_ridge",
    "final_layer_tune",
    "source_head_tune",
)
REPLICATED_METHODS = (
    "fresh_head_standardized_pretrained_encoder",
    "fresh_head_standardized_random_encoder",
)
RANDOM_RIDGE_METHOD = "random_encoder_residual_ridge"
GRAPH_METHODS = ("scratch_full", "pretrained_full_tune")
CALIBRATION_METHODS = (
    "identity_affine", "pretrained_residual_ridge", RANDOM_RIDGE_METHOD,
)
ALL_METHODS = (
    *DETERMINISTIC_METHODS, *REPLICATED_METHODS,
    RANDOM_RIDGE_METHOD, *GRAPH_METHODS,
)

EXPECTED_E5_METADATA_SHA256 = (
    "5b5d18e9575b2dfd6d8c78404e2518cc67e9468b639636c4dfabd8f1296e48c6"
)
EXPECTED_E5_PARTITIONS_SHA256 = (
    "8614dfd22de1365f9dd40663e9a265ede25a2e51543714a992848eedf454b5cf"
)
EXPECTED_PREPROCESSING_SHA256 = (
    "72befc5733f5fb1484dcb370d1b59cd5537848964ce3ad2b297bd58b0ee45ad3"
)
EXPECTED_SOURCE_CHECKPOINT_SHA256 = (
    "dbcf278abe0c95d3c7e1ccd564277e04bb38dff4b9fda5c9ddd9048ca10d8c48"
)
EXPECTED_SOURCE_FEATURES_SHA256 = (
    "ba4734c501ac98f278bc8451c0afb1fca459797e0669e39262dd5c7836181502"
)

RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HEAD_BATCH_SIZE = 32
GRAPH_BATCH_SIZE = 8
LOG_HUBER_DELTA = 0.35
HURDLE_CLASSIFICATION_WEIGHT = 0.25
GRADIENT_CLIP_NORM = 1.0
PRECISION = "bf16"


@dataclass(frozen=True)
class Spec:
    method: str
    budget: int
    replicate: int = 0


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


def _write_csv(path: Path, rows: list[dict]) -> None:
    _write_frame_atomic(path, pd.DataFrame(rows))


def _tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _module_state_sha256(module: torch.nn.Module) -> str:
    return _tensor_state_sha256(dict(module.state_dict()))


def _specs(budgets: tuple[int, ...]) -> list[Spec]:
    specs = [Spec(method, budget) for budget in budgets for method in DETERMINISTIC_METHODS]
    specs += [
        Spec(method, budget, replicate)
        for budget in budgets
        for method in REPLICATED_METHODS
        for replicate in HEAD_REPLICATES
    ]
    specs += [
        Spec(RANDOM_RIDGE_METHOD, budget, replicate)
        for budget in budgets for replicate in HEAD_REPLICATES
    ]
    specs += [
        Spec(method, budget)
        for method in GRAPH_METHODS for budget in budgets if budget in GRAPH_BUDGETS
    ]
    return specs


def _epochs_for_spec(spec: Spec) -> int | None:
    if spec.method == "scratch_full":
        return SCRATCH_EPOCHS
    if spec.method == "pretrained_full_tune":
        return FULL_TUNE_EPOCHS
    if spec.method in (*REPLICATED_METHODS, "final_layer_tune", "source_head_tune"):
        return EPOCHS[spec.budget]
    return None


def _run_dir(metadata: dict, architecture: str, draw: int, spec: Spec) -> Path:
    return (
        Path(metadata["output_dir"])
        / "runs" / architecture / f"draw{draw}" / spec.method
        / f"replicate{spec.replicate}" / f"k{spec.budget}"
    )


def _checkpoint_path(metadata: dict, architecture: str, draw: int, spec: Spec) -> Path:
    return _run_dir(metadata, architecture, draw, spec) / "checkpoint.pt"


def _build_partitions(old: dict) -> dict:
    result = {
        "study_id": STUDY_ID,
        "partition_version": PARTITION_VERSION,
        "source_e5_partitions_sha256": EXPECTED_E5_PARTITIONS_SHA256,
        "partition_seed": old["partition_seed"],
        "budgets": list(BUDGETS),
        "draw_seeds": list(DRAW_SEEDS),
        "query_role": "sealed evaluation only; inherited unchanged from E5",
        "architectures": {},
    }
    for architecture, record in sorted(old["architectures"].items()):
        pool = sorted([*record["validation"], *record["support_pool"]])
        if len(pool) < max(BUDGETS):
            raise ValueError(f"{architecture} has only {len(pool)} adaptation labels")
        draws = {}
        for draw in DRAW_SEEDS:
            rng = np.random.default_rng(
                _derived_seed(PARTITION_VERSION, old["partition_seed"], architecture, draw)
            )
            order = [pool[index] for index in rng.permutation(len(pool))]
            draws[str(draw)] = {
                "support_order": order,
                "support": {str(k): order[:k] for k in BUDGETS},
            }
        result["architectures"][architecture] = {
            "architecture_summary": record.get("architecture_summary"),
            "topology_id": record.get("topology_id"),
            "n_total": record["n_total"],
            "query": record["query"],
            "adaptation_pool": pool,
            "draws": draws,
        }
    _audit_partitions(result, old)
    return result


def _audit_partitions(partitions: dict, old: dict | None = None) -> None:
    if tuple(partitions["budgets"]) != BUDGETS or tuple(partitions["draw_seeds"]) != DRAW_SEEDS:
        raise AssertionError("Unexpected E5c protocol grid")
    for architecture, record in partitions["architectures"].items():
        query, pool = set(record["query"]), set(record["adaptation_pool"])
        if query & pool or len(query | pool) != record["n_total"]:
            raise AssertionError(f"Coverage/overlap failure for {architecture}")
        if old is not None:
            prior = old["architectures"][architecture]
            if record["query"] != prior["query"]:
                raise AssertionError(f"E5 query changed for {architecture}")
            if pool != set(prior["validation"]) | set(prior["support_pool"]):
                raise AssertionError(f"Adaptation pool changed for {architecture}")
        for draw in DRAW_SEEDS:
            previous: set[str] = set()
            for budget in BUDGETS:
                current = set(record["draws"][str(draw)]["support"][str(budget)])
                if len(current) != budget or not previous <= current or not current <= pool:
                    raise AssertionError(f"Non-nested support for {architecture}/{draw}/k{budget}")
                previous = current


def _build_fusion_model(metadata: dict) -> torch.nn.Module:
    resolved = json.loads(Path(metadata["source_resolved_config"]).read_text())
    checkpoint = torch.load(metadata["source_checkpoint"], map_location="cpu", weights_only=True)
    vocabulary, max_pos, _ = load_vocab(metadata["vocab"])
    state = checkpoint["model"]
    return build(
        "hierarchical_high_level_fusion",
        instruction_vocab_size=len(vocabulary), edge_pos_vocab_size=max_pos,
        high_level_input_dim=PROCESSED_FEATURE_DIM,
        y_means=state["y_means"], y_stds=state["y_stds"],
        hidden_dim=int(resolved.get("hidden_dim", 64)),
        num_layers=int(resolved.get("num_layers", 3)),
        heads=int(resolved.get("heads", 1)), dropout=float(resolved.get("dropout", 0.15)),
        high_level_encoder=resolved.get("high_level_encoder", "gatv2"),
        use_global_features=bool(resolved.get("use_global_features", True)),
        use_context=bool(resolved.get("use_context", True)),
        context_mode=resolved.get("context_mode", "core"), split_heads=True,
        hurdle_heads=True,
        hurdle_prediction_mode=resolved.get("hurdle_prediction_mode", "threshold"),
    )


def _prepare_random_cache(metadata: dict, replicate: int) -> None:
    seed = RANDOM_ENCODER_SEEDS[replicate]
    path = Path(metadata["random_feature_caches"][str(replicate)])
    marker = path.with_suffix(".json")
    if path.is_file() and marker.is_file():
        summary = json.loads(marker.read_text())
        if summary["cache_sha256"] != _sha256(path):
            raise RuntimeError(f"Random cache hash mismatch: {path}")
        return
    if path.exists() or marker.exists():
        raise RuntimeError(f"Incomplete random encoder cache: {path}")
    source_cache = torch.load(metadata["source_feature_cache"], map_location="cpu", weights_only=True)
    paths = source_cache["paths"]
    high_level = torch.load(metadata["high_level_cache"], map_location="cpu", weights_only=False)
    preprocessing = torch.load(metadata["preprocessing_path"], map_location="cpu", weights_only=True)
    _set_seed(seed)
    model = _build_fusion_model(metadata)
    encoder_state = {
        name: tensor for name, tensor in model.state_dict().items()
        if not name.startswith("classifier.") and name not in {"y_means", "y_stds"}
    }
    dataset = _fusion_dataset(
        metadata, paths, high_level,
        preprocessing["high_level_means"], preprocessing["high_level_stds"],
    )
    loader = make_loader(
        dataset, batch_size=metadata["evaluation_batch_size"], shuffle=False,
        num_workers=metadata["num_workers"], pin_memory=True,
        prefetch_factor=metadata["prefetch_factor"],
        thread_prefetch=metadata["thread_prefetch"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    features = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            with _autocast(device, metadata["precision"]):
                features.append(model.encode(batch).float().cpu())
    payload = {"format_version": 1, "paths": paths, "features": torch.cat(features)}
    # Retained only to satisfy the shared cache interface; attribution uses these
    # caches exclusively for features, never for source-model predictions.
    payload["source_predictions"] = source_cache["source_predictions"]
    if payload["features"].shape != source_cache["features"].shape:
        raise RuntimeError("Random and pretrained feature shapes differ")
    if not torch.isfinite(payload["features"]).all():
        raise RuntimeError("Random encoder produced non-finite features")
    _torch_save_atomic(path, payload)
    _write_json_atomic(marker, {
        "status": "complete", "replicate": replicate, "encoder_seed": seed,
        "encoder_state_sha256": _tensor_state_sha256(encoder_state),
        "samples": len(paths), "feature_width": int(payload["features"].shape[1]),
        "cache_sha256": _sha256(path),
    })


def _prepare(metadata: dict) -> None:
    source_cache = Path(metadata["source_feature_cache"])
    if _sha256(source_cache) != EXPECTED_SOURCE_FEATURES_SHA256:
        raise ValueError("Frozen E5 source feature cache hash changed")
    for replicate in HEAD_REPLICATES:
        _prepare_random_cache(metadata, replicate)
    random_states = [
        json.loads(
            Path(metadata["random_feature_caches"][str(replicate)]).with_suffix(".json").read_text()
        )["encoder_state_sha256"]
        for replicate in HEAD_REPLICATES
    ]
    if len(set(random_states)) != len(random_states):
        raise RuntimeError("Random encoder state hashes are not unique")
    if metadata["source_encoder_state_sha256"] in random_states:
        raise RuntimeError("A random encoder unexpectedly matches the source encoder")
    summary = {
        "status": "complete", "source_feature_cache_sha256": _sha256(source_cache),
        "random_feature_cache_sha256": {
            str(r): _sha256(Path(metadata["random_feature_caches"][str(r)]))
            for r in HEAD_REPLICATES
        },
        "random_encoder_state_sha256": random_states,
        "source_encoder_state_sha256": metadata["source_encoder_state_sha256"],
    }
    _write_json_atomic(Path(metadata["output_dir"]) / "cache/prepare_summary.json", summary)


def _head_model(metadata: dict, cache: dict, initialization: str, seed: int):
    if initialization == "source":
        return _source_head(metadata, cache)
    _set_seed(seed)
    source = _source_head(metadata, cache)
    # Reconstructing after setting the seed gives paired, byte-identical fresh heads.
    return type(source)(
        input_dim=int(cache["features"].shape[1]),
        hidden_dim=source.classifier.resource[0].out_features,
        dropout=source.classifier.resource[2].p,
        y_means=source.y_means.cpu(), y_stds=source.y_stds.cpu(),
        hurdle_prediction_mode=source.hurdle_prediction_mode,
    )


def _training_seed(architecture: str, draw: int, budget: int, replicate: int) -> int:
    # Deliberately excludes encoder condition for paired head training.
    return _derived_seed(STUDY_ID, "paired_head", architecture, draw, budget, replicate)


def _fit_fixed_epochs(
    model, dataset, epochs: int, batch_size: int, seed: int, device: torch.device,
    precision: str, learning_rate: float = LEARNING_RATE,
) -> list[dict]:
    _set_seed(seed)
    loader = make_loader(
        dataset, batch_size=min(batch_size, len(dataset)), shuffle=True,
        num_workers=0, pin_memory=True,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = LogHuberHurdleLoss(
        model.y_means, model.y_stds, delta=LOG_HUBER_DELTA,
        classification_weight=HURDLE_CLASSIFICATION_WEIGHT,
    ).to(device)
    model.to(device)
    history = []
    for epoch in range(epochs):
        loss = train_one_epoch(
            model, loader, criterion, optimizer, device,
            precision=precision, gradient_clip_norm=GRADIENT_CLIP_NORM,
        )
        history.append({"epoch": epoch + 1, "train_loss": loss, "lr": optimizer.param_groups[0]["lr"]})
        scheduler.step()
    return history


def _smape(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(200 * np.abs(truth - prediction) / (np.abs(truth) + np.abs(prediction) + 1.0)))


def _ridge_solution(x: np.ndarray, y: np.ndarray, alpha: float, prior: np.ndarray) -> np.ndarray:
    # Dual form: every E5c support set is much smaller than the 261-column
    # design, so this solves a k-by-k rather than 261-by-261 system.
    inverse_penalty = np.full(x.shape[1], 1.0 / alpha)
    inverse_penalty[0] = 100.0 / alpha
    weighted_xt = inverse_penalty[:, None] * x.T
    correction = weighted_xt @ np.linalg.solve(
        np.eye(len(x)) + x @ weighted_xt,
        y - x @ prior,
    )
    return prior + correction


def _loocv_affine(source: np.ndarray, target: np.ndarray) -> dict:
    log_x, log_y = np.log1p(source.clip(min=0)), np.log1p(target.clip(min=0))
    coefficients = []
    for column, name in enumerate(LABEL_KEYS):
        x = np.column_stack([np.ones(len(source)), log_x[:, column]])
        y, prior = log_y[:, column], np.array([0.0, 1.0])
        scores = []
        for alpha in RIDGE_GRID:
            predictions = []
            for held_out in range(len(y)):
                keep = np.arange(len(y)) != held_out
                theta = _ridge_solution(x[keep], y[keep], alpha, prior)
                predictions.append(max(0.0, np.expm1(x[held_out] @ theta)))
            scores.append((_smape(target[:, column], np.asarray(predictions)), alpha))
        score, alpha = min(scores)
        theta = _ridge_solution(x, y, alpha, prior)
        coefficients.append({
            "target": name, "intercept": float(theta[0]), "slope": float(theta[1]),
            "ridge": alpha, "loocv_smape": score,
        })
    return {"space": "log1p_target", "selection": "support-only LOOCV", "coefficients": coefficients}


def _feature_standardizer(cache: dict, source_paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    features, _ = _cache_rows(cache, source_paths)
    mean = features.numpy().mean(axis=0)
    std = features.numpy().std(axis=0)
    return mean, np.maximum(std, FEATURE_STD_FLOOR)


def _standardized_cache_rows(
    cache: dict, paths: list[str], source_paths: list[str]
) -> tuple[torch.Tensor, torch.Tensor, str]:
    features, predictions = _cache_rows(cache, paths)
    mean, std = _feature_standardizer(cache, source_paths)
    standardized = (features - torch.from_numpy(mean)) / torch.from_numpy(std)
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(mean).tobytes())
    digest.update(np.ascontiguousarray(std).tobytes())
    digest.update(str(FEATURE_STD_FLOOR).encode())
    return standardized.float(), predictions, digest.hexdigest()


def _fit_residual_ridge(features: np.ndarray, source: np.ndarray, target: np.ndarray,
                        mean: np.ndarray, std: np.ndarray) -> dict:
    z = (features - mean) / std
    x = np.column_stack([np.ones(len(z)), z])
    residual = np.log1p(target.clip(min=0)) - np.log1p(source.clip(min=0))
    prior = np.zeros(x.shape[1])
    scores = []
    for alpha in RIDGE_GRID:
        predictions = np.empty_like(target)
        for held_out in range(len(x)):
            keep = np.arange(len(x)) != held_out
            beta = np.column_stack([
                _ridge_solution(x[keep], residual[keep, column], alpha, prior)
                for column in range(len(LABEL_KEYS))
            ])
            predictions[held_out] = np.expm1(
                np.log1p(source[held_out].clip(min=0)) + x[held_out] @ beta
            ).clip(min=0)
        scores.append((_smape(target, predictions), alpha))
    score, alpha = min(scores)
    beta = np.column_stack([
        _ridge_solution(x, residual[:, column], alpha, prior)
        for column in range(len(LABEL_KEYS))
    ])
    return {
        "selection": "support-only LOOCV macro-SMAPE", "ridge": alpha,
        "loocv_smape": score, "feature_mean": mean.tolist(),
        "feature_std": std.tolist(), "coefficients": beta.tolist(),
    }


def _apply_residual_ridge(features: torch.Tensor, source: torch.Tensor, calibration: dict) -> torch.Tensor:
    z = (features.numpy() - np.asarray(calibration["feature_mean"])) / np.asarray(calibration["feature_std"])
    x = np.column_stack([np.ones(len(z)), z])
    beta = np.asarray(calibration["coefficients"])
    prediction = np.expm1(np.log1p(source.numpy().clip(min=0)) + x @ beta).clip(min=0)
    return torch.tensor(prediction, dtype=torch.float32)


def _fit_spec(metadata: dict, architecture: str, draw: int, spec: Spec,
              support: list[str], rows: dict[str, dict]) -> None:
    run_dir = _run_dir(metadata, architecture, draw, spec)
    marker, checkpoint = run_dir / "fit_summary.json", _checkpoint_path(metadata, architecture, draw, spec)
    artifact = run_dir / "calibration.json" if spec.method in CALIBRATION_METHODS else checkpoint
    if marker.is_file() and artifact.is_file():
        summary = json.loads(marker.read_text())
        if summary["artifact_sha256"] != _sha256(artifact):
            raise RuntimeError(f"Fit artifact hash mismatch: {run_dir}")
        return
    if marker.exists() or artifact.exists():
        raise RuntimeError(f"Incomplete run; move it aside before retrying: {run_dir}")
    cache_path = metadata["source_feature_cache"]
    if spec.method in {"fresh_head_standardized_random_encoder", RANDOM_RIDGE_METHOD}:
        cache_path = metadata["random_feature_caches"][str(spec.replicate)]
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    config = {
        "study_id": STUDY_ID, "architecture_id": architecture, "draw_seed": draw,
        **asdict(spec), "total_target_labels": spec.budget,
        "epochs": _epochs_for_spec(spec),
        "selection_data": "support-only; no validation/query labels",
        "learning_rate": (
            FULL_TUNE_LEARNING_RATE
            if spec.method == "pretrained_full_tune" else LEARNING_RATE
        ),
        "weight_decay": WEIGHT_DECAY,
        "head_batch_size": HEAD_BATCH_SIZE, "graph_batch_size": GRAPH_BATCH_SIZE,
        "log_huber_delta": LOG_HUBER_DELTA,
        "hurdle_classification_weight": HURDLE_CLASSIFICATION_WEIGHT,
        "gradient_clip_norm": GRADIENT_CLIP_NORM, "precision": metadata["precision"],
    }
    _write_stable_json(run_dir / "run_config.json", config)
    started = time.perf_counter()
    trainable = 0
    history: list[dict] = []
    initial_head_sha256 = None
    feature_normalizer_sha256 = None
    if spec.method == "identity_affine":
        _, source = _cache_rows(cache, support)
        calibration = _loocv_affine(source.numpy(), _targets(support, rows).numpy())
        _write_json_atomic(artifact, calibration)
    elif spec.method in {"pretrained_residual_ridge", RANDOM_RIDGE_METHOD}:
        features, source = _cache_rows(cache, support)
        manifest = json.loads(Path(metadata["source_manifest"]).read_text())
        source_paths = [row["tensor_path"] for row in manifest["test"]]
        mean, std = _feature_standardizer(cache, source_paths)
        calibration = _fit_residual_ridge(
            features.numpy(), source.numpy(), _targets(support, rows).numpy(), mean, std
        )
        _write_json_atomic(artifact, calibration)
        trainable = (features.shape[1] + 1) * len(LABEL_KEYS)
    elif spec.method in GRAPH_METHODS:
        seed = _derived_seed(STUDY_ID, spec.method, architecture, draw)
        _set_seed(seed)
        model = (
            _build_fusion_model(metadata)
            if spec.method == "scratch_full" else _build_source_model(metadata)
        )
        high_level = torch.load(metadata["high_level_cache"], map_location="cpu", weights_only=False)
        preprocessing = torch.load(metadata["preprocessing_path"], map_location="cpu", weights_only=True)
        dataset = _fusion_dataset(
            metadata, support, high_level,
            preprocessing["high_level_means"], preprocessing["high_level_stds"],
        )
        epochs = _epochs_for_spec(spec)
        learning_rate = (
            FULL_TUNE_LEARNING_RATE
            if spec.method == "pretrained_full_tune" else LEARNING_RATE
        )
        history = _fit_fixed_epochs(
            model, dataset, epochs, GRAPH_BATCH_SIZE,
            _derived_seed(seed, spec.budget), torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            metadata["precision"], learning_rate=learning_rate,
        )
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _torch_save_atomic(checkpoint, {"model": model.cpu().state_dict(), "epochs": epochs})
    else:
        seed = _training_seed(architecture, draw, spec.budget, spec.replicate)
        initialization = "source" if spec.method in {"source_head_tune", "final_layer_tune"} else "fresh"
        model = _head_model(metadata, cache, initialization, seed)
        initial_head_sha256 = _module_state_sha256(model.classifier)
        if spec.method == "final_layer_tune":
            for parameter in model.parameters():
                parameter.requires_grad = False
            for tower in (model.classifier.resource, model.classifier.timing):
                for parameter in tower[3].parameters():
                    parameter.requires_grad = True
        if spec.method in REPLICATED_METHODS:
            manifest = json.loads(Path(metadata["source_manifest"]).read_text())
            source_paths = [row["tensor_path"] for row in manifest["test"]]
            features, _, feature_normalizer_sha256 = _standardized_cache_rows(
                cache, support, source_paths
            )
        else:
            features, _ = _cache_rows(cache, support)
        dataset = CachedFeatureDataset(features, _targets(support, rows))
        history = _fit_fixed_epochs(
            model, dataset, EPOCHS[spec.budget], HEAD_BATCH_SIZE, seed,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"), metadata["precision"],
        )
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _torch_save_atomic(checkpoint, {"model": model.cpu().state_dict(), "epochs": EPOCHS[spec.budget]})
    if history:
        _write_csv(run_dir / "learning_curves.csv", history)
    _write_json_atomic(marker, {
        "status": "complete", **asdict(spec), "architecture_id": architecture,
        "draw_seed": draw, "total_target_labels": spec.budget,
        "trainable_parameters": trainable, "wall_seconds": time.perf_counter() - started,
        "artifact": str(artifact), "artifact_sha256": _sha256(artifact),
        "feature_cache_sha256": _sha256(Path(cache_path)),
        "feature_normalizer_sha256": feature_normalizer_sha256,
        "initial_head_sha256": initial_head_sha256,
    })


def _fit_bundle(metadata: dict, architecture: str, draw: int, budgets: tuple[int, ...]) -> None:
    manifest = json.loads(Path(metadata["source_manifest"]).read_text())
    rows = _rows_by_path(manifest)
    partitions = json.loads(Path(metadata["partitions_path"]).read_text())
    record = partitions["architectures"][architecture]
    for spec in _specs(budgets):
        support = record["draws"][str(draw)]["support"][str(spec.budget)]
        print(f"FIT {architecture} draw={draw} {spec.method} r={spec.replicate} k={spec.budget}", flush=True)
        _fit_spec(metadata, architecture, draw, spec, support, rows)


def _evaluate_spec(metadata: dict, architecture: str, draw: int, spec: Spec,
                   query: list[str], rows: dict[str, dict]) -> None:
    run_dir = _run_dir(metadata, architecture, draw, spec)
    marker, output = run_dir / "evaluation_summary.json", run_dir / "predictions.csv"
    if marker.is_file() and output.is_file():
        if json.loads(marker.read_text())["predictions_sha256"] != _sha256(output):
            raise RuntimeError(f"Prediction hash mismatch: {run_dir}")
        return
    if not (run_dir / "fit_summary.json").is_file():
        raise FileNotFoundError(f"Missing fit: {run_dir}")
    source_cache = torch.load(metadata["source_feature_cache"], map_location="cpu", weights_only=True)
    cache = source_cache
    if spec.method in {"fresh_head_standardized_random_encoder", RANDOM_RIDGE_METHOD}:
        cache = torch.load(metadata["random_feature_caches"][str(spec.replicate)], map_location="cpu", weights_only=True)
    if spec.method == "identity_affine":
        _, source = _cache_rows(source_cache, query)
        prediction = _apply_affine(source, json.loads((run_dir / "calibration.json").read_text()))
    elif spec.method in {"pretrained_residual_ridge", RANDOM_RIDGE_METHOD}:
        features, source = _cache_rows(cache, query)
        prediction = _apply_residual_ridge(features, source, json.loads((run_dir / "calibration.json").read_text()))
    elif spec.method in GRAPH_METHODS:
        model = (
            _build_fusion_model(metadata)
            if spec.method == "scratch_full" else _build_source_model(metadata)
        )
        model.load_state_dict(torch.load(_checkpoint_path(metadata, architecture, draw, spec), map_location="cpu", weights_only=True)["model"])
        high_level = torch.load(metadata["high_level_cache"], map_location="cpu", weights_only=False)
        preprocessing = torch.load(metadata["preprocessing_path"], map_location="cpu", weights_only=True)
        dataset = _fusion_dataset(metadata, query, high_level, preprocessing["high_level_means"], preprocessing["high_level_stds"])
        loader = make_loader(dataset, batch_size=metadata["evaluation_batch_size"], shuffle=False,
                             num_workers=metadata["num_workers"], pin_memory=True,
                             prefetch_factor=metadata["prefetch_factor"], thread_prefetch=metadata["thread_prefetch"])
        prediction, _ = _predict_model(model, loader, torch.device("cuda" if torch.cuda.is_available() else "cpu"), metadata["precision"])
    else:
        seed = _training_seed(architecture, draw, spec.budget, spec.replicate)
        initialization = "source" if spec.method in {"source_head_tune", "final_layer_tune"} else "fresh"
        model = _head_model(metadata, cache, initialization, seed)
        state = torch.load(_checkpoint_path(metadata, architecture, draw, spec), map_location="cpu", weights_only=True)["model"]
        model.load_state_dict(state, strict=True)
        if spec.method in REPLICATED_METHODS:
            manifest = json.loads(Path(metadata["source_manifest"]).read_text())
            source_paths = [row["tensor_path"] for row in manifest["test"]]
            features, _, _ = _standardized_cache_rows(cache, query, source_paths)
        else:
            features, _ = _cache_rows(cache, query)
        dataset = CachedFeatureDataset(features, _targets(query, rows))
        loader = make_loader(dataset, batch_size=metadata["evaluation_batch_size"], shuffle=False, num_workers=0, pin_memory=True)
        prediction, _ = _predict_model(model, loader, torch.device("cuda" if torch.cuda.is_available() else "cpu"), metadata["precision"])
    prediction_rows = _prediction_rows(query, "query", prediction, rows, architecture, draw, spec.method, spec.budget)
    for row in prediction_rows:
        row["replicate"] = spec.replicate
        row["total_target_labels"] = spec.budget
        row["encoder_seed"] = (
            RANDOM_ENCODER_SEEDS[spec.replicate]
            if spec.method in {"fresh_head_standardized_random_encoder", RANDOM_RIDGE_METHOD}
            else ""
        )
    _write_csv(output, prediction_rows)
    _write_json_atomic(marker, {"status": "complete", **asdict(spec), "query_samples": len(query), "predictions_sha256": _sha256(output)})


def _evaluate_bundle(metadata: dict, architecture: str, draw: int, budgets: tuple[int, ...]) -> None:
    manifest = json.loads(Path(metadata["source_manifest"]).read_text())
    rows = _rows_by_path(manifest)
    record = json.loads(Path(metadata["partitions_path"]).read_text())["architectures"][architecture]
    for spec in _specs(budgets):
        print(f"EVALUATE {architecture} draw={draw} {spec.method} r={spec.replicate} k={spec.budget}", flush=True)
        _evaluate_spec(metadata, architecture, draw, spec, record["query"], rows)


def _bundle_complete(metadata: dict, stage: str, architecture: str, draw: int,
                     budgets: tuple[int, ...]) -> bool:
    for spec in _specs(budgets):
        run_dir = _run_dir(metadata, architecture, draw, spec)
        marker = run_dir / ("fit_summary.json" if stage == "fit" else "evaluation_summary.json")
        artifact = (run_dir / "calibration.json") if stage == "fit" and spec.method in CALIBRATION_METHODS else (
            _checkpoint_path(metadata, architecture, draw, spec) if stage == "fit" else run_dir / "predictions.csv"
        )
        if not marker.is_file() or not artifact.is_file():
            return False
        field = "artifact_sha256" if stage == "fit" else "predictions_sha256"
        try:
            if json.loads(marker.read_text())[field] != _sha256(artifact):
                return False
        except (OSError, json.JSONDecodeError, KeyError):
            return False
    return True


def _matrix_complete(metadata: dict, stage: str, architectures, draws, budgets) -> bool:
    return all(_bundle_complete(metadata, stage, architecture, draw, budgets)
               for architecture in architectures for draw in draws)


def _zero_metrics(metadata: dict) -> pd.DataFrame:
    manifest = json.loads(Path(metadata["source_manifest"]).read_text())
    rows = _rows_by_path(manifest)
    partitions = json.loads(Path(metadata["partitions_path"]).read_text())
    cache = torch.load(metadata["source_feature_cache"], map_location="cpu", weights_only=True)
    output = []
    for architecture, record in partitions["architectures"].items():
        _, prediction = _cache_rows(cache, record["query"])
        frame = pd.DataFrame(_prediction_rows(record["query"], "query", prediction, rows, architecture, None, "zero_shot", 0))
        output.append({"architecture_id": architecture, "draw_seed": -1, "method": "zero_shot", "budget": 0, "replicate": 0, **_metric_record(frame)})
    return pd.DataFrame(output)


def _hierarchical_effect_interval(frame: pd.DataFrame, column: str, replicates: int,
                                  seed: int, coverage: float) -> tuple[float, float]:
    groups = {a: g[column].to_numpy(float) for a, g in frame.groupby("architecture_id")}
    architectures = sorted(groups)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates)
    for index in range(replicates):
        selected = rng.choice(architectures, len(architectures), replace=True)
        estimates[index] = np.mean([rng.choice(groups[a]) for a in selected])
    tail = (100.0 - coverage) / 2
    return tuple(np.percentile(estimates, [tail, 100 - tail]))


def _sign_flip_p(values: np.ndarray) -> float:
    observed = abs(values.mean())
    statistics = [abs(np.mean(values * np.asarray(signs))) for signs in itertools.product((-1, 1), repeat=len(values))]
    return float(np.mean(np.asarray(statistics) >= observed - 1e-12))


def _holm(rows: list[dict]) -> None:
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p_value"])
    running = 0.0
    count = len(rows)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (count - rank) * rows[index]["p_value"])
        running = max(running, adjusted)
        rows[index]["p_holm"] = running


def _analyze(metadata: dict) -> None:
    output = Path(metadata["output_dir"])
    frames = []
    partitions = json.loads(Path(metadata["partitions_path"]).read_text())
    for architecture in sorted(partitions["architectures"]):
        for draw in DRAW_SEEDS:
            for spec in _specs(BUDGETS):
                frame = pd.read_csv(_run_dir(metadata, architecture, draw, spec) / "predictions.csv")
                frames.append({"architecture_id": architecture, "draw_seed": draw, "method": spec.method,
                               "budget": spec.budget, "replicate": spec.replicate, **_metric_record(frame)})
    per_run = pd.concat([_zero_metrics(metadata), pd.DataFrame(frames)], ignore_index=True)
    analysis = output / "analysis"
    _write_frame_atomic(analysis / "per_run_metrics.csv", per_run)
    draw_level = per_run[per_run.method != "zero_shot"].groupby(
        ["architecture_id", "draw_seed", "method", "budget"], as_index=False
    ).mean(numeric_only=True)
    zero = per_run[per_run.method == "zero_shot"][["architecture_id", "smape_overall"]].rename(columns={"smape_overall": "zero_smape"})
    summaries = []
    for (method, budget), group in draw_level.groupby(["method", "budget"]):
        for scope in SCOPES:
            column = f"smape_{scope}"
            lo, hi = _hierarchical_effect_interval(group, column, metadata["bootstrap_replicates"], _derived_seed(STUDY_ID, method, budget, scope), 95)
            summaries.append({"method": method, "budget": budget, "scope": scope,
                              "mean_smape": group.groupby("architecture_id")[column].mean().mean(),
                              "ci95_low": lo, "ci95_high": hi})
    zero_frame = per_run[per_run.method == "zero_shot"]
    for scope in SCOPES:
        column = f"smape_{scope}"
        lo, hi = _hierarchical_effect_interval(zero_frame, column, metadata["bootstrap_replicates"], _derived_seed(STUDY_ID, "zero", scope), 95)
        summaries.append({"method": "zero_shot", "budget": 0, "scope": scope,
                          "mean_smape": zero_frame[column].mean(), "ci95_low": lo, "ci95_high": hi})
    _write_frame_atomic(analysis / "equal_architecture_summary.csv", pd.DataFrame(summaries))

    primary = []
    global_rows = []
    architecture_rows = []
    for claim, positive_method, negative_method in (
        ("P1_source_head_vs_zero", "zero_shot", "source_head_tune"),
        (
            "P2_pretrained_vs_random_encoder",
            "fresh_head_standardized_random_encoder",
            "fresh_head_standardized_pretrained_encoder",
        ),
    ):
        claim_rows = []
        all_pairs = []
        for budget in PRIMARY_BUDGETS:
            if claim.startswith("P1"):
                paired = draw_level[(draw_level.method == negative_method) & (draw_level.budget == budget)].merge(zero, on="architecture_id")
                paired["advantage_smape"] = paired.zero_smape - paired.smape_overall
            else:
                left = draw_level[(draw_level.method == positive_method) & (draw_level.budget == budget)]
                right = draw_level[(draw_level.method == negative_method) & (draw_level.budget == budget)]
                paired = left.merge(right, on=["architecture_id", "draw_seed", "budget"], suffixes=("_positive", "_negative"))
                paired["advantage_smape"] = paired.smape_overall_positive - paired.smape_overall_negative
            paired["claim"] = claim
            all_pairs.append(paired[["architecture_id", "draw_seed", "budget", "advantage_smape"]])
            arch_effect = paired.groupby("architecture_id").advantage_smape.mean()
            ci95 = _hierarchical_effect_interval(paired, "advantage_smape", metadata["bootstrap_replicates"], _derived_seed(STUDY_ID, claim, budget, 95), 95)
            # Bonferroni simultaneous intervals over the five registered budgets.
            ci99 = _hierarchical_effect_interval(paired, "advantage_smape", metadata["bootstrap_replicates"], _derived_seed(STUDY_ID, claim, budget, 99), 99)
            row = {"claim": claim, "budget": budget, "advantage_smape": arch_effect.mean(),
                   "ci95_low": ci95[0], "ci95_high": ci95[1], "simultaneous_ci99_low": ci99[0], "simultaneous_ci99_high": ci99[1],
                   "architecture_wins": int((arch_effect > 0).sum()), "architectures": len(arch_effect),
                   "p_value": _sign_flip_p(arch_effect.to_numpy())}
            row["pointwise_claim_pass"] = bool(row["simultaneous_ci99_low"] > 0 and row["architecture_wins"] >= 6)
            claim_rows.append(row)
            architecture_rows.extend({
                "claim": claim, "estimand": "budget_specific", "budget": budget,
                "architecture_id": architecture, "advantage_smape": effect,
            } for architecture, effect in arch_effect.items())
        curve = pd.concat(all_pairs, ignore_index=True).groupby(
            ["architecture_id", "draw_seed"], as_index=False
        ).advantage_smape.mean()
        curve_arch = curve.groupby("architecture_id").advantage_smape.mean()
        curve_ci = _hierarchical_effect_interval(
            curve, "advantage_smape", metadata["bootstrap_replicates"],
            _derived_seed(STUDY_ID, claim, "curve"), 97.5,
        )
        global_rows.append({
            "claim": claim, "estimand": "equal_weight_mean_over_k_4_8_16_32_64",
            "advantage_smape": curve_arch.mean(), "ci97_5_low": curve_ci[0],
            "ci97_5_high": curve_ci[1],
            "architecture_wins": int((curve_arch > 0).sum()),
            "architectures": len(curve_arch), "p_value": _sign_flip_p(curve_arch.to_numpy()),
        })
        architecture_rows.extend({
            "claim": claim, "estimand": "curve_average", "budget": "all",
            "architecture_id": architecture, "advantage_smape": effect,
        } for architecture, effect in curve_arch.items())
        primary.extend(claim_rows)
    _holm(global_rows)
    for row in global_rows:
        row["strong_global_claim_pass"] = bool(
            row["ci97_5_low"] > 0 and row["architecture_wins"] >= 6 and row["p_holm"] < 0.05
        )
    _write_frame_atomic(analysis / "primary_curve_contrasts.csv", pd.DataFrame(global_rows))
    _write_frame_atomic(analysis / "primary_contrasts.csv", pd.DataFrame(primary))
    _write_frame_atomic(analysis / "architecture_contrasts.csv", pd.DataFrame(architecture_rows))

    secondary = []
    for budget in BUDGETS:
        fresh = draw_level[(draw_level.method == "fresh_head_standardized_pretrained_encoder") & (draw_level.budget == budget)]
        source = draw_level[(draw_level.method == "source_head_tune") & (draw_level.budget == budget)]
        paired = fresh.merge(source, on=["architecture_id", "draw_seed", "budget"], suffixes=("_fresh", "_source"))
        paired["advantage_smape"] = paired.smape_overall_fresh - paired.smape_overall_source
        arch = paired.groupby("architecture_id").advantage_smape.mean()
        lo, hi = _hierarchical_effect_interval(paired, "advantage_smape", metadata["bootstrap_replicates"], _derived_seed(STUDY_ID, "P3", budget), 95)
        secondary.append({"contrast": "source_head_initialization_vs_fresh_head", "budget": budget,
                          "advantage_smape": arch.mean(), "ci95_low": lo, "ci95_high": hi,
                          "architecture_wins": int((arch > 0).sum())})
        random_ridge = draw_level[(draw_level.method == RANDOM_RIDGE_METHOD) & (draw_level.budget == budget)]
        pretrained_ridge = draw_level[(draw_level.method == "pretrained_residual_ridge") & (draw_level.budget == budget)]
        ridge_pair = random_ridge.merge(
            pretrained_ridge, on=["architecture_id", "draw_seed", "budget"],
            suffixes=("_random", "_pretrained"),
        )
        ridge_pair["advantage_smape"] = (
            ridge_pair.smape_overall_random - ridge_pair.smape_overall_pretrained
        )
        ridge_arch = ridge_pair.groupby("architecture_id").advantage_smape.mean()
        ridge_lo, ridge_hi = _hierarchical_effect_interval(
            ridge_pair, "advantage_smape", metadata["bootstrap_replicates"],
            _derived_seed(STUDY_ID, "P2_convex", budget), 95,
        )
        secondary.append({
            "contrast": "pretrained_vs_random_encoder_residual_ridge",
            "budget": budget, "advantage_smape": ridge_arch.mean(),
            "ci95_low": ridge_lo, "ci95_high": ridge_hi,
            "architecture_wins": int((ridge_arch > 0).sum()),
        })
    _write_frame_atomic(analysis / "secondary_contrasts.csv", pd.DataFrame(secondary))
    report = ["# E5c honest-budget representation attribution", "", "Positive advantages favor the named treatment.", "", "## Primary curve-average claims", "",
              _markdown_table(pd.DataFrame(global_rows)), "", "## Registered budget-specific contrasts", "",
              _markdown_table(pd.DataFrame(primary)), "", "## Interpretation rule", "",
              "A strong curve-level headline requires its 97.5% hierarchical interval to exclude zero, Holm-adjusted p < 0.05 across the two claims, and at least 6/7 architecture means to agree. A budget-specific claim requires its within-claim Bonferroni 99% interval to exclude zero and at least 6/7 architecture means to agree.", ""]
    _write_stable_text(analysis / "report.md", "\n".join(report))
    _write_json_atomic(analysis / "analysis_summary.json", {
        "status": "complete",
        "primary_curve_contrasts_sha256": _sha256(analysis / "primary_curve_contrasts.csv"),
        "primary_contrasts_sha256": _sha256(analysis / "primary_contrasts.csv"),
        "per_run_metrics_sha256": _sha256(analysis / "per_run_metrics.csv"),
    })


def _worker_command(stage: str, metadata_path: Path, job: Job, budgets) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), "--worker-stage", stage,
            "--study-metadata", str(metadata_path), "--architecture", job.architecture_id,
            "--draw-seed", str(job.draw_seed), "--budgets", *map(str, budgets)]


def _dispatch(stage: str, metadata: dict, metadata_path: Path, architectures, draws,
              budgets, devices, jobs_per_device: int, dry_run: bool, fail_fast: bool) -> None:
    slots = [device for device in devices for _ in range(jobs_per_device)]
    jobs = []
    for position, (architecture, draw) in enumerate(itertools.product(architectures, draws)):
        jobs.append(Job(architecture, draw, stage,
                        str(Path(metadata["output_dir"]) / "logs" / stage / f"{architecture}_draw{draw}.log"),
                        slots[position % len(slots)],
                        "SKIPPED_COMPLETE" if _bundle_complete(metadata, stage, architecture, draw, budgets) else "PENDING"))
    index_path = Path(metadata["output_dir"]) / f"{stage}_index.json"
    def persist():
        _write_json_atomic(index_path, {"stage": stage, "budgets": list(budgets), "draw_seeds": list(draws),
                                        "architectures": list(architectures), "devices": list(devices),
                                        "jobs_per_device": jobs_per_device, "dry_run": dry_run,
                                        "jobs": [asdict(job) for job in jobs]})
    persist()
    pending = [job for job in jobs if job.status == "PENDING"]
    if dry_run:
        for job in pending:
            print(f"CUDA_VISIBLE_DEVICES={shlex.quote(job.device)} " + shlex.join(_worker_command(stage, metadata_path, job, budgets)))
        print(f"E5c {stage}: runnable={len(pending)} index={index_path}")
        return
    for device in dict.fromkeys(job.device for job in pending):
        _gpu_preflight(device)
    available, active, failed = list(slots), [], False
    try:
        while pending or active:
            while pending and available and not (failed and fail_fast):
                job, device = pending.pop(0), available.pop(0)
                job.device = device
                log = Path(job.log_path); log.parent.mkdir(parents=True, exist_ok=True)
                handle = log.open("a", buffering=1)
                command = _worker_command(stage, metadata_path, job, budgets)
                handle.write(f"\n===== {time.strftime('%F %T')} {shlex.join(command)} =====\n")
                environment = os.environ.copy(); environment.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": device, "LL_HLS4ML_TQDM": "0"})
                process = subprocess.Popen(command, cwd=REPO, env=environment, stdout=handle, stderr=subprocess.STDOUT, text=True)
                job.status = "RUNNING"; active.append((process, job, handle)); persist()
                print(f"Started E5c {stage} {job.architecture_id} draw {job.draw_seed} on GPU {device}", flush=True)
            if not active:
                break
            time.sleep(2)
            remaining = []
            for process, job, handle in active:
                code = process.poll()
                if code is None:
                    remaining.append((process, job, handle)); continue
                handle.close(); available.append(job.device); job.returncode = code
                valid = code == 0 and _bundle_complete(metadata, stage, job.architecture_id, job.draw_seed, budgets)
                job.status = "COMPLETE" if valid else "FAILED"; failed |= not valid; persist()
                print(f"{job.status}: E5c {stage} {job.architecture_id} draw {job.draw_seed}", flush=True)
            active = remaining
    except KeyboardInterrupt:
        for process, job, handle in active:
            process.terminate(); process.wait(timeout=30); handle.close(); job.status = "INTERRUPTED"
        persist(); raise
    if failed:
        raise SystemExit(1)


def _resolve_inputs(args) -> tuple[dict, Path]:
    e5_root = args.e5_root.resolve()
    e5_metadata_path = e5_root / "study_metadata.json"
    old_partitions_path = e5_root / "protocol/e5_partitions.json"
    preprocessing_path = e5_root / "cache/preprocessing.pt"
    source_feature_cache = e5_root / "cache/source_features.pt"
    expected = {e5_metadata_path: EXPECTED_E5_METADATA_SHA256, old_partitions_path: EXPECTED_E5_PARTITIONS_SHA256,
                preprocessing_path: EXPECTED_PREPROCESSING_SHA256, source_feature_cache: EXPECTED_SOURCE_FEATURES_SHA256}
    for path, digest in expected.items():
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"Missing or changed frozen E5 input: {path}")
    e5 = json.loads(e5_metadata_path.read_text())
    old = json.loads(old_partitions_path.read_text())
    inputs = {
        "source_manifest": _existing_or(e5["source_manifest"], REPO / "artifacts/releases/vitis-a31-coarsearch1-hierarchy2-vocab-2026-09-04/e2_structural_v1/architecture_grouped_structural_v1.json"),
        "source_resolved_config": _existing_or(e5["source_resolved_config"], REPO / "artifacts/results/e2_replication_v1/seed42/e2_fusion_structural_seed42/resolved_config.json"),
        "source_checkpoint": _existing_or(e5["source_checkpoint"], REPO / "artifacts/results/e2_replication_v1/seed42/e2_fusion_structural_seed42/checkpoints/e2_fusion_structural_seed42_checkpoint.pt"),
        "vocab": _existing_or(e5["vocab"], REPO / "artifacts/vocab/vocab.json"),
        "high_level_cache": _existing_or(e5["high_level_cache"], REPO / "artifacts/cache/wa_high_level_archives1_32.pt"),
    }
    expected_hashes = {"source_manifest": e5["source_manifest_sha256"], "source_resolved_config": e5["source_resolved_config_sha256"],
                       "source_checkpoint": EXPECTED_SOURCE_CHECKPOINT_SHA256, "vocab": e5["vocab_sha256"], "high_level_cache": e5["high_level_cache_sha256"]}
    for name, path in inputs.items():
        if not path.is_file() or _sha256(path) != expected_hashes[name]:
            raise ValueError(f"Frozen input mismatch: {name} ({path})")
    _validate_source_manifest(inputs["source_manifest"], e5["source_split_sha256"])
    tensor_dir = (args.tensor_dir or Path(e5["tensor_dir"])).expanduser().resolve()
    if not tensor_dir.is_dir():
        fallback = Path("/home/brend/projects/data/tensors")
        tensor_dir = fallback.resolve() if fallback.is_dir() else tensor_dir
    if not args.dry_run and not (tensor_dir / "labels.json").is_file():
        raise FileNotFoundError(tensor_dir / "labels.json")
    output = args.output_dir.resolve(); provenance = output / "provenance"
    _write_stable_text(provenance / "run_e5c.py", Path(__file__).read_text())
    _write_stable_text(provenance / "run_e5.py", (REPO / "scripts/run_e5.py").read_text())
    partitions_path = output / "protocol/e5c_partitions.json"
    _write_stable_json(partitions_path, _build_partitions(old))
    protocol_path = output / "protocol/e5c_protocol.json"
    protocol = {
        "study_id": STUDY_ID, "protocol_id": PROTOCOL_ID, "registered_on": "2026-09-15",
        "status": "late-added after original E5 query inspection; not fresh external validation; outcome-blind implementation smoke predictions were generated before the production run",
        "question_P1": "Across exact total-label budgets 4, 8, 16, 32, and 64, does source-head tuning improve equal-architecture query SMAPE over zero-shot transfer?",
        "question_P2": "Across exact total-label budgets 4, 8, 16, 32, and 64, does a fresh head trained on source-standardized frozen pretrained features beat the identically initialized fresh head trained on source-standardized frozen random features?",
        "primary_budgets": list(PRIMARY_BUDGETS),
        "draw_seeds": list(DRAW_SEEDS), "head_replicates": len(HEAD_REPLICATES),
        "random_encoder_seeds": list(RANDOM_ENCODER_SEEDS), "fixed_epochs": EPOCHS,
        "graph_baseline_budgets": list(GRAPH_BUDGETS),
        "scratch_epochs": SCRATCH_EPOCHS,
        "pretrained_full_tune_epochs": FULL_TUNE_EPOCHS,
        "pretrained_full_tune_learning_rate": FULL_TUNE_LEARNING_RATE,
        "feature_standardization": {
            "reference": "unlabeled source-test features, separately per encoder",
            "std_floor": FEATURE_STD_FLOOR,
            "applies_to": list(REPLICATED_METHODS),
        },
        "query_used_for_selection": False,
        "pre_run_query_check": "one-architecture temporary predictions checked only for shape, finiteness, and paired artifact hashes; no error metric or method comparison inspected",
        "selection": "fixed epoch schedules or support-only leave-one-out cross-validation",
        "aggregation": "average initialization replicates, then support draws, then equal weight over architectures",
        "uncertainty": "hierarchical bootstrap: resample architectures, then draws within architecture",
        "primary_estimand": "equal-weight mean paired SMAPE advantage over all five log2-spaced budgets",
        "multiplicity": "97.5% curve-average intervals and Holm-adjusted exact sign-flip tests across two claims; Bonferroni 99% intervals across five budget-specific contrasts within each claim",
        "strong_claim_rule": "curve: adjusted interval excludes zero, Holm p<0.05, and at least 6/7 architecture means agree; pointwise: simultaneous interval excludes zero and at least 6/7 agree",
        "method_roles": {
            "identity_affine": "low-variance source-output calibration",
            "pretrained_residual_ridge": "frozen pretrained representation plus convex residual learner",
            "random_encoder_residual_ridge": "matched optimization-independent random-representation control",
            "final_layer_tune": "minimal source-head update",
            "source_head_tune": "pretrained encoder and pretrained complete head",
            "fresh_head_standardized_pretrained_encoder": "scale-controlled pretrained representation without source-head initialization",
            "fresh_head_standardized_random_encoder": "scale-controlled paired causal encoder-state control",
            "scratch_full": "operational target-only baseline; not an attribution control",
            "pretrained_full_tune": "standard full-model adaptation baseline; not an attribution control",
        },
    }
    _write_stable_json(protocol_path, protocol)
    metadata_path = output / "study_metadata.json"
    random_caches = {str(r): str(output / "cache" / f"random_encoder_replicate{r}.pt") for r in HEAD_REPLICATES}
    metadata = {
        "study_id": STUDY_ID, "protocol_id": PROTOCOL_ID, "output_dir": str(output),
        "metadata_path": str(metadata_path), "protocol_path": str(protocol_path),
        "partitions_path": str(partitions_path), "e5_root": str(e5_root),
        "source_feature_cache": str(source_feature_cache), "preprocessing_path": str(preprocessing_path),
        "tensor_dir": str(tensor_dir), "random_feature_caches": random_caches,
        **{name: str(path) for name, path in inputs.items()},
        "precision": PRECISION, "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor, "thread_prefetch": args.thread_prefetch,
        "evaluation_batch_size": args.evaluation_batch_size,
        "bootstrap_replicates": args.bootstrap_replicates,
        "git": _git_revision(), "input_hashes": expected_hashes,
    }
    source_state = torch.load(inputs["source_checkpoint"], map_location="cpu", weights_only=True)["model"]
    metadata["source_encoder_state_sha256"] = _tensor_state_sha256({
        name: tensor for name, tensor in source_state.items()
        if not name.startswith("classifier.") and name not in {"y_means", "y_stds"}
    })
    _write_stable_json(metadata_path, metadata)
    return metadata, metadata_path


def _load_metadata(path: Path) -> dict:
    metadata = json.loads(path.read_text())
    if metadata.get("study_id") != STUDY_ID:
        raise ValueError(f"Not E5c metadata: {path}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "fit", "evaluate", "analyze", "all"), default="all")
    parser.add_argument("--e5-root", type=Path, default=DEFAULT_E5_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tensor-dir", type=Path)
    parser.add_argument("--budgets", nargs="+", type=int, default=list(BUDGETS))
    parser.add_argument("--draw-seeds", nargs="+", type=int, default=list(DRAW_SEEDS))
    parser.add_argument("--architectures", nargs="+")
    parser.add_argument("--devices", nargs="+", default=["0"])
    parser.add_argument("--jobs-per-device", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--thread-prefetch", action="store_true")
    parser.add_argument("--evaluation-batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--worker-stage", choices=("fit", "evaluate"), help=argparse.SUPPRESS)
    parser.add_argument("--study-metadata", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--architecture", help=argparse.SUPPRESS)
    parser.add_argument("--draw-seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    budgets, draws = tuple(args.budgets), tuple(args.draw_seeds)
    if args.worker_stage:
        metadata = _load_metadata(args.study_metadata.resolve())
        (_fit_bundle if args.worker_stage == "fit" else _evaluate_bundle)(metadata, args.architecture, args.draw_seed, budgets)
        return
    if tuple(sorted(set(budgets))) != budgets or not set(budgets) <= set(BUDGETS):
        parser.error("--budgets must be a unique increasing subset of 4 8 16 32 64")
    if len(set(draws)) != len(draws) or not set(draws) <= set(DRAW_SEEDS):
        parser.error("--draw-seeds must be a unique subset of 7 42 137 271 911")
    if len(set(args.devices)) != len(args.devices) or args.jobs_per_device < 1:
        parser.error("invalid device concurrency")
    metadata, metadata_path = _resolve_inputs(args)
    partitions = json.loads(Path(metadata["partitions_path"]).read_text())
    architectures = args.architectures or sorted(partitions["architectures"])
    if not set(architectures) <= set(partitions["architectures"]):
        parser.error("unknown architecture")
    stages = ("prepare", "fit", "evaluate", "analyze") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "prepare":
            if args.dry_run:
                print(f"Prepare and hash {len(HEAD_REPLICATES)} random-encoder feature caches.")
            else:
                _prepare(metadata)
        elif stage in {"fit", "evaluate"}:
            if stage == "fit" and not args.dry_run and not (Path(metadata["output_dir"]) / "cache/prepare_summary.json").is_file():
                raise RuntimeError("Run --stage prepare first")
            if stage == "evaluate" and not args.dry_run and not _matrix_complete(metadata, "fit", sorted(partitions["architectures"]), DRAW_SEEDS, BUDGETS):
                raise RuntimeError("E5c query is sealed until the complete registered fit matrix is finished")
            _dispatch(stage, metadata, metadata_path, architectures, draws, budgets, args.devices, args.jobs_per_device, args.dry_run, args.fail_fast)
        elif args.dry_run:
            print("Analyze the complete registered E5c matrix.")
        else:
            if not _matrix_complete(metadata, "evaluate", sorted(partitions["architectures"]), DRAW_SEEDS, BUDGETS):
                raise RuntimeError("Analysis requires the complete registered query-evaluation matrix")
            _analyze(metadata)


if __name__ == "__main__":
    main()
