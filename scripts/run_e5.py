#!/usr/bin/env python3
"""Prepare, run, evaluate, and package the frozen E5 adaptation study."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
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
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.data import Data

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hls-surrogate-lab-matplotlib")

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from ll_hls4ml.data.dataset import HeteroGraphDataset
from ll_hls4ml.data.high_level import (
    CDFGHighLevelDataset,
    PROCESSED_FEATURE_DIM,
    feature_statistics,
)
from ll_hls4ml.data.protocols import SPLIT_NAMES
from ll_hls4ml.data.vocab import load_vocab
from ll_hls4ml.io.schema import LABEL_KEYS
from ll_hls4ml.models.readout import SplitRegressionHead
from ll_hls4ml.models.registry import build
from ll_hls4ml.reporting.accounting import split_sha256
from ll_hls4ml.training.loaders import make_loader
from ll_hls4ml.training.loops import _autocast, fit
from ll_hls4ml.training.targets import (
    apply_hurdle_prediction,
    denormalize_target,
    LogHuberHurdleLoss,
)
from ll_hls4ml.training.telemetry import NvidiaSmiMonitor


STUDY_ID = "e5_exemplar_design_point_adaptation_v1"
PARTITION_VERSION = "e5_fixed_query_nested_support_v1"
OFFICIAL_SPLIT_SHA256 = (
    "7d3fb85fd8664e5b2ea01b627e0f8401426098255a6c8c693f505e163160f2ca"
)
OFFICIAL_VOCAB_SHA256 = (
    "2a40117368c2536372ffdd0109198bb787227cb0ef9e7da9ac9c1b41d4f1b2bb"
)
OFFICIAL_HIGH_LEVEL_SHA256 = (
    "1fc6a8392ee583b25e8f8a5e16ffe81ed1e753a5391c3e89a843aa60e733b078"
)
OFFICIAL_TENSOR_INDEX_SHA256 = (
    "1e0e3dc80edc71098e0fd9a05c4683b304e6574944928c804570e795412db226"
)
DEFAULT_E2_ROOT = (
    _REPO_ROOT
    / "artifacts/results/e2_structural_v1/e2_structural_v1"
)
DEFAULT_SOURCE_RUN = (
    DEFAULT_E2_ROOT / "results/seed42/e2_fusion_structural_seed42"
)
DEFAULT_METHODS = ("affine", "head", "full")
METHODS = DEFAULT_METHODS
DEFAULT_BUDGETS = (0, 4, 16, 32)
DEFAULT_DRAW_SEEDS = (7, 42, 137)
SCOPES = {
    "overall": tuple(range(6)),
    "resource": (0, 1, 2, 3),
    "timing": (4, 5),
    **{target: (index,) for index, target in enumerate(LABEL_KEYS)},
}


@dataclass
class Job:
    architecture_id: str
    draw_seed: int
    stage: str
    log_path: str
    device: str | None = None
    status: str = "PENDING"
    returncode: int | None = None


class CachedFeatureDataset(Dataset):
    def __init__(self, features: torch.Tensor, targets: torch.Tensor):
        self.features = features.float()
        self.targets = targets.float()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Data:
        return Data(
            x=self.features[index].unsqueeze(0),
            y=self.targets[index].unsqueeze(0),
        )


class CachedHeadModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        y_means: torch.Tensor,
        y_stds: torch.Tensor,
        hurdle_prediction_mode: str,
    ):
        super().__init__()
        self.register_buffer("y_means", y_means.clone())
        self.register_buffer("y_stds", y_stds.clone())
        self.hurdle_prediction_mode = hurdle_prediction_mode
        self.classifier = SplitRegressionHead(
            input_dim, hidden_dim, dropout, hurdle_heads=True
        )

    def forward(self, data):
        return self.classifier(data.x)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _derived_seed(*parts: object) -> int:
    encoded = "\0".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big")


def _write_stable_json(path: Path, value: object) -> None:
    content = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != content:
            raise FileExistsError(f"Refusing to replace different metadata: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def _write_stable_text(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text() != content:
            raise FileExistsError(f"Refusing to replace different provenance: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _write_frame_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_csv_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if not rows:
        temporary.write_text("")
    else:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def _torch_save_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _git_revision() -> dict[str, object]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"], cwd=_REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    return {"commit": revision, "tracked_worktree_dirty": bool(status), "status": status}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_source_manifest(path: Path, expected_hash: str) -> tuple[dict, str]:
    manifest = json.loads(path.read_text())
    if set(manifest) != set(SPLIT_NAMES):
        raise ValueError(f"{path}: expected exactly {SPLIT_NAMES}")
    rows = [row for split in SPLIT_NAMES for row in manifest[split]]
    paths = [row["tensor_path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError(f"{path}: duplicate tensor paths")
    if any(not all(row.get("label_validity_mask", [])) for row in rows):
        raise ValueError(f"{path}: E5 requires six valid labels for every sample")
    exemplar = manifest["exemplar"]
    groups = {row["architecture_id"] for row in exemplar}
    if len(exemplar) != 886 or len(groups) != 7:
        raise ValueError(
            f"{path}: expected 886 exemplar points in seven architectures; "
            f"got {len(exemplar)} in {len(groups)}"
        )
    actual_hash = split_sha256(manifest)
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(
            f"{path}: split SHA-256 is {actual_hash}, expected {expected_hash}"
        )
    return manifest, actual_hash


def build_partitions(
    exemplar_rows: list[dict],
    budgets: tuple[int, ...],
    draw_seeds: tuple[int, ...],
    partition_seed: int,
    query_fraction: float,
    validation_fraction: float,
) -> dict:
    if not 0 < query_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("query and validation fractions must lie in (0, 1)")
    if query_fraction + validation_fraction >= 1:
        raise ValueError("query and validation fractions must sum to less than one")
    if not budgets or budgets[0] != 0 or tuple(sorted(set(budgets))) != budgets:
        raise ValueError("budgets must be unique, increasing, and begin at zero")
    if len(draw_seeds) != 3 or len(set(draw_seeds)) != 3:
        raise ValueError("E5 requires exactly three distinct support-draw seeds")

    by_architecture: dict[str, list[dict]] = {}
    for row in exemplar_rows:
        by_architecture.setdefault(row["architecture_id"], []).append(row)
    result = {
        "study_id": STUDY_ID,
        "partition_version": PARTITION_VERSION,
        "partition_seed": partition_seed,
        "query_fraction": query_fraction,
        "validation_fraction": validation_fraction,
        "budgets": list(budgets),
        "draw_seeds": list(draw_seeds),
        "architectures": {},
    }
    maximum_budget = max(budgets)
    for architecture_id, rows in sorted(by_architecture.items()):
        ordered = sorted(rows, key=lambda row: row["tensor_path"])
        rng = np.random.default_rng(
            _derived_seed(PARTITION_VERSION, partition_seed, architecture_id)
        )
        permutation = rng.permutation(len(ordered))
        query_count = int(round(len(ordered) * query_fraction))
        validation_count = int(round(len(ordered) * validation_fraction))
        query = [ordered[index]["tensor_path"] for index in permutation[:query_count]]
        validation = [
            ordered[index]["tensor_path"]
            for index in permutation[query_count:query_count + validation_count]
        ]
        support_pool = [
            ordered[index]["tensor_path"]
            for index in permutation[query_count + validation_count:]
        ]
        if len(support_pool) < maximum_budget:
            raise ValueError(
                f"Architecture {architecture_id} has only {len(support_pool)} "
                f"support candidates for k={maximum_budget}"
            )
        draws = {}
        for draw_seed in draw_seeds:
            draw_rng = np.random.default_rng(
                _derived_seed(
                    PARTITION_VERSION, partition_seed, architecture_id, draw_seed
                )
            )
            support_order = [
                support_pool[index]
                for index in draw_rng.permutation(len(support_pool))
            ]
            draws[str(draw_seed)] = {
                "support_order": support_order,
                "support": {
                    str(budget): support_order[:budget] for budget in budgets
                },
            }
        result["architectures"][architecture_id] = {
            "architecture_summary": ordered[0].get("architecture_summary"),
            "topology_id": ordered[0].get("topology_id"),
            "n_total": len(ordered),
            "query": query,
            "validation": validation,
            "support_pool": support_pool,
            "draws": draws,
        }
    audit_partitions(result)
    return result


def audit_partitions(partitions: dict) -> None:
    budgets = [int(value) for value in partitions["budgets"]]
    for architecture_id, record in partitions["architectures"].items():
        query = set(record["query"])
        validation = set(record["validation"])
        support_pool = set(record["support_pool"])
        if query & validation or query & support_pool or validation & support_pool:
            raise AssertionError(f"Partition overlap for {architecture_id}")
        if len(query | validation | support_pool) != record["n_total"]:
            raise AssertionError(f"Partition coverage mismatch for {architecture_id}")
        for draw in record["draws"].values():
            prior: set[str] = set()
            for budget in budgets:
                current = set(draw["support"][str(budget)])
                if len(current) != budget or not prior <= current:
                    raise AssertionError(
                        f"Non-nested support sets for {architecture_id} at k={budget}"
                    )
                if not current <= support_pool:
                    raise AssertionError(f"Support escaped pool for {architecture_id}")
                prior = current


def _rows_by_path(manifest: dict) -> dict[str, dict]:
    return {
        row["tensor_path"]: row
        for split in SPLIT_NAMES
        for row in manifest[split]
    }


def _targets(paths: list[str], rows_by_path: dict[str, dict]) -> torch.Tensor:
    return torch.tensor(
        [rows_by_path[path]["labels"] for path in paths], dtype=torch.float32
    )


def _load_metadata(path: Path) -> dict:
    metadata = json.loads(path.read_text())
    if metadata.get("study_id") != STUDY_ID:
        raise ValueError(f"Unexpected E5 study metadata: {path}")
    return metadata


def _load_preprocessing(metadata: dict) -> dict:
    return torch.load(
        metadata["preprocessing_path"], map_location="cpu", weights_only=True
    )


def _build_source_model(metadata: dict) -> nn.Module:
    resolved = json.loads(Path(metadata["source_resolved_config"]).read_text())
    checkpoint = torch.load(
        metadata["source_checkpoint"], map_location="cpu", weights_only=True
    )
    vocabulary, max_pos, _ = load_vocab(metadata["vocab"])
    state = checkpoint["model"]
    model = build(
        "hierarchical_high_level_fusion",
        instruction_vocab_size=len(vocabulary),
        edge_pos_vocab_size=max_pos,
        high_level_input_dim=PROCESSED_FEATURE_DIM,
        y_means=state["y_means"],
        y_stds=state["y_stds"],
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
        hurdle_prediction_mode=resolved.get(
            "hurdle_prediction_mode", "threshold"
        ),
    )
    model.load_state_dict(state, strict=True)
    return model


def _fusion_dataset(
    metadata: dict,
    paths: list[str],
    high_level_cache: dict,
    high_level_means: torch.Tensor,
    high_level_stds: torch.Tensor,
) -> CDFGHighLevelDataset:
    cdfg = HeteroGraphDataset(
        metadata["tensor_dir"], relative_paths=paths, silent=True
    )
    return CDFGHighLevelDataset(
        cdfg, range(len(paths)), high_level_cache,
        high_level_means, high_level_stds,
    )


def _raw_prediction(model: nn.Module, logits: torch.Tensor) -> torch.Tensor:
    normalized = apply_hurdle_prediction(
        logits,
        model.y_means,
        model.y_stds,
        mode=model.hurdle_prediction_mode,
    )
    return denormalize_target(normalized, model.y_means, model.y_stds).clamp_min(0)


def _predict_model(
    model: nn.Module,
    loader,
    device: torch.device,
    precision: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.to(device).eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            with _autocast(device, precision):
                logits = model(batch)
            predictions.append(_raw_prediction(model, logits).float().cpu())
            targets.append(batch.y.view(-1, len(LABEL_KEYS)).float().cpu())
    return torch.cat(predictions), torch.cat(targets)


def _prepare_cache(metadata: dict) -> None:
    summary_path = Path(metadata["cache_summary_path"])
    cache_path = Path(metadata["feature_cache_path"])
    preprocessing_path = Path(metadata["preprocessing_path"])
    if summary_path.is_file() and cache_path.is_file() and preprocessing_path.is_file():
        summary = json.loads(summary_path.read_text())
        if (
            summary.get("feature_cache_sha256") != _sha256(cache_path)
            or summary.get("preprocessing_sha256") != _sha256(preprocessing_path)
        ):
            raise RuntimeError("Prepared E5 cache hash mismatch")
        print(f"Prepared cache already complete: {cache_path}")
        return
    if summary_path.exists():
        raise RuntimeError(
            "E5 cache completion marker exists without valid artifacts"
        )

    manifest = json.loads(Path(metadata["source_manifest"]).read_text())
    high_level_cache = torch.load(
        metadata["high_level_cache"], map_location="cpu", weights_only=False
    )
    if preprocessing_path.is_file():
        preprocessing = torch.load(
            preprocessing_path, map_location="cpu", weights_only=True
        )
        high_level_means = preprocessing["high_level_means"]
        high_level_stds = preprocessing["high_level_stds"]
    else:
        train_paths = [row["tensor_path"] for row in manifest["train"]]
        high_level_means, high_level_stds = feature_statistics(
            high_level_cache, train_paths
        )
        source_model = _build_source_model(metadata)
        preprocessing = {
            "high_level_means": high_level_means,
            "high_level_stds": high_level_stds,
            "y_means": source_model.y_means.cpu(),
            "y_stds": source_model.y_stds.cpu(),
        }
        _torch_save_atomic(preprocessing_path, preprocessing)

    paths = [
        row["tensor_path"]
        for split in ("test", "exemplar")
        for row in manifest[split]
    ]
    if cache_path.is_file():
        feature_cache = torch.load(
            cache_path, map_location="cpu", weights_only=True
        )
        if feature_cache.get("format_version") != 1 or feature_cache.get("paths") != paths:
            raise RuntimeError("Partial E5 feature cache does not match this study")
    else:
        model = _build_source_model(metadata)
        dataset = _fusion_dataset(
            metadata, paths, high_level_cache, high_level_means, high_level_stds
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        precision = metadata["precision"]
        model.to(device).eval()
        features = []
        predictions = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device, non_blocking=device.type == "cuda")
                with _autocast(device, precision):
                    encoded = model.encode(batch)
                    logits = model.classifier(encoded)
                features.append(encoded.float().cpu())
                predictions.append(_raw_prediction(model, logits).float().cpu())
        feature_cache = {
            "format_version": 1,
            "paths": paths,
            "features": torch.cat(features),
            "source_predictions": torch.cat(predictions),
        }
        _torch_save_atomic(cache_path, feature_cache)
    if (
        feature_cache["features"].shape[0] != len(paths)
        or feature_cache["source_predictions"].shape != (len(paths), len(LABEL_KEYS))
    ):
        raise RuntimeError("E5 feature cache has invalid tensor dimensions")

    expected = pd.read_csv(metadata["source_predictions"])
    expected = expected[expected["split"].isin(["test", "exemplar"])].set_index(
        "tensor_path"
    )
    actual = feature_cache["source_predictions"].numpy()
    wanted = expected.loc[paths, [f"prediction_{key}" for key in LABEL_KEYS]].to_numpy()
    smape_difference = np.mean(
        200 * np.abs(actual - wanted) / (np.abs(actual) + np.abs(wanted) + 1.0)
    )
    if smape_difference > metadata["source_reconstruction_smape_tolerance"]:
        raise ValueError(
            "Reconstructed source predictions do not match the frozen E2 bundle: "
            f"mean pairwise SMAPE={smape_difference:.6f}"
        )
    _write_json_atomic(summary_path, {
        "status": "complete",
        "samples": len(paths),
        "feature_width": int(feature_cache["features"].shape[1]),
        "source_test_samples": len(manifest["test"]),
        "exemplar_samples": len(manifest["exemplar"]),
        "source_reconstruction_mean_pairwise_smape": float(smape_difference),
        "feature_cache_sha256": _sha256(cache_path),
        "preprocessing_sha256": _sha256(preprocessing_path),
    })
    print(f"Prepared frozen E5 feature cache: {cache_path}")


def _cache_rows(cache: dict, paths: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    position = {path: index for index, path in enumerate(cache["paths"])}
    indices = torch.tensor([position[path] for path in paths], dtype=torch.long)
    return cache["features"][indices], cache["source_predictions"][indices]


def _run_dir(metadata: dict, architecture_id: str, draw_seed: int, method: str, budget: int) -> Path:
    return (
        Path(metadata["output_dir"]) / "runs" / architecture_id
        / f"draw{draw_seed}" / f"{method}_k{budget}"
    )


def fit_affine_coefficients(
    support_prediction: np.ndarray,
    support_target: np.ndarray,
    validation_prediction: np.ndarray,
    validation_target: np.ndarray,
    ridge_grid: list[float],
) -> list[dict]:
    """Select independent log-target affine maps using validation data only."""
    support_log_x = np.log1p(np.clip(support_prediction, 0, None))
    support_log_y = np.log1p(np.clip(support_target, 0, None))
    validation_log_x = np.log1p(np.clip(validation_prediction, 0, None))
    coefficients = []
    for target_index, target_name in enumerate(LABEL_KEYS):
        design = np.column_stack([
            np.ones(len(support_log_x)), support_log_x[:, target_index]
        ])
        prior = np.asarray([0.0, 1.0])
        candidates = []
        for ridge in ridge_grid:
            ridge = float(ridge)
            if ridge < 0:
                raise ValueError("affine ridge penalties must be non-negative")
            penalty = ridge * np.eye(2)
            normal_matrix = design.T @ design + penalty
            right_hand_side = (
                design.T @ support_log_y[:, target_index] + penalty @ prior
            )
            # k=4 can be rank-deficient when source predictions are constant.
            # lstsq gives the minimum-norm OLS solution at ridge=0 and is stable
            # for every candidate in the deliberately tiny calibration problem.
            theta = np.linalg.lstsq(
                normal_matrix, right_hand_side, rcond=None
            )[0]
            prediction = np.expm1(
                theta[0] + theta[1] * validation_log_x[:, target_index]
            ).clip(min=0)
            target = validation_target[:, target_index]
            smape = float(np.mean(
                200 * np.abs(target - prediction)
                / (np.abs(target) + np.abs(prediction) + 1.0)
            ))
            candidates.append((smape, -ridge, theta, ridge))
        smape, _negative_ridge, theta, ridge = min(
            candidates, key=lambda row: row[:2]
        )
        coefficients.append({
            "target": target_name,
            "intercept": float(theta[0]),
            "slope": float(theta[1]),
            "ridge": ridge,
            "validation_smape": smape,
        })
    return coefficients


def _fit_affine(
    metadata: dict,
    architecture_id: str,
    draw_seed: int,
    budget: int,
    support_paths: list[str],
    validation_paths: list[str],
    rows_by_path: dict[str, dict],
    cache: dict,
) -> None:
    run_dir = _run_dir(metadata, architecture_id, draw_seed, "affine", budget)
    marker = run_dir / "fit_summary.json"
    calibration_path = run_dir / "calibration.json"
    if marker.is_file() and calibration_path.is_file():
        summary = json.loads(marker.read_text())
        if summary.get("calibration_sha256") != _sha256(calibration_path):
            raise RuntimeError(f"Affine calibration hash mismatch: {run_dir}")
        return
    if marker.exists() or calibration_path.exists():
        raise RuntimeError(f"Incomplete affine fit artifacts: {run_dir}")
    seed = _derived_seed(STUDY_ID, architecture_id, draw_seed, "affine", budget)
    config = {
        "study_id": STUDY_ID,
        "architecture_id": architecture_id,
        "draw_seed": draw_seed,
        "method": "affine",
        "budget": budget,
        "seed": seed,
        "ridge_grid": metadata["affine_ridge_grid"],
        "selection_split": "adaptation_validation",
    }
    _write_stable_json(run_dir / "run_config.json", config)
    started = time.perf_counter()
    _, support_prediction = _cache_rows(cache, support_paths)
    _, validation_prediction = _cache_rows(cache, validation_paths)
    support_target = _targets(support_paths, rows_by_path).numpy()
    validation_target = _targets(validation_paths, rows_by_path).numpy()
    coefficients = fit_affine_coefficients(
        support_prediction.numpy(), support_target,
        validation_prediction.numpy(), validation_target,
        metadata["affine_ridge_grid"],
    )
    wall_seconds = time.perf_counter() - started
    _write_json_atomic(calibration_path, {
        "space": "log1p_target",
        "identity_prior": {"intercept": 0.0, "slope": 1.0},
        "coefficients": coefficients,
    })
    _write_json_atomic(marker, {
        "status": "complete",
        "method": "affine",
        "budget": budget,
        "architecture_id": architecture_id,
        "draw_seed": draw_seed,
        "trainable_parameters": 2 * len(LABEL_KEYS),
        "wall_seconds": wall_seconds,
        "calibration_sha256": _sha256(calibration_path),
        "validation_macro_smape": float(np.mean([
            row["validation_smape"] for row in coefficients
        ])),
    })


def _source_head(metadata: dict, cache: dict) -> CachedHeadModel:
    resolved = json.loads(Path(metadata["source_resolved_config"]).read_text())
    checkpoint = torch.load(
        metadata["source_checkpoint"], map_location="cpu", weights_only=True
    )
    state = checkpoint["model"]
    model = CachedHeadModel(
        input_dim=int(cache["features"].shape[1]),
        hidden_dim=int(resolved.get("hidden_dim", 64)),
        dropout=float(resolved.get("dropout", 0.15)),
        y_means=state["y_means"],
        y_stds=state["y_stds"],
        hurdle_prediction_mode=resolved.get(
            "hurdle_prediction_mode", "threshold"
        ),
    )
    classifier_state = {
        key.removeprefix("classifier."): value
        for key, value in state.items()
        if key.startswith("classifier.")
    }
    model.classifier.load_state_dict(classifier_state, strict=True)
    return model


def _check_neural_run_state(
    run_dir: Path,
    marker: Path,
    best_checkpoint: Path,
    backup: Path,
    resume: bool,
) -> bool:
    """Return true for a valid completion and reject unsafe partial restarts."""
    if marker.is_file() and best_checkpoint.is_file():
        summary = json.loads(marker.read_text())
        if summary.get("checkpoint_sha256") != _sha256(best_checkpoint):
            raise RuntimeError(f"Completed checkpoint hash mismatch: {run_dir}")
        return True
    if marker.exists() or best_checkpoint.exists():
        raise RuntimeError(f"Inconsistent completion artifacts: {run_dir}")
    partial_entries = (
        [path for path in run_dir.iterdir() if path.name != "run_config.json"]
        if run_dir.is_dir() else []
    )
    if not partial_entries:
        return False
    if not resume:
        raise RuntimeError(f"Partial run requires --resume: {run_dir}")
    if not backup.is_file():
        raise RuntimeError(f"Partial run has no resumable backup checkpoint: {run_dir}")
    return False


def _archive_telemetry_segment(run_dir: Path) -> None:
    """Keep telemetry from earlier process lifetimes when resuming a fit."""
    for name in ("gpu_telemetry.csv", "system_telemetry.csv"):
        source = run_dir / name
        if not source.is_file():
            continue
        segment = 1
        while (run_dir / f"{source.stem}.segment{segment}.csv").exists():
            segment += 1
        source.replace(run_dir / f"{source.stem}.segment{segment}.csv")


def _fit_head(
    metadata: dict,
    architecture_id: str,
    draw_seed: int,
    budget: int,
    support_paths: list[str],
    validation_paths: list[str],
    rows_by_path: dict[str, dict],
    cache: dict,
    resume: bool,
) -> None:
    run_dir = _run_dir(metadata, architecture_id, draw_seed, "head", budget)
    marker = run_dir / "fit_summary.json"
    checkpoint_dir = run_dir / "checkpoints"
    experiment = f"e5_head_{architecture_id}_draw{draw_seed}_k{budget}"
    best_checkpoint = checkpoint_dir / f"{experiment}_checkpoint.pt"
    backup = checkpoint_dir / f"{experiment}_backup.pt"
    if _check_neural_run_state(
        run_dir, marker, best_checkpoint, backup, resume
    ):
        return
    seed = _derived_seed(STUDY_ID, architecture_id, draw_seed, "head", budget)
    config = {
        "study_id": STUDY_ID,
        "architecture_id": architecture_id,
        "draw_seed": draw_seed,
        "method": "head",
        "budget": budget,
        "seed": seed,
        "learning_rate": metadata["head_learning_rate"],
        "weight_decay": metadata["head_weight_decay"],
        "epochs": metadata["head_epochs"],
        "patience": metadata["head_patience"],
        "selection_split": "adaptation_validation",
        "source_checkpoint_sha256": metadata["source_checkpoint_sha256"],
    }
    _write_stable_json(run_dir / "run_config.json", config)
    _set_seed(seed)
    model = _source_head(metadata, cache)
    support_features, _ = _cache_rows(cache, support_paths)
    validation_features, _ = _cache_rows(cache, validation_paths)
    train_dataset = CachedFeatureDataset(
        support_features, _targets(support_paths, rows_by_path)
    )
    validation_dataset = CachedFeatureDataset(
        validation_features, _targets(validation_paths, rows_by_path)
    )
    train_loader = make_loader(
        train_dataset, batch_size=min(metadata["head_batch_size"], budget),
        shuffle=True, num_workers=0, pin_memory=True,
    )
    validation_loader = make_loader(
        validation_dataset, batch_size=metadata["evaluation_batch_size"],
        shuffle=False, num_workers=0, pin_memory=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=metadata["head_learning_rate"],
        weight_decay=metadata["head_weight_decay"],
    )
    criterion = LogHuberHurdleLoss(
        model.y_means,
        model.y_stds,
        delta=metadata["log_huber_delta"],
        classification_weight=metadata["hurdle_classification_weight"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    if resume and backup.is_file():
        _archive_telemetry_segment(run_dir)
    monitor = NvidiaSmiMonitor(
        run_dir / "gpu_telemetry.csv",
        interval_ms=metadata["gpu_telemetry_interval_ms"],
        gpu=os.environ.get("E5_MONITOR_GPU", "0"),
    )
    monitor.start()
    try:
        model = fit(
            model,
            train_loader,
            validation_loader,
            epochs=metadata["head_epochs"],
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,
            device=device,
            patience=metadata["head_patience"],
            mode="min",
            restore_best_weights=True,
            verbose=metadata["verbose"],
            experiment_name=experiment,
            checkpoint_dir=checkpoint_dir,
            resume_from_backup=backup if resume and backup.is_file() else None,
            early_stopping_metric="smape",
            precision=metadata["precision"],
            checkpoint_interval=1,
            max_training_seconds=metadata["max_training_seconds_per_run"],
            history_path=run_dir / "learning_curves.csv",
            gradient_clip_norm=metadata["gradient_clip_norm"],
            lr_scheduler_patience=metadata["lr_scheduler_patience"],
            lr_scheduler_factor=metadata["lr_scheduler_factor"],
            min_learning_rate=metadata["min_learning_rate"],
        )
    finally:
        monitor.stop()
    _write_json_atomic(marker, {
        "status": "complete",
        "method": "head",
        "budget": budget,
        "architecture_id": architecture_id,
        "draw_seed": draw_seed,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "wall_seconds": time.perf_counter() - started,
        "best_epoch": model.best_epoch,
        "best_validation_smape": model.best_metric,
        "stop_reason": model.stop_reason,
        "gpu_telemetry": monitor.summary(),
        "checkpoint_sha256": _sha256(best_checkpoint),
    })


def _fit_full(
    metadata: dict,
    architecture_id: str,
    draw_seed: int,
    budget: int,
    support_paths: list[str],
    validation_paths: list[str],
    high_level_cache: dict,
    resume: bool,
) -> None:
    run_dir = _run_dir(metadata, architecture_id, draw_seed, "full", budget)
    marker = run_dir / "fit_summary.json"
    checkpoint_dir = run_dir / "checkpoints"
    experiment = f"e5_full_{architecture_id}_draw{draw_seed}_k{budget}"
    best_checkpoint = checkpoint_dir / f"{experiment}_checkpoint.pt"
    backup = checkpoint_dir / f"{experiment}_backup.pt"
    if _check_neural_run_state(
        run_dir, marker, best_checkpoint, backup, resume
    ):
        return
    seed = _derived_seed(STUDY_ID, architecture_id, draw_seed, "full", budget)
    config = {
        "study_id": STUDY_ID,
        "architecture_id": architecture_id,
        "draw_seed": draw_seed,
        "method": "full",
        "budget": budget,
        "seed": seed,
        "learning_rate": metadata["full_learning_rate"],
        "weight_decay": metadata["full_weight_decay"],
        "epochs": metadata["full_epochs"],
        "patience": metadata["full_patience"],
        "selection_split": "adaptation_validation",
        "source_checkpoint_sha256": metadata["source_checkpoint_sha256"],
    }
    _write_stable_json(run_dir / "run_config.json", config)
    _set_seed(seed)
    model = _build_source_model(metadata)
    preprocessing = _load_preprocessing(metadata)
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
        batch_size=min(metadata["full_batch_size"], budget),
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
        model.parameters(),
        lr=metadata["full_learning_rate"],
        weight_decay=metadata["full_weight_decay"],
    )
    criterion = LogHuberHurdleLoss(
        model.y_means,
        model.y_stds,
        delta=metadata["log_huber_delta"],
        classification_weight=metadata["hurdle_classification_weight"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    if resume and backup.is_file():
        _archive_telemetry_segment(run_dir)
    monitor = NvidiaSmiMonitor(
        run_dir / "gpu_telemetry.csv",
        interval_ms=metadata["gpu_telemetry_interval_ms"],
        gpu=os.environ.get("E5_MONITOR_GPU", "0"),
    )
    monitor.start()
    try:
        model = fit(
            model,
            train_loader,
            validation_loader,
            epochs=metadata["full_epochs"],
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,
            device=device,
            patience=metadata["full_patience"],
            mode="min",
            restore_best_weights=True,
            verbose=metadata["verbose"],
            experiment_name=experiment,
            checkpoint_dir=checkpoint_dir,
            resume_from_backup=backup if resume and backup.is_file() else None,
            early_stopping_metric="smape",
            precision=metadata["precision"],
            checkpoint_interval=1,
            max_training_seconds=metadata["max_training_seconds_per_run"],
            history_path=run_dir / "learning_curves.csv",
            gradient_clip_norm=metadata["gradient_clip_norm"],
            lr_scheduler_patience=metadata["lr_scheduler_patience"],
            lr_scheduler_factor=metadata["lr_scheduler_factor"],
            min_learning_rate=metadata["min_learning_rate"],
        )
    finally:
        monitor.stop()
    _write_json_atomic(marker, {
        "status": "complete",
        "method": "full",
        "budget": budget,
        "architecture_id": architecture_id,
        "draw_seed": draw_seed,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "wall_seconds": time.perf_counter() - started,
        "best_epoch": model.best_epoch,
        "best_validation_smape": model.best_metric,
        "stop_reason": model.stop_reason,
        "gpu_telemetry": monitor.summary(),
        "checkpoint_sha256": _sha256(best_checkpoint),
    })


def _selected_specs(methods: tuple[str, ...], budgets: tuple[int, ...]):
    for method in methods:
        selected = (
            (32,) if method == "full" and 32 in budgets
            else () if method == "full"
            else tuple(k for k in budgets if k > 0)
        )
        for budget in selected:
            yield method, budget


def _fit_bundle(
    metadata: dict,
    architecture_id: str,
    draw_seed: int,
    methods: tuple[str, ...],
    budgets: tuple[int, ...],
    resume: bool,
) -> None:
    manifest = json.loads(Path(metadata["source_manifest"]).read_text())
    rows_by_path = _rows_by_path(manifest)
    partitions = json.loads(Path(metadata["partitions_path"]).read_text())
    record = partitions["architectures"][architecture_id]
    validation_paths = record["validation"]
    draw = record["draws"][str(draw_seed)]
    cache = torch.load(
        metadata["feature_cache_path"], map_location="cpu", weights_only=True
    )
    high_level_cache = None
    for method, budget in _selected_specs(methods, budgets):
        support_paths = draw["support"][str(budget)]
        print(
            f"FIT architecture={architecture_id} draw={draw_seed} "
            f"method={method} k={budget}",
            flush=True,
        )
        if method == "affine":
            _fit_affine(
                metadata, architecture_id, draw_seed, budget,
                support_paths, validation_paths, rows_by_path, cache,
            )
        elif method == "head":
            _fit_head(
                metadata, architecture_id, draw_seed, budget,
                support_paths, validation_paths, rows_by_path, cache, resume,
            )
        elif method == "full":
            if high_level_cache is None:
                high_level_cache = torch.load(
                    metadata["high_level_cache"],
                    map_location="cpu",
                    weights_only=False,
                )
            _fit_full(
                metadata, architecture_id, draw_seed, budget,
                support_paths, validation_paths, high_level_cache, resume,
            )


def _apply_affine(predictions: torch.Tensor, calibration: dict) -> torch.Tensor:
    log_predictions = torch.log1p(predictions.clamp_min(0))
    output = torch.empty_like(predictions)
    by_target = {row["target"]: row for row in calibration["coefficients"]}
    for index, target in enumerate(LABEL_KEYS):
        row = by_target[target]
        output[:, index] = torch.expm1(
            row["intercept"] + row["slope"] * log_predictions[:, index]
        ).clamp_min(0)
    return output


def _prediction_rows(
    paths: list[str],
    split: str,
    predictions: torch.Tensor,
    rows_by_path: dict[str, dict],
    architecture_id: str,
    draw_seed: int | None,
    method: str,
    budget: int,
) -> list[dict]:
    result = []
    for position, path in enumerate(paths):
        source = rows_by_path[path]
        row = {
            "split": split,
            "adapted_for_architecture": architecture_id,
            "draw_seed": draw_seed,
            "method": method,
            "budget": budget,
            "tensor_path": path,
            "kernel_family": source["kernel_family"],
            "architecture_id": source["architecture_id"],
        }
        for index, target in enumerate(LABEL_KEYS):
            row[f"target_{target}"] = float(source["labels"][index])
            row[f"prediction_{target}"] = float(predictions[position, index])
        result.append(row)
    return result


def _evaluate_zero_shot(metadata: dict) -> None:
    run_dir = Path(metadata["output_dir"]) / "runs/zero_shot"
    marker = run_dir / "evaluation_summary.json"
    prediction_path = run_dir / "predictions.csv"
    if marker.is_file() and prediction_path.is_file():
        summary = json.loads(marker.read_text())
        if summary.get("predictions_sha256") != _sha256(prediction_path):
            raise RuntimeError("Zero-shot prediction hash mismatch")
        return
    if marker.exists() or prediction_path.exists():
        raise RuntimeError("Incomplete zero-shot evaluation artifacts")
    manifest = json.loads(Path(metadata["source_manifest"]).read_text())
    rows_by_path = _rows_by_path(manifest)
    partitions = json.loads(Path(metadata["partitions_path"]).read_text())
    cache = torch.load(
        metadata["feature_cache_path"], map_location="cpu", weights_only=True
    )
    output = []
    for architecture_id, record in partitions["architectures"].items():
        paths = record["query"]
        _, predictions = _cache_rows(cache, paths)
        output.extend(_prediction_rows(
            paths, "query", predictions, rows_by_path,
            architecture_id, None, "zero_shot", 0,
        ))
    source_paths = [row["tensor_path"] for row in manifest["test"]]
    _, source_predictions = _cache_rows(cache, source_paths)
    output.extend(_prediction_rows(
        source_paths, "source_test", source_predictions, rows_by_path,
        "all", None, "zero_shot", 0,
    ))
    _write_csv_atomic(prediction_path, output)
    _write_json_atomic(marker, {
        "status": "complete",
        "method": "zero_shot",
        "query_samples": sum(
            len(record["query"])
            for record in partitions["architectures"].values()
        ),
        "source_test_samples": len(source_paths),
        "predictions_sha256": _sha256(prediction_path),
    })


def _evaluate_bundle(
    metadata: dict,
    architecture_id: str,
    draw_seed: int,
    methods: tuple[str, ...],
    budgets: tuple[int, ...],
) -> None:
    manifest = json.loads(Path(metadata["source_manifest"]).read_text())
    rows_by_path = _rows_by_path(manifest)
    partitions = json.loads(Path(metadata["partitions_path"]).read_text())
    partition = partitions["architectures"][architecture_id]
    query_paths = partition["query"]
    validation_paths = partition["validation"]
    source_paths = [row["tensor_path"] for row in manifest["test"]]
    cache = torch.load(
        metadata["feature_cache_path"], map_location="cpu", weights_only=True
    )
    high_level_cache = None
    preprocessing = None
    for method, budget in _selected_specs(methods, budgets):
        run_dir = _run_dir(metadata, architecture_id, draw_seed, method, budget)
        marker = run_dir / "evaluation_summary.json"
        prediction_path = run_dir / "predictions.csv"
        if marker.is_file() and prediction_path.is_file():
            summary = json.loads(marker.read_text())
            if summary.get("predictions_sha256") != _sha256(prediction_path):
                raise RuntimeError(f"Prediction hash mismatch: {run_dir}")
            continue
        if not (run_dir / "fit_summary.json").is_file():
            raise FileNotFoundError(f"Fit is incomplete: {run_dir}")
        print(
            f"EVALUATE architecture={architecture_id} draw={draw_seed} "
            f"method={method} k={budget}",
            flush=True,
        )
        support_paths = partition["draws"][str(draw_seed)]["support"][str(budget)]
        combined_paths = [
            *support_paths, *validation_paths, *query_paths, *source_paths
        ]
        if method == "affine":
            _, source_prediction = _cache_rows(cache, combined_paths)
            calibration = json.loads((run_dir / "calibration.json").read_text())
            predictions = _apply_affine(source_prediction, calibration)
        elif method == "head":
            model = _source_head(metadata, cache)
            experiment = (
                f"e5_head_{architecture_id}_draw{draw_seed}_k{budget}"
            )
            checkpoint_path = run_dir / "checkpoints" / f"{experiment}_checkpoint.pt"
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            model.load_state_dict(checkpoint["model"], strict=True)
            features, _ = _cache_rows(cache, combined_paths)
            targets = _targets(combined_paths, rows_by_path)
            dataset = CachedFeatureDataset(features, targets)
            loader = make_loader(
                dataset,
                batch_size=metadata["evaluation_batch_size"],
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            predictions, _ = _predict_model(
                model, loader, device, metadata["precision"]
            )
        else:
            if high_level_cache is None:
                high_level_cache = torch.load(
                    metadata["high_level_cache"],
                    map_location="cpu",
                    weights_only=False,
                )
                preprocessing = _load_preprocessing(metadata)
            model = _build_source_model(metadata)
            experiment = (
                f"e5_full_{architecture_id}_draw{draw_seed}_k{budget}"
            )
            checkpoint_path = run_dir / "checkpoints" / f"{experiment}_checkpoint.pt"
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            model.load_state_dict(checkpoint["model"], strict=True)
            dataset = _fusion_dataset(
                metadata, combined_paths, high_level_cache,
                preprocessing["high_level_means"],
                preprocessing["high_level_stds"],
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
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            predictions, _ = _predict_model(
                model, loader, device, metadata["precision"]
            )
        support_end = len(support_paths)
        validation_end = support_end + len(validation_paths)
        query_end = validation_end + len(query_paths)
        rows = _prediction_rows(
            support_paths, "support", predictions[:support_end], rows_by_path,
            architecture_id, draw_seed, method, budget,
        )
        rows.extend(_prediction_rows(
            validation_paths, "adaptation_validation",
            predictions[support_end:validation_end], rows_by_path,
            architecture_id, draw_seed, method, budget,
        ))
        rows.extend(_prediction_rows(
            query_paths, "query", predictions[validation_end:query_end], rows_by_path,
            architecture_id, draw_seed, method, budget,
        ))
        rows.extend(_prediction_rows(
            source_paths, "source_test", predictions[query_end:], rows_by_path,
            architecture_id, draw_seed, method, budget,
        ))
        _write_csv_atomic(prediction_path, rows)
        _write_json_atomic(marker, {
            "status": "complete",
            "method": method,
            "budget": budget,
            "architecture_id": architecture_id,
            "draw_seed": draw_seed,
            "support_samples": len(support_paths),
            "adaptation_validation_samples": len(validation_paths),
            "query_samples": len(query_paths),
            "source_test_samples": len(source_paths),
            "predictions_sha256": _sha256(prediction_path),
        })


def _metric_record(frame: pd.DataFrame) -> dict[str, float]:
    errors = []
    result = {}
    for target in LABEL_KEYS:
        truth = frame[f"target_{target}"].to_numpy(float)
        prediction = frame[f"prediction_{target}"].to_numpy(float)
        error = 200 * np.abs(truth - prediction) / (
            np.abs(truth) + np.abs(prediction) + 1.0
        )
        errors.append(error)
        denominator = np.square(truth - truth.mean()).sum()
        result[f"smape_{target}"] = float(error.mean())
        result[f"r2_{target}"] = (
            float("nan") if denominator == 0
            else float(1 - np.square(truth - prediction).sum() / denominator)
        )
    matrix = np.asarray(errors).T
    for scope, positions in SCOPES.items():
        result[f"smape_{scope}"] = float(matrix[:, positions].mean())
    return result


def _hierarchical_interval(
    frame: pd.DataFrame,
    value: str,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    by_architecture = {
        architecture: group[value].to_numpy(float)
        for architecture, group in frame.groupby("architecture_id")
    }
    architectures = sorted(by_architecture)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates)
    for replicate in range(replicates):
        selected = rng.choice(architectures, len(architectures), replace=True)
        estimates[replicate] = np.mean([
            rng.choice(by_architecture[architecture])
            for architecture in selected
        ])
    return tuple(np.percentile(estimates, [2.5, 97.5]))


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without an optional tabulate dependency."""
    columns = [str(column) for column in frame.columns]

    def cell(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = ["| " + " | ".join(columns) + " |"]
    rows.append("| " + " | ".join("---" for _ in columns) + " |")
    rows.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def _analyze(
    metadata: dict,
    methods: tuple[str, ...],
    budgets: tuple[int, ...],
) -> None:
    partitions = json.loads(Path(metadata["partitions_path"]).read_text())
    architectures = sorted(partitions["architectures"])
    draw_seeds = tuple(partitions["draw_seeds"])
    metric_rows = []
    zero_path = Path(metadata["output_dir"]) / "runs/zero_shot/predictions.csv"
    zero = pd.read_csv(zero_path)
    for architecture_id in architectures:
        frame = zero.query(
            "split == 'query' and adapted_for_architecture == @architecture_id"
        )
        metric_rows.append({
            "method": "zero_shot",
            "budget": 0,
            "architecture_id": architecture_id,
            "draw_seed": np.nan,
            "split": "query",
            "n_samples": len(frame),
            **_metric_record(frame),
        })
    source_zero = zero[zero["split"] == "source_test"]
    metric_rows.append({
        "method": "zero_shot",
        "budget": 0,
        "architecture_id": "all",
        "draw_seed": np.nan,
        "split": "source_test",
        "n_samples": len(source_zero),
        **_metric_record(source_zero),
    })
    workload = []
    for architecture_id in architectures:
        for draw_seed in draw_seeds:
            for method, budget in _selected_specs(methods, budgets):
                run_dir = _run_dir(
                    metadata, architecture_id, draw_seed, method, budget
                )
                prediction_path = run_dir / "predictions.csv"
                if not prediction_path.is_file():
                    raise FileNotFoundError(prediction_path)
                frame = pd.read_csv(prediction_path)
                for split in ("query", "source_test"):
                    selected = frame[frame["split"] == split]
                    metric_rows.append({
                        "method": method,
                        "budget": budget,
                        "architecture_id": architecture_id,
                        "draw_seed": draw_seed,
                        "split": split,
                        "n_samples": len(selected),
                        **_metric_record(selected),
                    })
                fit_summary = json.loads(
                    (run_dir / "fit_summary.json").read_text()
                )
                workload.append({
                    "method": method,
                    "budget": budget,
                    "architecture_id": architecture_id,
                    "draw_seed": draw_seed,
                    **{
                        key: fit_summary.get(key)
                        for key in (
                            "trainable_parameters", "wall_seconds", "best_epoch",
                            "best_validation_smape", "stop_reason",
                        )
                    },
                })
    metrics = pd.DataFrame(metric_rows)
    analysis_dir = Path(metadata["output_dir"]) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _write_frame_atomic(analysis_dir / "per_run_metrics.csv", metrics)
    _write_frame_atomic(analysis_dir / "workload.csv", pd.DataFrame(workload))

    summary_rows = []
    query = metrics[metrics["split"] == "query"]
    for (method, budget), frame in query.groupby(["method", "budget"]):
        for scope in SCOPES:
            value = f"smape_{scope}"
            architecture_means = frame.groupby("architecture_id")[value].mean()
            low, high = _hierarchical_interval(
                frame, value, metadata["bootstrap_replicates"],
                _derived_seed(STUDY_ID, "summary", method, budget, scope),
            )
            summary_rows.append({
                "method": method,
                "budget": int(budget),
                "scope": scope,
                "equal_architecture_smape": float(architecture_means.mean()),
                "ci95_low": low,
                "ci95_high": high,
                "architectures": len(architecture_means),
                "draws_per_architecture": int(frame.groupby("architecture_id").size().max()),
            })
    summary = pd.DataFrame(summary_rows)
    _write_frame_atomic(analysis_dir / "equal_architecture_summary.csv", summary)

    zero_by_architecture = query[query["method"] == "zero_shot"].set_index(
        "architecture_id"
    )
    paired_rows = []
    for (method, budget), frame in query[query["method"] != "zero_shot"].groupby(
        ["method", "budget"]
    ):
        paired = frame.copy()
        for scope in SCOPES:
            value = f"smape_{scope}"
            paired[f"delta_{scope}"] = [
                row[value] - zero_by_architecture.loc[row["architecture_id"], value]
                for _, row in paired.iterrows()
            ]
            low, high = _hierarchical_interval(
                paired, f"delta_{scope}", metadata["bootstrap_replicates"],
                _derived_seed(STUDY_ID, "paired", method, budget, scope),
            )
            architecture_means = paired.groupby("architecture_id")[f"delta_{scope}"].mean()
            paired_rows.append({
                "method": method,
                "budget": int(budget),
                "scope": scope,
                "delta_smape_vs_zero_shot": float(architecture_means.mean()),
                "ci95_low": low,
                "ci95_high": high,
                "architecture_win_fraction": float((architecture_means < 0).mean()),
            })
    paired_summary = pd.DataFrame(paired_rows, columns=(
        "method", "budget", "scope", "delta_smape_vs_zero_shot",
        "ci95_low", "ci95_high", "architecture_win_fraction",
    ))
    _write_frame_atomic(analysis_dir / "paired_vs_zero_shot.csv", paired_summary)

    source = metrics[metrics["split"] == "source_test"].copy()
    source_baseline = source[source["method"] == "zero_shot"].iloc[0]
    degradation_rows = []
    for (method, budget), frame in source[source["method"] != "zero_shot"].groupby(
        ["method", "budget"]
    ):
        for scope in SCOPES:
            value = f"smape_{scope}"
            frame = frame.copy()
            frame["degradation"] = frame[value] - source_baseline[value]
            low, high = _hierarchical_interval(
                frame, "degradation", metadata["bootstrap_replicates"],
                _derived_seed(STUDY_ID, "degradation", method, budget, scope),
            )
            degradation_rows.append({
                "method": method,
                "budget": int(budget),
                "scope": scope,
                "source_smape": float(frame.groupby("architecture_id")[value].mean().mean()),
                "source_zero_shot_smape": float(source_baseline[value]),
                "source_degradation_smape": float(
                    frame.groupby("architecture_id")["degradation"].mean().mean()
                ),
                "ci95_low": low,
                "ci95_high": high,
            })
    degradation = pd.DataFrame(degradation_rows, columns=(
        "method", "budget", "scope", "source_smape",
        "source_zero_shot_smape", "source_degradation_smape",
        "ci95_low", "ci95_high",
    ))
    _write_frame_atomic(analysis_dir / "source_degradation.csv", degradation)
    _write_analysis_figures(
        analysis_dir, summary, query, paired_summary, degradation
    )
    report = _analysis_report(summary, paired_summary, degradation, workload)
    _write_text_atomic(analysis_dir / "REPORT.md", report)
    _write_json_atomic(analysis_dir / "analysis_provenance.json", {
        "study_id": STUDY_ID,
        "study_metadata_sha256": _sha256(Path(metadata["metadata_path"])),
        "partitions_sha256": _sha256(Path(metadata["partitions_path"])),
        "methods": list(methods),
        "budgets": list(budgets),
        "bootstrap_replicates": metadata["bootstrap_replicates"],
        "outer_unit": "exemplar architecture_id",
        "support_draws_are_repeated_measures": True,
        "query_opened_only_after_fit_completion": True,
    })
    inventory = []
    output_root = Path(metadata["output_dir"])
    inventory_path = analysis_dir / "artifact_inventory.csv"
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path != inventory_path and not path.name.endswith(".tmp"):
            inventory.append({
                "relative_path": str(path.relative_to(output_root)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
    _write_frame_atomic(inventory_path, pd.DataFrame(inventory))


def _write_analysis_figures(
    output: Path,
    summary: pd.DataFrame,
    query: pd.DataFrame,
    paired: pd.DataFrame,
    degradation: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt

    method_order = ["zero_shot", "affine", "head", "full"]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    for axis, scope in zip(axes, SCOPES):
        selected = summary[summary["scope"] == scope]
        for method in method_order:
            rows = selected[selected["method"] == method].sort_values("budget")
            if rows.empty:
                continue
            x = rows["budget"].to_numpy(dtype=float)
            axis.plot(
                x, rows["equal_architecture_smape"].to_numpy(dtype=float),
                marker="o", label=method,
            )
            axis.fill_between(
                x, rows["ci95_low"].to_numpy(dtype=float),
                rows["ci95_high"].to_numpy(dtype=float), alpha=0.15,
            )
        axis.set_title(scope)
        axis.set_xlabel("support labels k")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("equal-architecture SMAPE")
    axes[-1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "adaptation_curves.png", dpi=180)
    plt.close(figure)

    overall = query.groupby(
        ["method", "budget", "architecture_id"], as_index=False
    )["smape_overall"].mean()
    figure, axis = plt.subplots(figsize=(9, 5))
    for (method, architecture), rows in overall.groupby(["method", "architecture_id"]):
        axis.plot(
            rows["budget"], rows["smape_overall"], marker="o", alpha=0.6,
            label=f"{method}:{architecture[:6]}",
        )
    axis.set_xlabel("support labels k")
    axis.set_ylabel("query SMAPE")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "per_architecture_curves.png", dpi=180)
    plt.close(figure)

    left = paired[paired["scope"] == "overall"].copy()
    right = degradation[degradation["scope"] == "overall"].copy()
    tradeoff = left.merge(right, on=["method", "budget"])
    figure, axis = plt.subplots(figsize=(6, 5))
    for _, row in tradeoff.iterrows():
        axis.scatter(-row["delta_smape_vs_zero_shot"], row["source_degradation_smape"])
        axis.annotate(f"{row['method']} k={int(row['budget'])}", (
            -row["delta_smape_vs_zero_shot"], row["source_degradation_smape"]
        ))
    axis.axhline(0, color="black", linewidth=0.8)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("query improvement over zero-shot (SMAPE points)")
    axis.set_ylabel("source-test degradation (SMAPE points)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "adaptation_source_tradeoff.png", dpi=180)
    plt.close(figure)


def _analysis_report(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    degradation: pd.DataFrame,
    workload: list[dict],
) -> str:
    headline = summary[summary["scope"] == "overall"].copy()
    headline["result"] = headline.apply(
        lambda row: (
            f"{row['equal_architecture_smape']:.2f} "
            f"[{row['ci95_low']:.2f}, {row['ci95_high']:.2f}]"
        ), axis=1,
    )
    table = _markdown_table(headline[["method", "budget", "result"]])
    paired_overall = paired[paired["scope"] == "overall"]
    paired_table = _markdown_table(paired_overall)
    degradation_overall = degradation[degradation["scope"] == "overall"]
    degradation_table = _markdown_table(degradation_overall)
    total_wall = sum(float(row.get("wall_seconds") or 0) for row in workload)
    return f"""# E5 exemplar design-point adaptation

## Contract

- Frozen source: E2 seed-42 fusion checkpoint.
- Outer reporting unit: seven exemplar architecture IDs.
- Support budgets: 0, 4, 16, and 32 with three nested deterministic draws.
- Checkpoint and hyperparameter selection: adaptation-validation only.
- Query labels opened only during the frozen evaluation stage.

## Equal-architecture query results

{table}

## Paired improvement relative to zero-shot

{paired_table}

## Source-domain degradation

{degradation_table}

## Compute

Summed per-run fitting time: {total_wall / 3600:.2f} hours. Concurrent wall time is lower.

All per-run predictions, fit histories, telemetry, partitions, resolved settings,
and bootstrap-ready tables are retained alongside this report.
"""


def _fit_complete(
    metadata: dict,
    architecture_id: str,
    draw_seed: int,
    methods: tuple[str, ...],
    budgets: tuple[int, ...],
) -> bool:
    for method, budget in _selected_specs(methods, budgets):
        run_dir = _run_dir(metadata, architecture_id, draw_seed, method, budget)
        marker = run_dir / "fit_summary.json"
        if not marker.is_file():
            return False
        if method == "affine":
            artifact = run_dir / "calibration.json"
        else:
            experiment = f"e5_{method}_{architecture_id}_draw{draw_seed}_k{budget}"
            artifact = run_dir / "checkpoints" / f"{experiment}_checkpoint.pt"
        if not artifact.is_file():
            return False
    return True


def _evaluation_complete(
    metadata: dict,
    architecture_id: str,
    draw_seed: int,
    methods: tuple[str, ...],
    budgets: tuple[int, ...],
) -> bool:
    for method, budget in _selected_specs(methods, budgets):
        run_dir = _run_dir(metadata, architecture_id, draw_seed, method, budget)
        if not (run_dir / "evaluation_summary.json").is_file() or not (
            run_dir / "predictions.csv"
        ).is_file():
            return False
    return True


def _all_core_fits_complete(metadata: dict) -> bool:
    partitions = json.loads(Path(metadata["partitions_path"]).read_text())
    return all(
        _fit_complete(
            metadata, architecture_id, draw_seed,
            DEFAULT_METHODS, DEFAULT_BUDGETS,
        )
        for architecture_id in partitions["architectures"]
        for draw_seed in partitions["draw_seeds"]
    )


def _fit_bundle_block_status(
    metadata: dict,
    architecture_id: str,
    draw_seed: int,
    methods: tuple[str, ...],
    budgets: tuple[int, ...],
    resume: bool,
) -> str | None:
    for method, budget in _selected_specs(methods, budgets):
        if method == "affine":
            continue
        run_dir = _run_dir(metadata, architecture_id, draw_seed, method, budget)
        experiment = f"e5_{method}_{architecture_id}_draw{draw_seed}_k{budget}"
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


def _write_index(path: Path, jobs: list[Job]) -> None:
    _write_json_atomic(path, [asdict(job) for job in jobs])


def _worker_command(
    stage: str,
    metadata_path: Path,
    job: Job,
    methods: tuple[str, ...],
    budgets: tuple[int, ...],
    resume: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-stage", stage,
        "--study-metadata", str(metadata_path),
        "--architecture", job.architecture_id,
        "--draw-seed", str(job.draw_seed),
        "--methods", *methods,
        "--budgets", *map(str, budgets),
    ]
    if resume:
        command.append("--resume")
    return command


def _gpu_preflight(device: str) -> None:
    environment = os.environ.copy()
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = device
    result = subprocess.run(
        [
            sys.executable, "-c",
            "import torch; assert torch.cuda.is_available(); "
            "print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory)",
        ],
        env=environment,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"CUDA preflight failed for {device}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    print(f"GPU {device}: {result.stdout.strip()}", flush=True)


def _dispatch(
    stage: str,
    metadata: dict,
    metadata_path: Path,
    architectures: list[str],
    draw_seeds: tuple[int, ...],
    methods: tuple[str, ...],
    budgets: tuple[int, ...],
    devices: list[str],
    jobs_per_device: int,
    resume: bool,
    fail_fast: bool,
    dry_run: bool,
) -> None:
    complete = _fit_complete if stage == "fit" else _evaluation_complete
    jobs = []
    slots = [device for device in devices for _ in range(jobs_per_device)]
    position = 0
    for architecture_id in architectures:
        for draw_seed in draw_seeds:
            status = (
                "SKIPPED_COMPLETE"
                if complete(
                    metadata, architecture_id, draw_seed, methods, budgets
                ) else "PENDING"
            )
            if stage == "fit" and status == "PENDING":
                status = _fit_bundle_block_status(
                    metadata, architecture_id, draw_seed,
                    methods, budgets, resume,
                ) or status
            jobs.append(Job(
                architecture_id=architecture_id,
                draw_seed=draw_seed,
                stage=stage,
                log_path=str(
                    Path(metadata["output_dir"]) / "logs" / stage
                    / f"{architecture_id}_draw{draw_seed}.log"
                ),
                device=slots[position % len(slots)],
                status=status,
            ))
            position += 1
    output_root = Path(metadata["output_dir"])
    index_path = output_root / f"{stage}_index.json"
    invocation_path = (
        output_root / "logs/invocations"
        / f"{time.strftime('%Y%m%dT%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}_{stage}.json"
    )

    def persist_indexes() -> None:
        _write_index(index_path, jobs)
        _write_json_atomic(invocation_path, {
            "stage": stage,
            "methods": list(methods),
            "budgets": list(budgets),
            "draw_seeds": list(draw_seeds),
            "architectures": list(architectures),
            "devices": list(devices),
            "jobs_per_device": jobs_per_device,
            "resume": resume,
            "dry_run": dry_run,
            "jobs": [asdict(job) for job in jobs],
        })

    persist_indexes()
    blocked = [job for job in jobs if job.status.startswith("BLOCKED")]
    if blocked:
        for job in blocked:
            print(
                f"{job.status}: {job.architecture_id} draw {job.draw_seed}; "
                f"inspect {job.log_path}",
                flush=True,
            )
        raise SystemExit(2)
    pending = [job for job in jobs if job.status == "PENDING"]
    if dry_run:
        for job in pending:
            command = _worker_command(
                stage, metadata_path, job, methods, budgets, resume
            )
            print(
                f"CUDA_VISIBLE_DEVICES={shlex.quote(str(job.device))} "
                + shlex.join(command)
            )
        print(f"{stage}: runnable={len(pending)} index={index_path}")
        return
    if not pending:
        print(f"All selected E5 {stage} bundles are complete.")
        return
    for device in dict.fromkeys(job.device for job in pending):
        _gpu_preflight(str(device))
    available = list(slots)
    active: list[tuple[subprocess.Popen, Job, object]] = []
    failed = False
    try:
        while pending or active:
            while pending and available and not (failed and fail_fast):
                job = pending.pop(0)
                device = available.pop(0)
                job.device = device
                command = _worker_command(
                    stage, metadata_path, job, methods, budgets, resume
                )
                log_path = Path(job.log_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_handle = log_path.open("a", buffering=1)
                log_handle.write(
                    f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"{shlex.join(command)} =====\n"
                )
                environment = os.environ.copy()
                environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                environment["CUDA_VISIBLE_DEVICES"] = device
                environment["E5_MONITOR_GPU"] = device
                environment["LL_HLS4ML_TQDM"] = "0"
                process = subprocess.Popen(
                    command,
                    cwd=_REPO_ROOT,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                job.status = "RUNNING"
                active.append((process, job, log_handle))
                print(
                    f"Started E5 {stage} {job.architecture_id} draw "
                    f"{job.draw_seed} on GPU {device}; log={job.log_path}",
                    flush=True,
                )
                persist_indexes()
            if not active:
                break
            time.sleep(2)
            remaining = []
            for process, job, log_handle in active:
                returncode = process.poll()
                if returncode is None:
                    remaining.append((process, job, log_handle))
                    continue
                log_handle.close()
                available.append(str(job.device))
                job.returncode = returncode
                valid = returncode == 0 and complete(
                    metadata, job.architecture_id, job.draw_seed, methods, budgets
                )
                job.status = "COMPLETE" if valid else "FAILED"
                failed = failed or not valid
                print(
                    f"{job.status}: {stage} {job.architecture_id} "
                    f"draw {job.draw_seed}",
                    flush=True,
                )
                persist_indexes()
            active = remaining
        if failed and fail_fast:
            for job in pending:
                job.status = "SKIPPED_FAIL_FAST"
            persist_indexes()
    except KeyboardInterrupt:
        for process, job, log_handle in active:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log_handle.close()
            job.status = "INTERRUPTED"
            job.returncode = process.returncode
        persist_indexes()
        raise
    if failed:
        raise SystemExit(1)


def _metadata_from_args(args) -> tuple[dict, Path]:
    source_manifest, source_split_hash = _validate_source_manifest(
        args.source_manifest, args.expected_source_split_sha256
    )
    hashes = {
        "vocab_sha256": _sha256(args.vocab),
        "high_level_cache_sha256": _sha256(args.high_level_cache),
        "tensor_index_sha256": _sha256(args.tensor_index),
        "source_checkpoint_sha256": _sha256(args.source_checkpoint),
        "source_resolved_config_sha256": _sha256(args.source_resolved_config),
        "source_predictions_sha256": _sha256(args.source_predictions),
        "source_manifest_sha256": _sha256(args.source_manifest),
    }
    expected = {
        "vocab_sha256": args.expected_vocab_sha256,
        "high_level_cache_sha256": args.expected_high_level_sha256,
        "tensor_index_sha256": args.expected_tensor_index_sha256,
    }
    for key, wanted in expected.items():
        if wanted and hashes[key] != wanted:
            raise ValueError(f"{key} is {hashes[key]}, expected {wanted}")
    resolved = json.loads(args.source_resolved_config.read_text())
    if resolved.get("model") != "hierarchical_high_level_fusion":
        raise ValueError("E5 source must be the E2 fusion model")
    if resolved.get("seed") != 42 or resolved.get("split_sha256") != source_split_hash:
        raise ValueError("E5 source checkpoint identity does not match E2 seed 42")
    checkpoint = torch.load(
        args.source_checkpoint, map_location="cpu", weights_only=True
    )
    if "model" not in checkpoint or "y_means" not in checkpoint["model"]:
        raise ValueError("Source checkpoint is incomplete")

    partitions = build_partitions(
        source_manifest["exemplar"],
        DEFAULT_BUDGETS,
        DEFAULT_DRAW_SEEDS,
        args.partition_seed,
        args.query_fraction,
        args.validation_fraction,
    )
    output = args.output_dir
    provenance_dir = output / "provenance"
    script_snapshot = provenance_dir / "run_e5.py"
    fusion_snapshot = provenance_dir / "fusion.py"
    script_source = Path(__file__).resolve()
    fusion_source = _REPO_ROOT / "src/ll_hls4ml/models/fusion.py"
    _write_stable_text(script_snapshot, script_source.read_text())
    _write_stable_text(fusion_snapshot, fusion_source.read_text())
    environment_path = provenance_dir / "environment.json"
    _write_stable_json(environment_path, {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    })
    partitions_path = output / "protocol/e5_partitions.json"
    partitions.update({
        "source_manifest": str(args.source_manifest),
        "source_manifest_sha256": hashes["source_manifest_sha256"],
        "source_split_sha256": source_split_hash,
        "git": _git_revision(),
        "script_snapshot": str(script_snapshot),
        "script_snapshot_sha256": _sha256(script_snapshot),
        "fusion_snapshot": str(fusion_snapshot),
        "fusion_snapshot_sha256": _sha256(fusion_snapshot),
        "environment_path": str(environment_path),
        "environment_sha256": _sha256(environment_path),
        "outer_unit": "architecture_id",
        "query_role": "opened only after all selected fits complete",
    })
    _write_stable_json(partitions_path, partitions)
    metadata_path = output / "study_metadata.json"
    metadata = {
        "study_id": STUDY_ID,
        "partition_version": PARTITION_VERSION,
        "output_dir": str(output),
        "metadata_path": str(metadata_path),
        "partitions_path": str(partitions_path),
        "feature_cache_path": str(output / "cache/source_features.pt"),
        "preprocessing_path": str(output / "cache/preprocessing.pt"),
        "cache_summary_path": str(output / "cache/cache_summary.json"),
        "tensor_dir": str(args.tensor_dir),
        "tensor_index": str(args.tensor_index),
        "vocab": str(args.vocab),
        "high_level_cache": str(args.high_level_cache),
        "source_manifest": str(args.source_manifest),
        "source_checkpoint": str(args.source_checkpoint),
        "source_resolved_config": str(args.source_resolved_config),
        "source_predictions": str(args.source_predictions),
        "source_split_sha256": source_split_hash,
        **hashes,
        "budgets": list(DEFAULT_BUDGETS),
        "draw_seeds": list(DEFAULT_DRAW_SEEDS),
        "query_fraction": args.query_fraction,
        "validation_fraction": args.validation_fraction,
        "partition_seed": args.partition_seed,
        "precision": args.precision,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "thread_prefetch": args.thread_prefetch,
        "evaluation_batch_size": args.evaluation_batch_size,
        "head_batch_size": args.head_batch_size,
        "head_learning_rate": args.head_learning_rate,
        "head_weight_decay": args.head_weight_decay,
        "head_epochs": args.head_epochs,
        "head_patience": args.head_patience,
        "full_batch_size": args.full_batch_size,
        "full_learning_rate": args.full_learning_rate,
        "full_weight_decay": args.full_weight_decay,
        "full_epochs": args.full_epochs,
        "full_patience": args.full_patience,
        "log_huber_delta": args.log_huber_delta,
        "hurdle_classification_weight": args.hurdle_classification_weight,
        "gradient_clip_norm": args.gradient_clip_norm,
        "lr_scheduler_patience": args.lr_scheduler_patience,
        "lr_scheduler_factor": args.lr_scheduler_factor,
        "min_learning_rate": args.min_learning_rate,
        "max_training_seconds_per_run": args.max_training_seconds_per_run,
        "gpu_telemetry_interval_ms": args.gpu_telemetry_interval_ms,
        "verbose": args.verbose,
        "affine_ridge_grid": args.affine_ridge_grid,
        "bootstrap_replicates": args.bootstrap_replicates,
        "source_reconstruction_smape_tolerance": (
            args.source_reconstruction_smape_tolerance
        ),
    }
    _write_stable_json(metadata_path, metadata)
    return metadata, metadata_path


def _resolve_main_paths(args) -> None:
    args.tensor_dir = args.tensor_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.vocab = args.vocab.resolve()
    args.tensor_index = (args.tensor_index or args.tensor_dir / "labels.json").resolve()
    for name in (
        "high_level_cache", "source_manifest", "source_checkpoint",
        "source_resolved_config", "source_predictions",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if not args.tensor_dir.is_dir():
        raise FileNotFoundError(args.tensor_dir)
    for path in (
        args.tensor_index, args.vocab, args.high_level_cache,
        args.source_manifest, args.source_checkpoint,
        args.source_resolved_config, args.source_predictions,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)


def _run_prepare_worker(metadata_path: Path, device: str, dry_run: bool) -> None:
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker-stage", "prepare",
        "--study-metadata", str(metadata_path),
    ]
    if dry_run:
        print(
            f"CUDA_VISIBLE_DEVICES={shlex.quote(device)} " + shlex.join(command)
        )
        return
    _gpu_preflight(device)
    metadata = _load_metadata(metadata_path)
    log_path = Path(metadata["output_dir"]) / "logs/prepare.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = device
    environment["E5_MONITOR_GPU"] = device
    environment["LL_HLS4ML_TQDM"] = "0"
    with log_path.open("a") as log:
        result = subprocess.run(
            command, cwd=_REPO_ROOT, env=environment,
            stdout=log, stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(f"E5 preparation failed; see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Inspect the complete 21-bundle schedule without running it.
  python scripts/run_e5.py --tensor-dir TENSORS --output-dir RESULTS --dry-run

  # Run the complete frozen study, resuming any interrupted neural fits.
  python scripts/run_e5.py --tensor-dir TENSORS --output-dir RESULTS --resume

  # Fill one section of the fit matrix; query evaluation remains sealed.
  python scripts/run_e5.py --stage fit --methods full --budgets 32 \\
      --draw-seeds 7 --tensor-dir TENSORS --output-dir RESULTS --resume

The same output directory is reused for every stage. Frozen metadata prevents
accidental changes to the dataset, source checkpoint, split, or hyperparameters.
""",
    )
    parser.add_argument(
        "--stage", choices=("prepare", "fit", "evaluate", "analyze", "all"),
        default="all", help="pipeline section to run (default: all)",
    )
    parser.add_argument("--tensor-dir", type=Path, help="frozen tensor snapshot")
    parser.add_argument(
        "--tensor-index", type=Path,
        help="tensor index to hash (default: TENSOR_DIR/labels.json)",
    )
    parser.add_argument("--output-dir", type=Path, help="persistent E5 study root")
    parser.add_argument("--vocab", type=Path, default=_REPO_ROOT / "artifacts/vocab/vocab.json")
    parser.add_argument(
        "--high-level-cache", type=Path,
        default=_REPO_ROOT / "artifacts/cache/wa_high_level_archives1_32.pt",
    )
    parser.add_argument(
        "--source-manifest", type=Path,
        default=DEFAULT_E2_ROOT / "protocol/architecture_grouped_structural_v1.json",
    )
    parser.add_argument(
        "--source-checkpoint", type=Path,
        default=DEFAULT_SOURCE_RUN / "checkpoints/e2_fusion_structural_seed42_checkpoint.pt",
    )
    parser.add_argument(
        "--source-resolved-config", type=Path,
        default=DEFAULT_SOURCE_RUN / "resolved_config.json",
    )
    parser.add_argument(
        "--source-predictions", type=Path,
        default=DEFAULT_SOURCE_RUN / "predictions.csv",
    )
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(DEFAULT_METHODS),
        help="fit/evaluate method subset (default: affine head full)",
    )
    parser.add_argument(
        "--budgets", nargs="+", type=int, default=list(DEFAULT_BUDGETS),
        help="execution subset of the frozen budgets 0 4 16 32",
    )
    parser.add_argument(
        "--draw-seeds", nargs="+", type=int, default=list(DEFAULT_DRAW_SEEDS),
        help="execution subset of frozen support draws 7 42 137",
    )
    parser.add_argument(
        "--architectures", nargs="+",
        help="execution subset of architecture IDs (default: all seven)",
    )
    parser.add_argument("--partition-seed", type=int, default=20260910)
    parser.add_argument("--query-fraction", type=float, default=0.4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--devices", nargs="+", default=["0"],
        help="CUDA indices or UUIDs (default: 0)",
    )
    parser.add_argument(
        "--jobs-per-device", type=int, default=2,
        help="independent CPU/IO-heavy workers per GPU (default: 2)",
    )
    parser.add_argument("--precision", choices=("float32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--thread-prefetch", action="store_true")
    parser.add_argument("--evaluation-batch-size", type=int, default=16)
    parser.add_argument("--head-batch-size", type=int, default=32)
    parser.add_argument("--head-learning-rate", type=float, default=0.001)
    parser.add_argument("--head-weight-decay", type=float, default=0.0001)
    parser.add_argument("--head-epochs", type=int, default=300)
    parser.add_argument("--head-patience", type=int, default=30)
    parser.add_argument("--full-batch-size", type=int, default=8)
    parser.add_argument("--full-learning-rate", type=float, default=0.00001)
    parser.add_argument("--full-weight-decay", type=float, default=0.0001)
    parser.add_argument("--full-epochs", type=int, default=200)
    parser.add_argument("--full-patience", type=int, default=25)
    parser.add_argument("--log-huber-delta", type=float, default=0.35)
    parser.add_argument("--hurdle-classification-weight", type=float, default=0.25)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--lr-scheduler-patience", type=int, default=8)
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--min-learning-rate", type=float, default=1e-7)
    parser.add_argument("--max-training-seconds-per-run", type=float)
    parser.add_argument("--gpu-telemetry-interval-ms", type=int, default=1000)
    parser.add_argument("--verbose", type=int, default=5)
    parser.add_argument(
        "--affine-ridge-grid", nargs="+", type=float,
        default=[0.0, 0.0001, 0.01, 1.0, 100.0],
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--expected-source-split-sha256", default=OFFICIAL_SPLIT_SHA256)
    parser.add_argument("--expected-vocab-sha256", default=OFFICIAL_VOCAB_SHA256)
    parser.add_argument(
        "--expected-high-level-sha256", default=OFFICIAL_HIGH_LEVEL_SHA256
    )
    parser.add_argument(
        "--expected-tensor-index-sha256", default=OFFICIAL_TENSOR_INDEX_SHA256
    )
    parser.add_argument(
        "--source-reconstruction-smape-tolerance", type=float, default=0.25
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="continue partial neural runs from epoch-boundary backups",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="freeze metadata and print the schedule without GPU work",
    )
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="stop launching new bundles after the first failure",
    )
    parser.add_argument("--worker-stage", choices=("prepare", "fit", "evaluate"), help=argparse.SUPPRESS)
    parser.add_argument("--study-metadata", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--architecture", help=argparse.SUPPRESS)
    parser.add_argument("--draw-seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    methods = tuple(dict.fromkeys(args.methods))
    budgets = tuple(sorted(set(args.budgets)))
    if args.worker_stage:
        if args.study_metadata is None:
            parser.error("worker mode requires --study-metadata")
        metadata = _load_metadata(args.study_metadata.resolve())
        if args.worker_stage == "prepare":
            _prepare_cache(metadata)
            return
        if args.architecture is None or args.draw_seed is None:
            parser.error("fit/evaluate workers require architecture and draw seed")
        if args.worker_stage == "fit":
            _fit_bundle(
                metadata, args.architecture, args.draw_seed,
                methods, budgets, args.resume,
            )
        else:
            _evaluate_bundle(
                metadata, args.architecture, args.draw_seed, methods, budgets
            )
        return

    if args.tensor_dir is None or args.output_dir is None:
        parser.error("--tensor-dir and --output-dir are required")
    if args.jobs_per_device < 1:
        parser.error("--jobs-per-device must be positive")
    if args.num_workers < 0 or args.prefetch_factor < 1:
        parser.error("--num-workers must be non-negative and --prefetch-factor positive")
    if min(args.head_epochs, args.full_epochs, args.head_patience, args.full_patience) < 1:
        parser.error("epoch and patience values must be positive")
    if args.bootstrap_replicates < 100:
        parser.error("--bootstrap-replicates must be at least 100")
    if len(args.devices) != len(set(args.devices)):
        parser.error("--devices must not contain duplicates")
    if methods != tuple(args.methods):
        parser.error("--methods must not contain duplicates")
    if budgets != tuple(args.budgets) or not set(budgets) <= set(DEFAULT_BUDGETS):
        parser.error("--budgets must be a unique increasing subset of 0 4 16 32")
    if not budgets:
        parser.error("--budgets must not be empty")
    if (
        len(args.draw_seeds) != len(set(args.draw_seeds))
        or not set(args.draw_seeds) <= set(DEFAULT_DRAW_SEEDS)
    ):
        parser.error("--draw-seeds must be a unique subset of 7 42 137")
    if not args.draw_seeds:
        parser.error("--draw-seeds must not be empty")
    if (
        args.partition_seed != 20260910
        or args.query_fraction != 0.4
        or args.validation_fraction != 0.2
    ):
        parser.error(
            "the frozen E5 core requires partition seed 20260910, "
            "query fraction 0.4, and validation fraction 0.2"
        )
    _resolve_main_paths(args)
    metadata, metadata_path = _metadata_from_args(args)
    partitions = json.loads(Path(metadata["partitions_path"]).read_text())
    available_architectures = sorted(partitions["architectures"])
    architectures = args.architectures or available_architectures
    unknown = set(architectures) - set(available_architectures)
    if unknown:
        parser.error(f"unknown architecture IDs: {sorted(unknown)}")
    if args.stage == "all" and (
        set(architectures) != set(available_architectures)
        or methods != DEFAULT_METHODS
        or budgets != DEFAULT_BUDGETS
        or tuple(args.draw_seeds) != DEFAULT_DRAW_SEEDS
    ):
        parser.error(
            "--stage all is the complete core study; use an explicit stage for subsets"
        )
    if args.stage == "analyze" and set(architectures) != set(available_architectures):
        parser.error("analysis requires all seven architecture IDs")
    draw_seeds = tuple(args.draw_seeds)

    stages = (
        ("prepare", "fit", "evaluate", "analyze")
        if args.stage == "all" else (args.stage,)
    )
    for stage in stages:
        if stage == "prepare":
            _run_prepare_worker(metadata_path, args.devices[0], args.dry_run)
        elif stage in {"fit", "evaluate"}:
            if not Path(metadata["cache_summary_path"]).is_file() and not args.dry_run:
                raise RuntimeError("Run --stage prepare before fitting/evaluation")
            if stage == "evaluate" and not args.dry_run:
                if not _all_core_fits_complete(metadata):
                    raise RuntimeError(
                        "Query evaluation is sealed until every mandatory E5 core fit "
                        "is complete across all architectures, draws, and methods"
                    )
                _evaluate_zero_shot(metadata)
            _dispatch(
                stage, metadata, metadata_path, architectures, draw_seeds,
                methods, budgets, args.devices, args.jobs_per_device,
                args.resume, args.fail_fast, args.dry_run,
            )
        else:
            if args.dry_run:
                print(f"Analyze completed evaluations under {metadata['output_dir']}")
            else:
                _analyze(metadata, methods, budgets)


if __name__ == "__main__":
    main()
