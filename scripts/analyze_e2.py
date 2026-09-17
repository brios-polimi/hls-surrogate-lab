#!/usr/bin/env python3
"""Validate E2 predictions and compute architecture-cluster bootstrap contrasts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from ll_hls4ml.io.schema import LABEL_KEYS
from ll_hls4ml.reporting.accounting import split_sha256


def _prediction_arg(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("Use NAME=/path/to/predictions.csv")
    return name, Path(path)


def _name_value_arg(value: str) -> tuple[str, str]:
    name, separator, item = value.partition("=")
    if not separator or not name or not item:
        raise argparse.ArgumentTypeError("Use NAME=VALUE")
    return name, item


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_configs(
    predictions: dict[str, Path],
    *,
    manifest_split_hash: str,
    training_seed: int | None,
    expected_models: dict[str, str],
) -> tuple[dict[str, dict], int]:
    resolved = {}
    for name, prediction_path in predictions.items():
        path = prediction_path.resolve().parent / "resolved_config.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"Prediction bundle lacks resolved_config.json: {prediction_path}"
            )
        config = json.loads(path.read_text())
        if config.get("split_sha256") != manifest_split_hash:
            raise ValueError(
                f"Split hash mismatch for {name}: "
                f"{config.get('split_sha256')} != {manifest_split_hash}"
            )
        expected_model = expected_models.get(name)
        if expected_model is not None and config.get("model") != expected_model:
            raise ValueError(
                f"Model mismatch for {name}: {config.get('model')!r} "
                f"!= {expected_model!r}"
            )
        resolved[name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "model": config.get("model"),
            "seed": config.get("seed"),
            "split_sha256": config.get("split_sha256"),
            "parameter_count": config.get("parameter_count"),
            "causal_control": config.get("causal_control"),
            "ll_hls4ml_git": config.get("ll_hls4ml_git"),
        }
    seeds = {details["seed"] for details in resolved.values()}
    if None in seeds or len(seeds) != 1:
        raise ValueError(f"Predictions do not share one recorded training seed: {seeds}")
    recorded_seed = int(next(iter(seeds)))
    if training_seed is not None and recorded_seed != training_seed:
        raise ValueError(
            f"Recorded training seed {recorded_seed} != requested {training_seed}"
        )
    return resolved, recorded_seed


def _smape_matrix(frame: pd.DataFrame) -> np.ndarray:
    values = []
    for target in LABEL_KEYS:
        truth = frame[f"target_{target}"].to_numpy(float)
        prediction = frame[f"prediction_{target}"].to_numpy(float)
        values.append(200 * np.abs(truth - prediction) / (
            np.abs(truth) + np.abs(prediction) + 1.0
        ))
    return np.asarray(values).T


def _smape(frame: pd.DataFrame) -> np.ndarray:
    return _smape_matrix(frame).mean(axis=1)


def _macro_r2(frame: pd.DataFrame, positions: tuple[int, ...]) -> float:
    values = []
    for index in positions:
        target = LABEL_KEYS[index]
        truth = frame[f"target_{target}"].to_numpy(float)
        prediction = frame[f"prediction_{target}"].to_numpy(float)
        denominator = np.square(truth - truth.mean()).sum()
        values.append(
            np.nan if denominator == 0
            else 1.0 - np.square(truth - prediction).sum() / denominator
        )
    values = np.asarray(values, dtype=float)
    return float("nan") if np.isnan(values).all() else float(np.nanmean(values))


def _cluster_bootstrap(
    values: np.ndarray, group_ids: np.ndarray, *, seed: int, replicates: int,
) -> tuple[float, float, float]:
    groups, inverse = np.unique(group_ids, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    sizes = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    batch = 250
    for start in range(0, replicates, batch):
        count = min(batch, replicates - start)
        sampled = rng.integers(0, len(groups), size=(count, len(groups)))
        estimates[start:start + count] = (
            sums[sampled].sum(axis=1) / sizes[sampled].sum(axis=1)
        )
    low, high = np.percentile(estimates, (2.5, 97.5))
    return float(low), float(high), float(np.mean(estimates >= 0))


def _validated_split(
    path: Path, expected: pd.DataFrame, split: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["split"] == split].copy()
    if frame["tensor_path"].duplicated().any():
        raise ValueError(f"Duplicate {split} prediction paths in {path}")
    actual = set(frame["tensor_path"])
    wanted = set(expected["tensor_path"])
    if actual != wanted:
        raise ValueError(
            f"{split} membership mismatch in {path}: "
            f"missing={len(wanted - actual)}, extra={len(actual - wanted)}"
        )
    # Neural prediction bundles preserve the full manifest row, including
    # ``labels`` and the signature fields.  Drop those duplicate provenance
    # columns after verifying them so the merge leaves one canonical copy.
    shared = [
        column for column in expected
        if column != "tensor_path" and column in frame
    ]
    indexed_expected = expected.set_index("tensor_path")
    indexed_actual = frame.set_index("tensor_path")
    for column in shared:
        left = indexed_expected.loc[sorted(wanted), column]
        right = indexed_actual.loc[sorted(wanted), column]
        if column == "labels":
            if left.map(str).tolist() != right.map(str).tolist():
                raise ValueError(f"Manifest labels differ in {path}")
        elif left.astype(str).tolist() != right.astype(str).tolist():
            raise ValueError(f"Manifest field {column!r} differs in {path}")
    frame = expected.merge(
        frame.drop(columns=shared), on="tensor_path", validate="one_to_one"
    )
    for index, target in enumerate(LABEL_KEYS):
        if not np.allclose(
            frame[f"target_{target}"].to_numpy(float),
            frame["labels"].map(lambda values: values[index]).to_numpy(float),
            rtol=1e-6, atol=1e-6,
        ):
            raise ValueError(f"Target mismatch for {target} in {path}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prediction", action="append", type=_prediction_arg, required=True)
    parser.add_argument(
        "--expected-model", action="append", type=_name_value_arg, default=[],
        help="Optional NAME=MODEL identity check against resolved_config.json",
    )
    parser.add_argument("--reference", default="h0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--bootstrap-seed", type=int)
    parser.add_argument(
        "--seed", type=int,
        help="Deprecated alias for --bootstrap-seed",
    )
    parser.add_argument("--replicates", type=int, default=10_000)
    args = parser.parse_args()

    predictions = dict(args.prediction)
    if len(predictions) != len(args.prediction):
        raise ValueError("Prediction names must be unique")
    if args.reference not in predictions:
        raise ValueError(f"Reference {args.reference!r} was not supplied")
    expected_models = dict(args.expected_model)
    if len(expected_models) != len(args.expected_model):
        raise ValueError("Expected-model names must be unique")
    unknown_expected = set(expected_models) - set(predictions)
    if unknown_expected:
        raise ValueError(
            f"Expected-model entries lack predictions: {sorted(unknown_expected)}"
        )
    manifest = json.loads(args.manifest.read_text())
    manifest_split_hash = split_sha256(manifest)
    resolved_configs, recorded_training_seed = _resolved_configs(
        predictions,
        manifest_split_hash=manifest_split_hash,
        training_seed=args.training_seed,
        expected_models=expected_models,
    )
    bootstrap_seed = (
        args.bootstrap_seed
        if args.bootstrap_seed is not None
        else (args.seed if args.seed is not None else 42)
    )
    expected_by_split = {
        split: pd.DataFrame(manifest[split])[
            [
                "tensor_path", "kernel_family", "architecture_id",
                "topology_id", "labels",
            ]
        ]
        for split in ("test", "exemplar")
    }
    validated = {
        (name, split): _validated_split(path, expected_by_split[split], split)
        for name, path in predictions.items()
        for split in ("test", "exemplar")
    }
    rows = []
    for split_index, split in enumerate(("test", "exemplar")):
        expected = expected_by_split[split]
        frames = {name: validated[(name, split)] for name in predictions}
        errors = {name: _smape(prediction) for name, prediction in frames.items()}
        reference = errors[args.reference]
        families = ["all", *sorted(expected["kernel_family"].unique())]
        for candidate, candidate_errors in errors.items():
            if candidate == args.reference:
                continue
            delta = candidate_errors - reference
            for family_index, family in enumerate(families):
                mask = (
                    np.ones(len(expected), dtype=bool)
                    if family == "all"
                    else expected["kernel_family"].to_numpy(str) == family
                )
                values = delta[mask]
                groups = expected.loc[mask, "architecture_id"].to_numpy(str)
                resample_seed = (
                    bootstrap_seed + 100_003 * split_index + 1009 * family_index
                )
                low, high, fraction_nonnegative = _cluster_bootstrap(
                    values, groups,
                    seed=resample_seed,
                    replicates=args.replicates,
                )
                unique_groups, group_sizes = np.unique(groups, return_counts=True)
                rows.append({
                    "candidate": candidate,
                    "reference": args.reference,
                    "split": split,
                    "kernel_family": family,
                    "n_samples": int(mask.sum()),
                    "n_architecture_groups": len(unique_groups),
                    "singleton_group_fraction": float(np.mean(group_sizes == 1)),
                    "reference_macro_smape": float(reference[mask].mean()),
                    "candidate_macro_smape": float(candidate_errors[mask].mean()),
                    "delta_macro_smape_candidate_minus_reference": float(values.mean()),
                    "cluster_bootstrap_ci95_low": low,
                    "cluster_bootstrap_ci95_high": high,
                    "bootstrap_fraction_delta_nonnegative": fraction_nonnegative,
                    "bootstrap_replicates": args.replicates,
                    "bootstrap_seed": resample_seed,
                    "estimand": "sample_weighted_mean_of_six_target_smape",
                    "resampling_unit": "architecture_id",
                })
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    cohort_frame = pd.DataFrame(rows)
    cohort_frame.to_csv(output / "cohort_cluster_bootstrap.csv", index=False)
    frame = cohort_frame[cohort_frame["split"] == "test"].copy()
    frame.to_csv(output / "architecture_cluster_bootstrap.csv", index=False)

    metric_rows = []
    positions_by_scope = {
        "overall": tuple(range(len(LABEL_KEYS))),
        "resource": (0, 1, 2, 3),
        "timing": (4, 5),
        **{target: (index,) for index, target in enumerate(LABEL_KEYS)},
    }
    for (name, split), prediction_frame in validated.items():
        families_for_split = [
            "all", *sorted(prediction_frame["kernel_family"].unique())
        ]
        matrix = _smape_matrix(prediction_frame)
        for family in families_for_split:
            mask = (
                np.ones(len(prediction_frame), dtype=bool)
                if family == "all"
                else prediction_frame["kernel_family"].to_numpy(str) == family
            )
            for scope, positions in positions_by_scope.items():
                metric_rows.append({
                    "model": name,
                    "split": split,
                    "kernel_family": family,
                    "scope": scope,
                    "n_samples": int(mask.sum()),
                    "smape": float(matrix[mask][:, positions].mean()),
                    "macro_r2": _macro_r2(
                        prediction_frame.loc[mask], positions
                    ),
                })
    pd.DataFrame(metric_rows).to_csv(output / "model_metrics.csv", index=False)

    scope_rows = []
    for split_index, split in enumerate(("test", "exemplar")):
        expected = expected_by_split[split]
        frames = {name: validated[(name, split)] for name in predictions}
        reference_matrix = _smape_matrix(frames[args.reference])
        groups = expected["architecture_id"].to_numpy(str)
        for candidate, candidate_frame in frames.items():
            if candidate == args.reference:
                continue
            candidate_matrix = _smape_matrix(candidate_frame)
            for scope_index, (scope, positions) in enumerate(positions_by_scope.items()):
                delta = (
                    candidate_matrix[:, positions].mean(axis=1)
                    - reference_matrix[:, positions].mean(axis=1)
                )
                resample_seed = (
                    bootstrap_seed + 100_003 * split_index + 1009 * scope_index
                )
                low, high, fraction_nonnegative = _cluster_bootstrap(
                    delta, groups,
                    seed=resample_seed,
                    replicates=args.replicates,
                )
                scope_rows.append({
                    "candidate": candidate,
                    "reference": args.reference,
                    "split": split,
                    "scope": scope,
                    "n_samples": len(delta),
                    "n_architecture_groups": len(np.unique(groups)),
                    "delta_macro_smape_candidate_minus_reference": float(delta.mean()),
                    "cluster_bootstrap_ci95_low": low,
                    "cluster_bootstrap_ci95_high": high,
                    "bootstrap_fraction_delta_nonnegative": fraction_nonnegative,
                    "candidate_win_fraction": float(np.mean(delta < 0)),
                    "bootstrap_replicates": args.replicates,
                    "bootstrap_seed": resample_seed,
                    "resampling_unit": "architecture_id",
                })
    cohort_scope_frame = pd.DataFrame(scope_rows)
    cohort_scope_frame.to_csv(
        output / "cohort_scope_cluster_bootstrap.csv", index=False
    )
    cohort_scope_frame[cohort_scope_frame["split"] == "test"].to_csv(
        output / "scope_cluster_bootstrap.csv", index=False
    )
    provenance = {
        "manifest": str(args.manifest.resolve()),
        "manifest_file_sha256": _sha256(args.manifest),
        "split_sha256": manifest_split_hash,
        "predictions": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for name, path in predictions.items()
        },
        "resolved_configs": resolved_configs,
        "reference": args.reference,
        "replicates": args.replicates,
        "training_seed": recorded_training_seed,
        "bootstrap_seed": bootstrap_seed,
        "interpretation": (
            f"Positive delta means the candidate is worse than {args.reference}. Confidence "
            "intervals resample held-out exact architecture groups and retain all "
            "samples in each selected group. Singleton-heavy families therefore "
            "measure transfer to unseen exact architectures, not within-architecture "
            "replication."
        ),
    }
    (output / "analysis_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
