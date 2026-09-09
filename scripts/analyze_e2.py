#!/usr/bin/env python3
"""Validate E2 predictions and compute architecture-cluster bootstrap contrasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from ll_hls4ml.io.schema import LABEL_KEYS


def _prediction_arg(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("Use NAME=/path/to/predictions.csv")
    return name, Path(path)


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
    parser.add_argument("--reference", default="h0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replicates", type=int, default=10_000)
    args = parser.parse_args()

    predictions = dict(args.prediction)
    if len(predictions) != len(args.prediction):
        raise ValueError("Prediction names must be unique")
    if args.reference not in predictions:
        raise ValueError(f"Reference {args.reference!r} was not supplied")
    manifest = json.loads(args.manifest.read_text())
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
    expected = expected_by_split["test"]
    frames = {name: validated[(name, "test")] for name in predictions}
    errors = {name: _smape(frame) for name, frame in frames.items()}
    reference = errors[args.reference]
    rows = []
    families = ["all", *sorted(expected["kernel_family"].unique())]
    for candidate, candidate_errors in errors.items():
        if candidate == args.reference:
            continue
        delta = candidate_errors - reference
        for family_index, family in enumerate(families):
            mask = np.ones(len(expected), dtype=bool) if family == "all" else (
                expected["kernel_family"].to_numpy(str) == family
            )
            values = delta[mask]
            groups = expected.loc[mask, "architecture_id"].to_numpy(str)
            low, high, fraction_nonnegative = _cluster_bootstrap(
                values, groups,
                seed=args.seed + 1009 * family_index,
                replicates=args.replicates,
            )
            unique_groups, group_sizes = np.unique(groups, return_counts=True)
            rows.append({
                "candidate": candidate,
                "reference": args.reference,
                "split": "test",
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
                "bootstrap_seed": args.seed + 1009 * family_index,
                "estimand": "sample_weighted_mean_of_six_target_smape",
                "resampling_unit": "architecture_id",
            })
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
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
            low, high, fraction_nonnegative = _cluster_bootstrap(
                delta, groups,
                seed=args.seed + 1009 * scope_index,
                replicates=args.replicates,
            )
            scope_rows.append({
                "candidate": candidate,
                "reference": args.reference,
                "split": "test",
                "scope": scope,
                "n_samples": len(delta),
                "n_architecture_groups": len(np.unique(groups)),
                "delta_macro_smape_candidate_minus_reference": float(delta.mean()),
                "cluster_bootstrap_ci95_low": low,
                "cluster_bootstrap_ci95_high": high,
                "bootstrap_fraction_delta_nonnegative": fraction_nonnegative,
                "candidate_win_fraction": float(np.mean(delta < 0)),
                "bootstrap_replicates": args.replicates,
                "resampling_unit": "architecture_id",
            })
    pd.DataFrame(scope_rows).to_csv(
        output / "scope_cluster_bootstrap.csv", index=False
    )
    provenance = {
        "manifest": str(args.manifest.resolve()),
        "predictions": {
            name: str(path.resolve()) for name, path in predictions.items()
        },
        "reference": args.reference,
        "replicates": args.replicates,
        "seed": args.seed,
        "interpretation": (
            "Positive delta means the candidate is worse than H0. Confidence "
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
