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


def _smape(frame: pd.DataFrame) -> np.ndarray:
    values = []
    for target in LABEL_KEYS:
        truth = frame[f"target_{target}"].to_numpy(float)
        prediction = frame[f"prediction_{target}"].to_numpy(float)
        values.append(200 * np.abs(truth - prediction) / (
            np.abs(truth) + np.abs(prediction) + 1.0
        ))
    return np.asarray(values).mean(axis=0)


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


def _validated_test(path: Path, expected: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["split"] == "test"].copy()
    if frame["tensor_path"].duplicated().any():
        raise ValueError(f"Duplicate test prediction paths in {path}")
    actual = set(frame["tensor_path"])
    wanted = set(expected["tensor_path"])
    if actual != wanted:
        raise ValueError(
            f"Test membership mismatch in {path}: "
            f"missing={len(wanted - actual)}, extra={len(actual - wanted)}"
        )
    frame = expected.merge(frame, on="tensor_path", validate="one_to_one")
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
    expected = pd.DataFrame(manifest["test"])[
        ["tensor_path", "kernel_family", "architecture_id", "topology_id", "labels"]
    ]
    frames = {
        name: _validated_test(path, expected)
        for name, path in predictions.items()
    }
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
