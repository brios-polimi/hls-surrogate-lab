#!/usr/bin/env python3
"""Validate and analyze paired E4 leave-one-family-out result bundles."""

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


SCOPES = {
    "overall": tuple(range(len(LABEL_KEYS))),
    "resource": (0, 1, 2, 3),
    "timing": (4, 5),
    **{target: (index,) for index, target in enumerate(LABEL_KEYS)},
}


def _smape_matrix(frame: pd.DataFrame) -> np.ndarray:
    columns = []
    for target in LABEL_KEYS:
        truth = frame[f"target_{target}"].to_numpy(float)
        prediction = frame[f"prediction_{target}"].to_numpy(float)
        columns.append(
            200.0 * np.abs(truth - prediction)
            / (np.abs(truth) + np.abs(prediction) + 1.0)
        )
    return np.asarray(columns).T


def _macro_r2(frame: pd.DataFrame, positions: tuple[int, ...]) -> float:
    values = []
    for index in positions:
        target = LABEL_KEYS[index]
        truth = frame[f"target_{target}"].to_numpy(float)
        prediction = frame[f"prediction_{target}"].to_numpy(float)
        denominator = np.square(truth - truth.mean()).sum()
        scale = max(float(np.square(truth).sum()), 1.0)
        if denominator <= np.finfo(float).eps * scale:
            values.append(np.nan)
        else:
            values.append(
                1.0 - np.square(truth - prediction).sum() / denominator
            )
    values = np.asarray(values, dtype=float)
    return float("nan") if np.isnan(values).all() else float(np.nanmean(values))


def _cluster_estimates(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    rng: np.random.Generator,
    replicates: int,
) -> np.ndarray:
    unique, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    sizes = np.bincount(inverse)
    estimates = np.empty(replicates, dtype=float)
    for start in range(0, replicates, 250):
        count = min(250, replicates - start)
        sampled = rng.integers(0, len(unique), size=(count, len(unique)))
        estimates[start:start + count] = (
            sums[sampled].sum(axis=1) / sizes[sampled].sum(axis=1)
        )
    return estimates


def _interval(estimates: np.ndarray) -> tuple[float, float, float]:
    low, high = np.percentile(estimates, (2.5, 97.5))
    return float(low), float(high), float(np.mean(estimates >= 0))


def _body_rows(manifest: dict, split: str) -> pd.DataFrame:
    return pd.DataFrame(manifest[split])[
        ["tensor_path", "kernel_family", "architecture_id", "labels"]
    ]


def _validate_prediction(
    path: Path,
    expected: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[frame["split"] == split].copy()
    if frame["tensor_path"].duplicated().any():
        raise ValueError(f"Duplicate {split} prediction paths in {path}")
    wanted = set(expected["tensor_path"])
    actual = set(frame["tensor_path"])
    if wanted != actual:
        raise ValueError(
            f"{split} membership mismatch in {path}: "
            f"missing={len(wanted - actual)}, extra={len(actual - wanted)}"
        )
    canonical = expected.set_index("tensor_path").loc[sorted(wanted)]
    actual_frame = frame.set_index("tensor_path").loc[sorted(wanted)]
    for column in ("kernel_family", "architecture_id"):
        if column in actual_frame and not np.array_equal(
            canonical[column].astype(str), actual_frame[column].astype(str)
        ):
            raise ValueError(f"Manifest field {column!r} differs in {path}")
    for index, target in enumerate(LABEL_KEYS):
        truth = actual_frame[f"target_{target}"].to_numpy(float)
        manifest_truth = canonical["labels"].map(
            lambda values: values[index]
        ).to_numpy(float)
        if not np.allclose(truth, manifest_truth, rtol=1e-6, atol=1e-6):
            raise ValueError(f"Target mismatch for {target} in {path}")
    return actual_frame.reset_index().merge(
        canonical.reset_index(),
        on="tensor_path",
        how="left",
        suffixes=("", "_manifest"),
        validate="one_to_one",
    )


def _family_from_config(config: dict) -> str:
    family = config.get("held_out_family")
    if not family:
        raise ValueError("Missing held_out_family in resolved config")
    return str(family)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h0-root", type=Path, required=True)
    parser.add_argument("--extra-trees-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--e2-metrics", type=Path)
    args = parser.parse_args()

    h0_seed = args.h0_root / f"seed{args.seed}"
    extra_seed = args.extra_trees_root / f"seed{args.seed}"
    h0_runs = sorted(h0_seed.glob(f"e4_h0_leave_*_out_seed{args.seed}"))
    if len(h0_runs) != 7:
        raise ValueError(f"Expected seven H0 folds, found {len(h0_runs)}")

    study_index_path = args.h0_root / "study_index.json"
    study_index = json.loads(study_index_path.read_text())
    completed = {
        (str(row["family"]), int(row["seed"]))
        for row in study_index
        if row.get("status") == "COMPLETE" and row.get("returncode") == 0
    }

    validated: dict[tuple[str, str, str], pd.DataFrame] = {}
    configs: dict[tuple[str, str], dict] = {}
    manifests: dict[str, dict] = {}
    fold_audit = []
    accounting_rows = []

    for h0_run in h0_runs:
        h0_config = json.loads((h0_run / "resolved_config.json").read_text())
        family = _family_from_config(h0_config)
        if (family, args.seed) not in completed:
            raise ValueError(f"H0 fold {family} is not COMPLETE in study index")
        extra_run = (
            extra_seed / f"e4_extra_trees_leave_{family}_out_seed{args.seed}"
        )
        required_extra = {
            "predictions.csv", "metrics.csv", "summary.csv", "resolved_config.json"
        }
        missing_extra = [
            name for name in sorted(required_extra) if not (extra_run / name).is_file()
        ]
        if missing_extra:
            raise ValueError(f"ExtraTrees fold {family} missing {missing_extra}")
        extra_config = json.loads((extra_run / "resolved_config.json").read_text())
        if _family_from_config(extra_config) != family:
            raise ValueError(f"ExtraTrees held-out family mismatch for {family}")
        if h0_config["split_sha256"] != extra_config["split_sha256"]:
            raise ValueError(f"Split hash mismatch for {family}")
        versions = {
            h0_config.get("architecture_signature_version"),
            extra_config.get("architecture_signature_version"),
        }
        if versions != {"wa_hls4ml_ordered_layer_structure_v1"}:
            raise ValueError(f"Unexpected signature version for {family}: {versions}")

        manifest = json.loads((h0_run / "split_manifest.json").read_text())
        manifests[family] = manifest
        split_families = {
            split: set(_body_rows(manifest, split)["kernel_family"])
            for split in ("train", "validation", "test", "exemplar")
        }
        if split_families["test"] != {family}:
            raise ValueError(f"Test membership is not exactly {family}")
        if family in split_families["train"] or family in split_families["validation"]:
            raise ValueError(f"Held-out family {family} leaked into train/validation")
        train_ids = set(_body_rows(manifest, "train")["architecture_id"])
        validation_ids = set(_body_rows(manifest, "validation")["architecture_id"])
        if train_ids & validation_ids:
            raise ValueError(f"Train/validation architecture leakage for {family}")

        for split in ("test", "exemplar"):
            expected = _body_rows(manifest, split)
            validated[(family, "h0", split)] = _validate_prediction(
                h0_run / "predictions.csv", expected, split
            )
            validated[(family, "extra_trees", split)] = _validate_prediction(
                extra_run / "predictions.csv", expected, split
            )
            left = validated[(family, "h0", split)].sort_values("tensor_path")
            right = validated[(family, "extra_trees", split)].sort_values(
                "tensor_path"
            )
            for target in LABEL_KEYS:
                if not np.allclose(
                    left[f"target_{target}"], right[f"target_{target}"],
                    rtol=1e-6, atol=1e-6,
                ):
                    raise ValueError(
                        f"H0/ExtraTrees target mismatch for {family}/{split}/{target}"
                    )
        # H0's standard bundle emits test/exemplar predictions only.  ExtraTrees
        # also emits validation rows, which can still be checked against the same
        # saved manifest without treating their absence from H0 as incompleteness.
        validated[(family, "extra_trees", "validation")] = _validate_prediction(
            extra_run / "predictions.csv", _body_rows(manifest, "validation"),
            "validation",
        )

        configs[(family, "h0")] = h0_config
        configs[(family, "extra_trees")] = extra_config
        test = _body_rows(manifest, "test")
        groups, counts = np.unique(test["architecture_id"], return_counts=True)
        fold_audit.append({
            "held_out_family": family,
            "split_sha256": h0_config["split_sha256"],
            "train_samples": len(manifest["train"]),
            "validation_samples": len(manifest["validation"]),
            "test_samples": len(manifest["test"]),
            "exemplar_samples": len(manifest["exemplar"]),
            "test_architecture_groups": len(groups),
            "test_singleton_group_fraction": float(np.mean(counts == 1)),
            "matched_prediction_membership": True,
            "matched_targets": True,
        })
        accounting_rows.append({
            "held_out_family": family,
            "model": "h0",
            "best_epoch": int(json.loads((h0_run / "summary.json").read_text())["best_epoch"]),
            "best_validation_smape": float(
                json.loads((h0_run / "summary.json").read_text())["best_metric"]
            ),
            "wall_hours": float(h0_config["wall_seconds"]) / 3600.0,
            "hours_to_best": float(h0_config["best_wall_seconds"]) / 3600.0,
            "peak_gpu_memory_mb": float(h0_config["peak_gpu_memory_mb"]),
            "fit_seconds": np.nan,
            "git_commit": h0_config["ll_hls4ml_git"]["commit"],
            "git_dirty": bool(h0_config["ll_hls4ml_git"]["dirty"]),
        })
        accounting_rows.append({
            "held_out_family": family,
            "model": "extra_trees",
            "best_epoch": np.nan,
            "best_validation_smape": float(
                pd.read_csv(extra_run / "summary.csv").query(
                    "split == 'validation'"
                )["macro_smape"].iloc[0]
            ),
            "wall_hours": np.nan,
            "hours_to_best": np.nan,
            "peak_gpu_memory_mb": np.nan,
            "fit_seconds": float(extra_config["fit_seconds"]),
            "git_commit": extra_config["git"]["commit"],
            "git_dirty": bool(extra_config["git"]["dirty"]),
        })

    families = sorted(manifests)
    if len(families) != 7:
        raise ValueError(f"Expected seven unique held-out families, got {families}")

    metric_rows = []
    bias_rows = []
    label_shift_rows = []
    matrices: dict[tuple[str, str], np.ndarray] = {}
    for family in families:
        train_labels = np.asarray(manifests[family]["train"], dtype=object)
        train_values = np.asarray(
            [row["labels"] for row in train_labels], dtype=float
        )
        test_values = np.asarray(
            [row["labels"] for row in manifests[family]["test"]], dtype=float
        )
        for target_index, target in enumerate(LABEL_KEYS):
            train_target = train_values[:, target_index]
            test_target = test_values[:, target_index]
            train_log = np.log1p(train_target)
            test_log = np.log1p(test_target)
            spread = float(train_log.std(ddof=0))
            low, high = np.quantile(train_target, (0.01, 0.99))
            train_median = float(np.median(train_target))
            test_median = float(np.median(test_target))
            label_shift_rows.append({
                "held_out_family": family,
                "target": target,
                "train_median": train_median,
                "test_median": test_median,
                "test_to_train_median_ratio": (
                    (test_median + 1.0) / (train_median + 1.0)
                ),
                "log_mean_shift_in_train_sd": (
                    float((test_log.mean() - train_log.mean()) / spread)
                    if spread > 0 else np.nan
                ),
                "test_fraction_below_train_p01": float(np.mean(test_target < low)),
                "test_fraction_above_train_p99": float(np.mean(test_target > high)),
                "train_nonzero_fraction": float(np.mean(train_target > 0)),
                "test_nonzero_fraction": float(np.mean(test_target > 0)),
            })
        for model in ("h0", "extra_trees"):
            for split in ("test", "exemplar"):
                frame = validated[(family, model, split)]
                matrix = _smape_matrix(frame)
                if split == "test":
                    matrices[(family, model)] = matrix
                for scope, positions in SCOPES.items():
                    metric_rows.append({
                        "held_out_family": family,
                        "model": model,
                        "split": split,
                        "scope": scope,
                        "n_samples": len(frame),
                        "n_architecture_groups": frame["architecture_id"].nunique(),
                        "smape": float(matrix[:, positions].mean()),
                        "macro_r2": _macro_r2(frame, positions),
                    })
            frame = validated[(family, model, "test")]
            for target in LABEL_KEYS:
                truth = frame[f"target_{target}"].to_numpy(float)
                prediction = frame[f"prediction_{target}"].to_numpy(float)
                log_ratio = np.log((prediction + 1.0) / (truth + 1.0))
                bias_rows.append({
                    "held_out_family": family,
                    "model": model,
                    "target": target,
                    "median_log_ratio": float(np.median(log_ratio)),
                    "geometric_median_prediction_to_target_ratio": float(
                        np.exp(np.median(log_ratio))
                    ),
                    "fraction_underpredicted": float(np.mean(prediction < truth)),
                })

    comparison_rows = []
    bootstrap_by_scope: dict[str, list[np.ndarray]] = {
        scope: [] for scope in SCOPES
    }
    for family_index, family in enumerate(families):
        h0 = validated[(family, "h0", "test")]
        extra = validated[(family, "extra_trees", "test")]
        groups = h0["architecture_id"].to_numpy(str)
        for scope_index, (scope, positions) in enumerate(SCOPES.items()):
            delta = (
                matrices[(family, "extra_trees")][:, positions].mean(axis=1)
                - matrices[(family, "h0")][:, positions].mean(axis=1)
            )
            estimates = _cluster_estimates(
                delta,
                groups,
                rng=np.random.default_rng(
                    args.seed + 1009 * family_index + 7919 * scope_index
                ),
                replicates=args.replicates,
            )
            bootstrap_by_scope[scope].append(estimates)
            low, high, fraction_nonnegative = _interval(estimates)
            comparison_rows.append({
                "held_out_family": family,
                "scope": scope,
                "n_samples": len(delta),
                "n_architecture_groups": len(np.unique(groups)),
                "h0_smape": float(
                    matrices[(family, "h0")][:, positions].mean()
                ),
                "extra_trees_smape": float(
                    matrices[(family, "extra_trees")][:, positions].mean()
                ),
                "delta_extra_trees_minus_h0": float(delta.mean()),
                "cluster_bootstrap_ci95_low": low,
                "cluster_bootstrap_ci95_high": high,
                "bootstrap_fraction_delta_nonnegative": fraction_nonnegative,
                "h0_sample_win_fraction": float(np.mean(delta > 0)),
            })

    equal_family_rows = []
    for scope, estimates_by_family in bootstrap_by_scope.items():
        positions = SCOPES[scope]
        h0_family = np.asarray([
            matrices[(family, "h0")][:, positions].mean()
            for family in families
        ])
        extra_family = np.asarray([
            matrices[(family, "extra_trees")][:, positions].mean()
            for family in families
        ])
        estimates = np.asarray(estimates_by_family).mean(axis=0)
        low, high, fraction_nonnegative = _interval(estimates)
        equal_family_rows.append({
            "scope": scope,
            "n_families": len(families),
            "h0_equal_family_smape": float(h0_family.mean()),
            "extra_trees_equal_family_smape": float(extra_family.mean()),
            "delta_extra_trees_minus_h0": float(
                (extra_family - h0_family).mean()
            ),
            "fixed_family_stratified_cluster_ci95_low": low,
            "fixed_family_stratified_cluster_ci95_high": high,
            "bootstrap_fraction_delta_nonnegative": fraction_nonnegative,
            "families_h0_wins": int(np.sum(h0_family < extra_family)),
            "estimand": "equal_weight_mean_across_seven_fixed_families",
        })

    shift_rows = []
    if args.e2_metrics:
        e2 = pd.read_csv(args.e2_metrics)
        e4 = pd.DataFrame(metric_rows)
        for model in ("h0", "extra_trees"):
            for family in families:
                for scope in ("overall", "resource", "timing"):
                    e2_match = e2.query(
                        "model == @model and split == 'test' and "
                        "kernel_family == @family and scope == @scope"
                    )
                    e4_match = e4.query(
                        "model == @model and held_out_family == @family and "
                        "split == 'test' and scope == @scope"
                    )
                    if len(e2_match) != 1 or len(e4_match) != 1:
                        raise ValueError(
                            f"Missing unique E2/E4 metric for {model}/{family}/{scope}"
                        )
                    old = float(e2_match["smape"].iloc[0])
                    new = float(e4_match["smape"].iloc[0])
                    shift_rows.append({
                        "model": model,
                        "kernel_family": family,
                        "scope": scope,
                        "e2_unseen_signature_smape": old,
                        "e4_unseen_family_smape": new,
                        "e4_minus_e2_smape": new - old,
                        "e4_to_e2_ratio": new / old,
                        "interpretation": "descriptive; test memberships differ",
                    })

    exemplar_ablation_rows = []
    if args.e2_metrics:
        e2 = pd.read_csv(args.e2_metrics)
        e4 = pd.DataFrame(metric_rows)
        for model in ("h0", "extra_trees"):
            for family in families:
                for scope in ("overall", "resource", "timing"):
                    reference = e2.query(
                        "model == @model and split == 'exemplar' and "
                        "kernel_family == 'all' and scope == @scope"
                    )
                    held_out = e4.query(
                        "model == @model and held_out_family == @family and "
                        "split == 'exemplar' and scope == @scope"
                    )
                    if len(reference) != 1 or len(held_out) != 1:
                        raise ValueError(
                            f"Missing exemplar metric for {model}/{family}/{scope}"
                        )
                    full = float(reference["smape"].iloc[0])
                    without = float(held_out["smape"].iloc[0])
                    exemplar_ablation_rows.append({
                        "model": model,
                        "source_family_omitted": family,
                        "scope": scope,
                        "e2_full_source_exemplar_smape": full,
                        "e4_fold_exemplar_smape": without,
                        "fold_minus_e2_smape": without - full,
                        "interpretation": (
                            "descriptive source-mixture diagnostic; protocols differ"
                        ),
                    })

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_audit).to_csv(output / "fold_audit.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(
        output / "paired_architecture_bootstrap.csv", index=False
    )
    pd.DataFrame(equal_family_rows).to_csv(
        output / "equal_family_summary.csv", index=False
    )
    pd.DataFrame(bias_rows).to_csv(output / "prediction_bias.csv", index=False)
    pd.DataFrame(label_shift_rows).to_csv(
        output / "label_support_shift.csv", index=False
    )
    pd.DataFrame(accounting_rows).to_csv(output / "fold_accounting.csv", index=False)
    if shift_rows:
        pd.DataFrame(shift_rows).to_csv(output / "e2_e4_shift.csv", index=False)
    if exemplar_ablation_rows:
        pd.DataFrame(exemplar_ablation_rows).to_csv(
            output / "exemplar_source_ablation.csv", index=False
        )

    provenance = {
        "h0_root": str(args.h0_root.resolve()),
        "extra_trees_root": str(args.extra_trees_root.resolve()),
        "study_index": str(study_index_path.resolve()),
        "seed": args.seed,
        "bootstrap_replicates": args.replicates,
        "held_out_families": families,
        "validation": {
            "seven_h0_folds_complete": True,
            "seven_extra_trees_folds_present": True,
            "matched_prediction_membership_and_targets": True,
            "held_out_family_absent_from_train_and_validation": True,
            "test_contains_only_held_out_family": True,
            "train_validation_architecture_ids_disjoint": True,
            "signature_version": "wa_hls4ml_ordered_layer_structure_v1",
        },
        "interpretation": (
            "Per-family intervals resample exact architecture IDs within the "
            "held-out family. Equal-family intervals independently resample "
            "architecture IDs within each of the seven fixed benchmark families; "
            "they do not make the families a random sample of all HLS domains. "
            "All neural comparisons remain conditional on training seed 42."
        ),
    }
    (output / "analysis_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(pd.DataFrame(equal_family_rows).to_string(index=False))
    print()
    print(
        pd.DataFrame(comparison_rows).query("scope == 'overall'").to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
