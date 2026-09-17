#!/usr/bin/env python3
"""Analyze the pre-specified E2 hierarchy ablation suite by training seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_e2 import HIERARCHY_ABLATIONS, NEURAL_MODELS


def _seed_path(value: str) -> tuple[int, Path]:
    seed, separator, path = value.partition("=")
    if not separator or not seed or not path:
        raise argparse.ArgumentTypeError("Use SEED=/path/to/predictions.csv")
    try:
        return int(seed), Path(path)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid seed in {value!r}") from error


def _prediction_path(results_dir: Path, control: str, seed: int) -> Path:
    experiment = NEURAL_MODELS[control][1].format(seed=seed)
    return results_dir / f"seed{seed}" / experiment / "predictions.csv"


def _cross_seed_descriptive(
    frame: pd.DataFrame, group_columns: list[str],
) -> pd.DataFrame:
    return (
        frame.groupby(group_columns, as_index=False)
        .agg(
            seeds=("training_seed", "count"),
            delta_smape_mean=("delta_macro_smape_candidate_minus_reference", "mean"),
            delta_smape_std=("delta_macro_smape_candidate_minus_reference", "std"),
            delta_smape_min=("delta_macro_smape_candidate_minus_reference", "min"),
            delta_smape_max=("delta_macro_smape_candidate_minus_reference", "max"),
            intervals_above_zero=(
                "cluster_bootstrap_ci95_low",
                lambda values: int((values > 0).sum()),
            ),
            intervals_below_zero=(
                "cluster_bootstrap_ci95_high",
                lambda values: int((values < 0).sum()),
            ),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--h0-prediction", action="append", type=_seed_path, required=True,
        help="Repeat as SEED=/path/to/same-seed LLVM-Hier predictions.csv",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 137])
    parser.add_argument(
        "--controls", nargs="+", choices=HIERARCHY_ABLATIONS,
        default=list(HIERARCHY_ABLATIONS),
    )
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--replicates", type=int, default=10_000)
    args = parser.parse_args()

    if len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must not contain duplicates")
    if len(args.controls) != len(set(args.controls)):
        parser.error("--controls must not contain duplicates")
    h0_predictions = dict(args.h0_prediction)
    if len(h0_predictions) != len(args.h0_prediction):
        parser.error("--h0-prediction seeds must be unique")
    missing_h0 = set(args.seeds) - set(h0_predictions)
    extra_h0 = set(h0_predictions) - set(args.seeds)
    if missing_h0 or extra_h0:
        parser.error(
            "H0 prediction seeds must exactly match --seeds: "
            f"missing={sorted(missing_h0)}, extra={sorted(extra_h0)}"
        )

    manifest = args.manifest.resolve()
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)

    family_rows = []
    scope_rows = []
    metric_rows = []
    training_rows = []
    commands = []
    parameter_counts = {}
    analyzer = _REPO_ROOT / "scripts" / "analyze_e2.py"
    for seed in args.seeds:
        predictions = {"h0": h0_predictions[seed].resolve()}
        predictions.update(
            {
                control: _prediction_path(results_dir, control, seed).resolve()
                for control in args.controls
            }
        )
        missing = [str(path) for path in predictions.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing prediction files: {missing}")
        seed_output = output_dir / f"seed{seed}"
        command = [
            sys.executable,
            str(analyzer),
            "--manifest", str(manifest),
            "--reference", "h0",
            "--output-dir", str(seed_output),
            "--training-seed", str(seed),
            "--bootstrap-seed", str(args.bootstrap_seed),
            "--replicates", str(args.replicates),
            "--prediction", f"h0={predictions['h0']}",
            "--expected-model", "h0=hierarchical",
        ]
        for control in args.controls:
            command.extend(
                [
                    "--prediction", f"{control}={predictions[control]}",
                    "--expected-model", f"{control}={NEURAL_MODELS[control][0]}",
                ]
            )
        subprocess.run(command, cwd=_REPO_ROOT, check=True)
        commands.append(command)
        seed_provenance = json.loads(
            (seed_output / "analysis_provenance.json").read_text()
        )
        seed_counts = {
            name: details["parameter_count"]
            for name, details in seed_provenance["resolved_configs"].items()
        }
        if any(count is None for count in seed_counts.values()):
            raise ValueError(
                f"Seed {seed} lacks a recorded parameter count: {seed_counts}"
            )
        if len(set(seed_counts.values())) != 1:
            raise ValueError(
                f"Seed {seed} parameter-count mismatch against H0: {seed_counts}"
            )
        parameter_counts[str(seed)] = seed_counts
        family_frame = pd.read_csv(seed_output / "cohort_cluster_bootstrap.csv")
        family_frame.insert(0, "training_seed", seed)
        family_rows.append(family_frame)
        scope_frame = pd.read_csv(
            seed_output / "cohort_scope_cluster_bootstrap.csv"
        )
        scope_frame.insert(0, "training_seed", seed)
        scope_rows.append(scope_frame)
        metric_frame = pd.read_csv(seed_output / "model_metrics.csv")
        metric_frame.insert(0, "training_seed", seed)
        metric_rows.append(metric_frame)
        for control in args.controls:
            run_dir = _prediction_path(results_dir, control, seed).parent
            summary = json.loads((run_dir / "summary.json").read_text())
            config = summary["resolved_config"]
            curve = pd.read_csv(run_dir / "learning_curves.csv")
            training_rows.append({
                "training_seed": seed,
                "control": control,
                "model": config["model"],
                "parameter_count": config["parameter_count"],
                "epochs_ran": int(curve["epoch"].max()) + 1,
                "best_epoch": int(summary["best_epoch"]),
                "best_validation_smape": float(summary["best_metric"]),
                "stop_reason": config.get("stop_reason"),
                "training_hours": float(config["cumulative_training_seconds"]) / 3600,
                "wall_hours": float(config["wall_seconds"]) / 3600,
                "peak_gpu_memory_mb": float(config["peak_gpu_memory_mb"]),
            })

    family_frame = pd.concat(family_rows, ignore_index=True)
    family_frame.to_csv(
        output_dir / "seed_specific_family_contrasts.csv", index=False
    )
    combined = family_frame[
        (family_frame["split"] == "test")
        & (family_frame["kernel_family"] == "all")
    ].copy()
    combined.to_csv(output_dir / "seed_specific_hierarchy_contrasts.csv", index=False)
    scope_frame = pd.concat(scope_rows, ignore_index=True)
    scope_frame.to_csv(
        output_dir / "seed_specific_scope_contrasts.csv", index=False
    )
    pd.concat(metric_rows, ignore_index=True).to_csv(
        output_dir / "seed_specific_model_metrics.csv", index=False
    )
    pd.DataFrame(training_rows).to_csv(
        output_dir / "training_diagnostics.csv", index=False
    )
    descriptive = _cross_seed_descriptive(
        combined, ["candidate", "reference", "split"]
    )
    score_means = (
        combined.groupby(["candidate", "reference", "split"], as_index=False)
        .agg(
            reference_smape_mean=("reference_macro_smape", "mean"),
            candidate_smape_mean=("candidate_macro_smape", "mean"),
        )
    )
    descriptive = descriptive.merge(
        score_means, on=["candidate", "reference", "split"],
        validate="one_to_one",
    )
    descriptive.to_csv(output_dir / "cross_seed_descriptive.csv", index=False)
    _cross_seed_descriptive(
        family_frame,
        ["candidate", "reference", "split", "kernel_family"],
    ).to_csv(output_dir / "cross_seed_family_descriptive.csv", index=False)
    _cross_seed_descriptive(
        scope_frame,
        ["candidate", "reference", "split", "scope"],
    ).to_csv(output_dir / "cross_seed_scope_descriptive.csv", index=False)
    provenance = {
        "study": "e2_hierarchy_ablation_v1",
        "manifest": str(manifest),
        "results_dir": str(results_dir),
        "seeds": args.seeds,
        "primary_control": "hierarchy_orderless",
        "secondary_controls": ["no_block_cfg", "no_callee"],
        "requested_controls": args.controls,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_replicates": args.replicates,
        "h0_predictions": {
            str(seed): str(path.resolve())
            for seed, path in h0_predictions.items()
        },
        "parameter_counts": parameter_counts,
        "commands": [command[1:] for command in commands],
        "cross_seed_interpretation": (
            "Cross-seed mean, standard deviation, minimum, and maximum are "
            "descriptive. Confidence intervals are seed-specific architecture-"
            "cluster intervals and do not estimate training-seed uncertainty."
        ),
    }
    (output_dir / "analysis_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(combined.to_string(index=False))
    print("\nCross-seed descriptive summary:")
    print(descriptive.to_string(index=False))


if __name__ == "__main__":
    main()
