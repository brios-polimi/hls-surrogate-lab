#!/usr/bin/env python3
"""Analyze E7 validation-only screening runs against reused E6 H0 runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from ll_hls4ml.io.schema import LABEL_KEYS
from ll_hls4ml.reporting.accounting import RESOURCE_TARGETS, TIMING_TARGETS
from scripts.run_e7_message_operators import (
    DEFAULT_SCREEN_ROOT,
    SCREEN_SPLIT_SHA256,
    candidate_config,
    candidate_track,
)


SCOPES = {
    "overall": tuple(LABEL_KEYS),
    "resource": tuple(RESOURCE_TARGETS),
    "timing": tuple(TIMING_TARGETS),
}


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _smape(truth: float, prediction: float) -> float:
    return 200 * abs(truth - prediction) / (abs(truth) + abs(prediction) + 1)


def _paired_rows(candidate_path: Path, baseline_path: Path) -> list[dict]:
    candidate_rows = _read_csv(candidate_path)
    candidate_splits = {row["split"] for row in candidate_rows}
    if candidate_splits != {"validation"}:
        raise ValueError(
            f"Screen candidate must contain validation only, got {candidate_splits}: "
            f"{candidate_path}"
        )
    baseline = {
        row["tensor_path"]: row
        for row in _read_csv(baseline_path)
        if row["split"] == "validation"
    }
    if len(baseline) != 2297:
        raise ValueError(f"Expected 2,297 baseline validation rows: {baseline_path}")
    if len(candidate_rows) != len(baseline):
        raise ValueError(
            f"Candidate/baseline validation count mismatch: "
            f"{len(candidate_rows)} != {len(baseline)}"
        )
    output = []
    for candidate in candidate_rows:
        path = candidate["tensor_path"]
        if path not in baseline:
            raise ValueError(f"Baseline lacks validation tensor {path}")
        reference = baseline[path]
        if candidate["architecture_id"] != reference["architecture_id"]:
            raise ValueError(f"Architecture mismatch for {path}")
        row = {
            "tensor_path": path,
            "architecture_id": candidate["architecture_id"],
            "kernel_family": candidate["kernel_family"],
        }
        for target in LABEL_KEYS:
            truth = float(candidate[f"target_{target}"])
            baseline_truth = float(reference[f"target_{target}"])
            if not np.isclose(truth, baseline_truth, rtol=1e-6, atol=1e-6):
                raise ValueError(f"Target mismatch for {path}: {target}")
            candidate_error = _smape(
                truth, float(candidate[f"prediction_{target}"])
            )
            baseline_error = _smape(
                truth, float(reference[f"prediction_{target}"])
            )
            row[f"candidate_{target}"] = candidate_error
            row[f"baseline_{target}"] = baseline_error
            row[f"delta_{target}"] = candidate_error - baseline_error
        output.append(row)
    return output


def _scope_values(rows: list[dict], targets: tuple[str, ...], prefix: str) -> np.ndarray:
    return np.asarray(
        [np.mean([row[f"{prefix}_{target}"] for target in targets]) for row in rows],
        dtype=float,
    )


def _cluster_interval(
    rows: list[dict], targets: tuple[str, ...], replicates: int, seed: int
) -> tuple[float, float]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(row["architecture_id"], []).append(index)
    identifiers = sorted(groups)
    values = _scope_values(rows, targets, "delta")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled = rng.integers(0, len(identifiers), len(identifiers))
        indices = [index for group in sampled for index in groups[identifiers[group]]]
        estimates[replicate] = values[indices].mean()
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def _derived_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _run_paths(results_dir: Path, candidate: str, seed: int) -> tuple[Path, Path]:
    run_dir = (
        results_dir
        / "runs"
        / candidate
        / f"seed{seed}"
        / f"e7_screen_{candidate}_seed{seed}"
    )
    return run_dir / "predictions.csv", run_dir / "resolved_config.json"


def _baseline_paths(screen_root: Path, seed: int) -> tuple[Path, Path]:
    run_dir = (
        screen_root
        / "runs"
        / "h0"
        / f"seed{seed}"
        / f"e6_h0_seed{seed}_p050"
    )
    return run_dir / "predictions.csv", run_dir / "resolved_config.json"


def _validate_config(
    path: Path, candidate: str, seed: int, split_hash: str
) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    config = json.loads(path.read_text())
    candidate_fields = candidate_config(candidate)
    expected = {
        **candidate_fields,
        "seed": seed,
        "split_sha256": split_hash,
        "e7_stage": "screen",
        "evaluation_splits": ["validation"],
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Incompatible candidate config {path}: {mismatches}")
    return config


def _contrast_rows(
    candidate: str,
    seed: int,
    rows: list[dict],
    config: dict,
    baseline_config: dict,
    replicates: int,
) -> list[dict]:
    output = []
    dimensions = [("scope", name, targets) for name, targets in SCOPES.items()]
    dimensions.extend(("target", target, (target,)) for target in LABEL_KEYS)
    families = sorted({row["kernel_family"] for row in rows})
    dimensions.extend(("family", family, tuple(LABEL_KEYS)) for family in families)
    for dimension, level, targets in dimensions:
        selected = (
            rows
            if dimension != "family"
            else [row for row in rows if row["kernel_family"] == level]
        )
        delta = _scope_values(selected, targets, "delta")
        candidate_error = _scope_values(selected, targets, "candidate")
        baseline_error = _scope_values(selected, targets, "baseline")
        low, high = _cluster_interval(
            selected,
            targets,
            replicates,
            _derived_seed(candidate, seed, dimension, level),
        )
        output.append(
            {
                "candidate": candidate,
                "track": candidate_track(candidate),
                "seed": seed,
                "dimension": dimension,
                "level": level,
                "n_designs": len(selected),
                "n_architectures": len(
                    {row["architecture_id"] for row in selected}
                ),
                "baseline_smape": float(baseline_error.mean()),
                "candidate_smape": float(candidate_error.mean()),
                "delta_smape": float(delta.mean()),
                "ci95_low": low,
                "ci95_high": high,
                "candidate_parameters": config["parameter_count"],
                "baseline_parameters": baseline_config["parameter_count"],
                "parameter_matched": (
                    config["parameter_count"] == baseline_config["parameter_count"]
                ),
                "training_contract_matched": True,
                "same_commit": False,
            }
        )
    return output


def _promotion_rows(contrasts: list[dict]) -> list[dict]:
    candidates = sorted({row["candidate"] for row in contrasts})
    output = []
    for candidate in candidates:
        rows = [row for row in contrasts if row["candidate"] == candidate]
        overall = sorted(
            (
                row
                for row in rows
                if row["dimension"] == "scope" and row["level"] == "overall"
            ),
            key=lambda row: int(row["seed"]),
        )
        scope = {
            level: [
                row["delta_smape"]
                for row in rows
                if row["dimension"] == "scope" and row["level"] == level
            ]
            for level in ("resource", "timing")
        }
        deltas = [row["delta_smape"] for row in overall]
        ordinary = all(delta < 0 for delta in deltas) and np.mean(deltas) <= -0.3
        strong_one = min(deltas) <= -1.0 and max(deltas) <= 0.3
        guardrail_failure = any(
            len(values) >= 2 and sum(value > 1.0 for value in values) >= 2
            for values in scope.values()
        )
        promote = (ordinary or strong_one) and not guardrail_failure
        output.append(
            {
                "candidate": candidate,
                "track": candidate_track(candidate),
                "seeds": ";".join(str(row["seed"]) for row in overall),
                "overall_deltas": ";".join(
                    f"{row['delta_smape']:.4f}" for row in overall
                ),
                "mean_delta_smape": float(np.mean(deltas)),
                "ordinary_rule": ordinary,
                "strong_one_rule": strong_one,
                "scope_guardrail_failure": guardrail_failure,
                "promotion_recommendation": promote,
                "parameter_count": overall[0]["candidate_parameters"],
                "parameter_matched": overall[0]["parameter_matched"],
            }
        )
    return sorted(output, key=lambda row: row["mean_delta_smape"])


def _write_report(path: Path, promotions: list[dict]) -> None:
    lines = [
        "# E7 validation-only screen",
        "",
        "Negative deltas favour the candidate. These are promotion diagnostics, "
        "not test-set results.",
        "",
        "| candidate | track | seed deltas | mean delta | parameters | rule recommendation |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in promotions:
        lines.append(
            f"| `{row['candidate']}` | {row['track']} | {row['overall_deltas']} | "
            f"{row['mean_delta_smape']:.3f} | {row['parameter_count']} | "
            f"{'advance' if row['promotion_recommendation'] else 'do not advance'} |"
        )
    lines.extend(
        [
            "",
            "All fits use the same hidden width and matched E2 training contract. "
            "They are not parameter- or wall-clock-matched unless explicitly shown. "
            "The H0 controls are reused E6 fits from an earlier code revision.",
            "",
            "Freeze at most five candidates before any confirmatory test evaluation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--screen-root", type=Path, default=DEFAULT_SCREEN_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--candidates", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42])
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    args = parser.parse_args()
    args.results_dir = args.results_dir.resolve()
    args.screen_root = args.screen_root.resolve()
    args.output_dir = (args.output_dir or args.results_dir / "analysis").resolve()
    metadata_path = args.results_dir / "study_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("stage") != "screen":
        raise ValueError("E7 screen analysis requires stage=screen metadata")
    candidates = args.candidates or metadata["candidates"]

    contrasts = []
    for candidate in candidates:
        for seed in args.seeds:
            if seed not in SCREEN_SPLIT_SHA256:
                raise ValueError(f"No registered E6 50% split for seed {seed}")
            prediction_path, config_path = _run_paths(
                args.results_dir, candidate, seed
            )
            baseline_prediction, baseline_config_path = _baseline_paths(
                args.screen_root, seed
            )
            if not prediction_path.is_file():
                raise FileNotFoundError(prediction_path)
            config = _validate_config(
                config_path, candidate, seed, SCREEN_SPLIT_SHA256[seed]
            )
            baseline_config = json.loads(baseline_config_path.read_text())
            if baseline_config.get("split_sha256") != SCREEN_SPLIT_SHA256[seed]:
                raise ValueError(f"Baseline split mismatch: {baseline_config_path}")
            paired = _paired_rows(prediction_path, baseline_prediction)
            contrasts.extend(
                _contrast_rows(
                    candidate,
                    seed,
                    paired,
                    config,
                    baseline_config,
                    args.bootstrap_replicates,
                )
            )

    promotions = _promotion_rows(contrasts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "validation_contrasts.csv", contrasts)
    _write_csv(args.output_dir / "promotion_summary.csv", promotions)
    _write_report(args.output_dir / "REPORT.md", promotions)
    print(f"Wrote E7 screen analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
