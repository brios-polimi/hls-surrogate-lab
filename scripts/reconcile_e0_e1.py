#!/usr/bin/env python3
"""Verify that an out-of-order E1 result is continuous with the frozen E0 release."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from ll_hls4ml.io.schema import LABEL_KEYS
from ll_hls4ml.reporting.accounting import split_sha256


def _paths(manifest: dict, split: str) -> set[str]:
    return {row["tensor_path"] for row in manifest[split]}


def _targets_match(
    release_manifest: dict,
    predictions: list[dict[str, str]],
) -> bool:
    expected = {
        (split, row["tensor_path"]): row["labels"]
        for split in ("test", "exemplar")
        for row in release_manifest[split]
    }
    for prediction in predictions:
        labels = expected.get((prediction["split"], prediction["tensor_path"]))
        if labels is None:
            return False
        observed = [float(prediction[f"target_{key}"]) for key in LABEL_KEYS]
        if not all(
            math.isclose(actual, wanted, rel_tol=0.0, abs_tol=1e-6)
            for actual, wanted in zip(observed, labels, strict=True)
        ):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e0-release-dir", type=Path, required=True)
    parser.add_argument("--e1-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite reconciliation: {args.output}")

    release = json.loads((args.e0_release_dir / "release.json").read_text())
    e0 = json.loads((args.e0_release_dir / "official.json").read_text())
    e1 = json.loads((args.e1_run_dir / "split_manifest.json").read_text())
    summary = json.loads((args.e1_run_dir / "summary.json").read_text())
    resolved = json.loads((args.e1_run_dir / "resolved_config.json").read_text())
    accounting = json.loads((args.e1_run_dir / "experiment_accounting.json").read_text())
    predictions = list(csv.DictReader((args.e1_run_dir / "predictions.csv").open()))

    membership = {
        split: {
            "e0": len(e0[split]),
            "e1": len(e1[split]),
            "exact_set_match": _paths(e0, split) == _paths(e1, split),
        }
        for split in ("train", "validation", "test", "exemplar")
    }
    expected_predictions = {
        (split, row["tensor_path"])
        for split in ("test", "exemplar") for row in e1[split]
    }
    actual_predictions = {
        (row["split"], row["tensor_path"]) for row in predictions
    }
    checks = {
        "exact_membership": all(row["exact_set_match"] for row in membership.values()),
        "split_hash_matches_output": split_sha256(e1) == accounting["split_sha256"],
        "prediction_membership": actual_predictions == expected_predictions,
        "prediction_rows_unique": len(actual_predictions) == len(predictions),
        "prediction_targets_match_release": _targets_match(e0, predictions),
        "six_target_contract": release["label_keys"] == LABEL_KEYS,
        "code_commit_matches_release_producer": (
            resolved.get("ll_hls4ml_git", {}).get("commit")
            == release.get("producer_commit")
        ),
        "archive_count": resolved.get("archive_count") == 31,
        "protocol": resolved.get("protocol_id") == "official_in_distribution",
        "model": resolved.get("model") == "hierarchical",
        "seed": resolved.get("seed") == 42,
        "normal_completion": resolved.get("evaluation_checkpoint_path") is None,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"E0/E1 continuity failed: {failed}")
    report = {
        "status": "passed",
        "e0_release_id": release["release_id"],
        "e0_tensor_index_sha256": release["tensor_index_sha256"],
        "e0_vocabulary_sha256": release["vocabulary_sha256"],
        "e1_tensor_source_revision": resolved.get("tensor_source_revision"),
        "e1_split_sha256": accounting["split_sha256"],
        "membership": membership,
        "checks": checks,
        "training": {
            "precision": resolved.get("precision"),
            "world_size": resolved.get("distributed_world_size"),
            "effective_global_batch": resolved.get("effective_global_batch"),
            "best_epoch": summary.get("best_epoch"),
            "best_validation_smape": summary.get("best_metric"),
            "stop_reason": accounting.get("stop_reason"),
            "resumed_from_epoch": accounting.get("resumed_from_epoch"),
            "cumulative_training_seconds": accounting.get(
                "cumulative_training_seconds"
            ),
            "best_wall_seconds": accounting.get("best_wall_seconds"),
            "peak_gpu_memory_mb": accounting.get("peak_gpu_memory_mb"),
        },
        "known_gap": (
            "E1 records the immutable tensor repository revision but not the vocab "
            "file hash; checkpoint reuse must retain that revision's vocab."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
