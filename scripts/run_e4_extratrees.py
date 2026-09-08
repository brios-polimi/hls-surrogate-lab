#!/usr/bin/env python3
"""Run the E4 ExtraTrees control across leave-one-family-out folds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from hierarchical_cpu_baseline import (
    HurdleRegressor,
    arrays,
    choose_mode,
    feature_frame,
    feature_sets,
    git_state,
    metric_rows,
)
from ll_hls4ml.data.dataset import HeteroGraphDataset
from ll_hls4ml.data.protocols import SPLIT_NAMES
from ll_hls4ml.data.signatures import HIGH_LEVEL_ARCHITECTURE_SIGNATURE_VERSION
from ll_hls4ml.data.vocab import load_vocab
from ll_hls4ml.io.schema import LABEL_KEYS
from ll_hls4ml.reporting.accounting import split_sha256
from run_e4 import DEFAULT_FAMILY_ORDER, FAMILIES, STUDY_ID, _validate_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--tensor-dir", type=Path, required=True)
    parser.add_argument("--vocab", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument(
        "--families", nargs="+", choices=FAMILIES,
        default=list(DEFAULT_FAMILY_ORDER),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.manifest_dir = args.manifest_dir.resolve()
    args.tensor_dir = args.tensor_dir.resolve()
    args.vocab = (args.vocab or args.tensor_dir / "vocab.json").resolve()
    manifests = {}
    universe = None
    for family in args.families:
        path = args.manifest_dir / f"leave_{family}_out.json"
        _validate_manifest(path, family)
        manifest = json.loads(path.read_text())
        paths = {
            row["tensor_path"] for split in SPLIT_NAMES for row in manifest[split]
        }
        if universe is not None and paths != universe:
            raise ValueError(f"{family}: fold has a different sample universe")
        universe = paths
        manifests[family] = (path, manifest)

    vocabulary, _, _ = load_vocab(args.vocab)
    dataset = HeteroGraphDataset(
        args.tensor_dir,
        types=list(FAMILIES),
        silent=False,
        relative_paths=sorted(universe),
    )
    frame = feature_frame(
        dataset, universe, len(vocabulary), args.feature_cache.resolve()
    )
    indexed = frame.set_index("tensor_path")
    columns = feature_sets(frame)["core_context"]

    for family, (manifest_path, manifest) in manifests.items():
        experiment = f"e4_extra_trees_leave_{family}_out_seed{args.seed}"
        output = (args.output_dir / f"seed{args.seed}" / experiment).resolve()
        summary_path = output / "summary.csv"
        if summary_path.exists():
            print(f"ExtraTrees already complete: {summary_path}")
            continue
        output.mkdir(parents=True, exist_ok=True)
        paths = {
            split: [row["tensor_path"] for row in manifest[split]]
            for split in SPLIT_NAMES
        }
        manifest_rows = {
            split: {row["tensor_path"]: row for row in manifest[split]}
            for split in SPLIT_NAMES
        }
        train_x, train_y = arrays(indexed, paths["train"], columns)
        validation_x, validation_y = arrays(
            indexed, paths["validation"], columns
        )
        started = time.perf_counter()
        model = HurdleRegressor("extra_trees", args.seed).fit(train_x, train_y)
        fit_seconds = time.perf_counter() - started
        mode = choose_mode(model, validation_x, validation_y)

        metrics = []
        predictions = []
        for split in ("validation", "test", "exemplar"):
            x, target = arrays(indexed, paths[split], columns)
            prediction = model.predict_modes(x)[mode]
            split_families = indexed.loc[
                paths[split], "kernel_family"
            ].to_numpy(str)
            metrics.extend(metric_rows(
                experiment, "extra_trees", "core_context", len(paths["train"]),
                split, split_families, prediction, target,
            ))
            for row_index, tensor_path in enumerate(paths[split]):
                source = manifest_rows[split][tensor_path]
                row = {
                    "experiment": experiment,
                    "model": "extra_trees",
                    "feature_set": "core_context",
                    "selected_hurdle_mode": mode,
                    "split": split,
                    "tensor_path": tensor_path,
                    "kernel_family": split_families[row_index],
                    "architecture_id": source["architecture_id"],
                    "topology_id": source["topology_id"],
                }
                for target_index, label in enumerate(LABEL_KEYS):
                    row[f"target_{label}"] = float(target[row_index, target_index])
                    row[f"prediction_{label}"] = float(
                        prediction[row_index, target_index]
                    )
                predictions.append(row)

        metric_frame = pd.DataFrame(metrics)
        metric_frame.to_csv(output / "metrics.csv", index=False)
        pd.DataFrame(predictions).to_csv(output / "predictions.csv", index=False)
        summary = (
            metric_frame[metric_frame.kernel_family == "all"]
            .groupby(
                ["experiment", "model", "feature_set", "train_size", "split"],
                as_index=False,
            )
            .agg(macro_smape=("smape", "mean"), macro_r2=("r2", "mean"))
        )
        summary.to_csv(summary_path, index=False)
        resolved = {
            "study_id": STUDY_ID,
            "protocol_id": f"leave_{family}_out_exact_v1",
            "held_out_family": family,
            "experiment_name": experiment,
            "model": "extra_trees",
            "seed": args.seed,
            "split_sha256": split_sha256(manifest),
            "manifest_sha256": _sha256(manifest_path),
            "architecture_signature_version": (
                HIGH_LEVEL_ARCHITECTURE_SIGNATURE_VERSION
            ),
            "sizes": {split: len(values) for split, values in paths.items()},
            "feature_set": "core_context",
            "feature_count": len(columns),
            "feature_interpretation": (
                "deterministic node/relation/opcode/context summaries; no adjacency"
            ),
            "fit_seconds": fit_seconds,
            "selected_hurdle_mode": mode,
            "extra_trees": {
                "n_estimators": 300,
                "min_samples_leaf": 2,
                "max_features": 0.8,
            },
            "git": git_state(),
        }
        (output / "resolved_config.json").write_text(
            json.dumps(resolved, indent=2) + "\n"
        )
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
