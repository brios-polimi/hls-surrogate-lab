#!/usr/bin/env python3
"""Run the four E0 neural contracts on tiny frozen-manifest membership."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


MODELS = (
    "hierarchical",
    "hierarchical_high_level_fusion",
    "pooled_control",
    "hierarchical_topology_destroyed",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    release_dir = args.release_dir.resolve()
    tensor_root = args.tensor_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite E0 smoke output: {output}")
    output.mkdir(parents=True)

    official = json.loads((release_dir / "official.json").read_text())
    sizes = {"train": 4, "validation": 2, "test": 2, "exemplar": 2}
    manifest = {name: official[name][:count] for name, count in sizes.items()}
    manifest_path = output / "smoke_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    common = {
        "tensor_dir": str(tensor_root),
        "vocab_path": str(tensor_root / "vocab.json"),
        "split_manifest_path": str(manifest_path),
        "require_complete_split_manifest": True,
        "high_level_cache": str(release_dir / "high_level_cache.pt"),
        "results_dir": str(output),
        "study_id": "e0_smoke_v1",
        "protocol_id": "official_tiny_smoke",
        "seed": 42,
        "batch_size": 2,
        "num_workers": 0,
        "precision": "float32",
        "epochs": 1,
        "patience": 0,
        "checkpoint_interval": 1,
        "hidden_dim": 8,
        "num_layers": 1,
        "dropout": 0.0,
        "use_global_features": True,
        "use_context": True,
        "split_heads": True,
        "hurdle_heads": False,
        "loss": "log_huber",
        "verbose": 0,
    }
    train_script = Path(__file__).with_name("train.py")
    required_prediction_columns = {
        "split", "kernel_family", "tensor_path",
        "target_lut", "prediction_lut", "target_interval_max",
        "prediction_interval_max",
    }
    report = {}
    for model in MODELS:
        experiment = f"e0_smoke_{model}"
        config = dict(common, model=model, experiment_name=experiment)
        config["checkpoint_dir"] = str(output / experiment / "checkpoints")
        if model == "pooled_control":
            config.update({"pool": "multi", "node_aggr": "concat"})
        if model == "hierarchical_topology_destroyed":
            config["corruption_seed"] = 42
        config_path = output / f"{experiment}.json"
        config_path.write_text(json.dumps(config, indent=2))
        subprocess.run(
            [sys.executable, str(train_script), "--config", str(config_path)],
            check=True, cwd=train_script.parents[1],
        )
        run_dir = output / experiment
        required_files = {
            "summary.json", "resolved_config.json", "split_manifest.json",
            "predictions.csv", "metrics.csv", "learning_curves.csv",
        }
        missing = sorted(name for name in required_files if not (run_dir / name).is_file())
        if missing:
            raise RuntimeError(f"{model} result bundle misses {missing}")
        with (run_dir / "predictions.csv").open(newline="") as handle:
            predictions = list(csv.DictReader(handle))
        columns = set(predictions[0]) if predictions else set()
        if not required_prediction_columns <= columns:
            raise RuntimeError(f"{model} prediction columns are incomplete: {columns}")
        expected_paths = {
            row["tensor_path"] for split in ("test", "exemplar")
            for row in manifest[split]
        }
        actual_paths = {row["tensor_path"] for row in predictions}
        if actual_paths != expected_paths:
            raise RuntimeError(f"{model} prediction membership mismatch")
        report[model] = {"predictions": len(predictions), "status": "passed"}
    (output / "e0_smoke_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
