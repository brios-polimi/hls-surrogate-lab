import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import pandas as pd
import torch

from ll_hls4ml.data.high_level import (
    HighLevelLayerDataset,
    feature_statistics,
)
from ll_hls4ml.io.schema import LABEL_KEYS
from ll_hls4ml.models.registry import build
from ll_hls4ml.training.loaders import make_loader
from scripts.run_e6 import (
    BUDGETS,
    FAMILIES,
    MODELS,
    SEEDS,
    _analyze,
    _find_run,
    _run_dir,
    build_nested_manifests,
)


def row(family: str, architecture: int, design: int) -> dict:
    return {
        "tensor_path": f"{family}/archive_1/a{architecture}-d{design}.pt",
        "kernel_family": family,
        "architecture_id": f"{family}-a{architecture}",
        "topology_id": f"{family}-t{architecture}",
        "archive": "archive_1",
        "architecture_signature_version": "test-architecture-v1",
        "topology_signature_version": "test-topology-v1",
    }


class E6ManifestTests(unittest.TestCase):
    @staticmethod
    def source() -> dict:
        train = [
            row(family, architecture, design)
            for family in sorted(FAMILIES)
            for architecture in range(20)
            for design in range(1 + architecture % 2)
        ]
        return {
            "train": train,
            "validation": [row("2layer", 100, 0)],
            "test": [row("2layer", 101, 0)],
            "exemplar": [row("2layer", 102, 0)],
        }

    def test_subsets_are_deterministic_nested_and_group_safe(self):
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first, _ = build_nested_manifests(
                self.source(), Path(first_dir), seeds=(7,), budgets=BUDGETS
            )
            second, _ = build_nested_manifests(
                self.source(), Path(second_dir), seeds=(7,), budgets=BUDGETS
            )
            prior = set()
            for budget in BUDGETS:
                first_manifest = json.loads(
                    Path(first["7"][str(budget)]["path"]).read_text()
                )
                second_manifest = json.loads(
                    Path(second["7"][str(budget)]["path"]).read_text()
                )
                first_paths = {
                    item["tensor_path"] for item in first_manifest["train"]
                }
                second_paths = {
                    item["tensor_path"] for item in second_manifest["train"]
                }
                self.assertEqual(first_paths, second_paths)
                self.assertTrue(prior <= first_paths)
                prior = first_paths

                selected_arch = {
                    item["architecture_id"] for item in first_manifest["train"]
                }
                all_rows = [
                    item
                    for item in self.source()["train"]
                    if item["architecture_id"] in selected_arch
                ]
                self.assertEqual(
                    first_paths, {item["tensor_path"] for item in all_rows}
                )
                self.assertEqual(first_manifest["test"], self.source()["test"])
            self.assertEqual(prior, {
                item["tensor_path"] for item in self.source()["train"]
            })

    def test_full_data_endpoint_discovery_is_root_relative(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = "e2_h0_structural_seed7"
            run_dir = root / "arbitrary" / "machine" / experiment
            run_dir.mkdir(parents=True)
            (run_dir / "resolved_config.json").write_text("{}")
            self.assertEqual(_find_run([root], experiment), run_dir.resolve())
            self.assertIsNone(_find_run([root], "e2_h0_structural_seed42"))


class E6ModelTests(unittest.TestCase):
    def test_layer_graph_model_has_a_distinct_registry_name(self):
        model = build(
            "high_level_layer_gnn",
            input_dim=33,
            y_means=torch.zeros(6),
            y_stds=torch.ones(6),
            hidden_dim=8,
            num_layers=1,
            heads=1,
            dropout=0.0,
        )
        self.assertEqual(model.encoder.output_dim, 12)

    def test_layer_graph_dataset_and_model_forward_are_aligned(self):
        paths = [f"2layer/archive_1/sample-{index}.pt" for index in range(4)]
        cache = {
            "samples": {
                path: {
                    "features": torch.zeros(2, 18),
                    "target": torch.arange(6, dtype=torch.float32) + index,
                    "kernel_family": "2layer",
                }
                for index, path in enumerate(paths)
            }
        }
        means, stds = feature_statistics(cache, paths)
        dataset = HighLevelLayerDataset(cache, paths, means, stds)
        loader = make_loader(dataset, batch_size=2, shuffle=False, num_workers=0)
        model = build(
            "high_level_layer_gnn",
            input_dim=33,
            y_means=torch.zeros(6),
            y_stds=torch.ones(6),
            hidden_dim=8,
            num_layers=1,
            heads=1,
            dropout=0.0,
        )
        prediction = model(next(iter(loader)))
        self.assertEqual(prediction.shape, (2, 8))


class E6AnalysisTests(unittest.TestCase):
    def test_analysis_packages_all_metrics_and_figures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_manifest = root / "source.json"
            test_rows = [
                {
                    "tensor_path": f"2layer/archive_1/test-{index}.pt",
                    "kernel_family": "2layer",
                    "architecture_id": f"test-architecture-{index}",
                }
                for index in range(2)
            ]
            source_manifest.write_text(json.dumps({"test": test_rows}))
            metadata_path = root / "study_metadata.json"
            metadata = {
                "output_dir": str(root),
                "metadata_path": str(metadata_path),
                "source_manifest": str(source_manifest),
                "source_split_sha256": "source-split",
                "test_membership_sha256": "test-split",
                "bootstrap_replicates": 20,
                "full_data_endpoints": {model: {} for model in MODELS},
                "manifests": {
                    str(seed): {
                        str(budget): {
                            "train_designs": budget * 10 + seed,
                            "train_architectures": budget + 5,
                        }
                        for budget in BUDGETS
                    }
                    for seed in SEEDS
                },
            }
            metadata_path.write_text(json.dumps(metadata))
            for model in MODELS:
                for seed in SEEDS:
                    for budget in BUDGETS:
                        run_dir = _run_dir(metadata, model, seed, budget)
                        run_dir.mkdir(parents=True)
                        (run_dir / "fit_summary.json").write_text(json.dumps({
                            "wall_seconds": 1.0,
                            "best_epoch": 3,
                        }))

            def predictions(_metadata, model, seed, budget):
                offset = {"h0": 2.0, "high_level": 1.0, "extra_trees": 3.0}[model]
                rows = []
                for split in ("validation", "test"):
                    for index, item in enumerate(test_rows):
                        record = {"split": split, **item}
                        for target in LABEL_KEYS:
                            record[f"target_{target}"] = float(index + 10)
                            record[f"prediction_{target}"] = float(index + 10 + offset)
                        rows.append(record)
                return pd.DataFrame(rows)

            with mock.patch(
                "scripts.run_e6._verified_predictions", side_effect=predictions
            ):
                _analyze(metadata)
            analysis = root / "analysis"
            for name in (
                "per_run_metrics.csv",
                "per_architecture_metrics.csv",
                "paired_architecture_deltas.csv",
                "scaling_diagnostics.csv",
                "scaling_curves.png",
                "per_target_scaling_curves.png",
                "REPORT.md",
                "artifact_inventory.csv",
            ):
                self.assertTrue((analysis / name).is_file(), name)
            runs = pd.read_csv(analysis / "per_run_metrics.csv")
            self.assertEqual(len(runs), len(MODELS) * len(SEEDS) * len(BUDGETS))
            paired = pd.read_csv(analysis / "paired_architecture_deltas.csv")
            selected = paired.query(
                "left_model == 'high_level' and right_model == 'h0'"
            )
            self.assertTrue((selected["left_minus_right_smape"] < 0).all())


if __name__ == "__main__":
    unittest.main()
