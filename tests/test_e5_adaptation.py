import unittest
from unittest import mock
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ll_hls4ml.io.schema import LABEL_KEYS
from ll_hls4ml.models.fusion import HierarchicalHighLevelFusion
from scripts.run_e5 import (
    _apply_affine,
    _analyze,
    _markdown_table,
    _run_dir,
    audit_partitions,
    build_partitions,
    fit_affine_coefficients,
)


class E5PartitionTests(unittest.TestCase):
    @staticmethod
    def rows():
        return [
            {
                "tensor_path": f"arch-{architecture}/point-{point}.pt",
                "architecture_id": f"arch-{architecture}",
                "architecture_summary": f"architecture {architecture}",
                "topology_id": f"topology-{architecture}",
            }
            for architecture in range(7)
            for point in range(100)
        ]

    def test_fixed_query_validation_and_nested_support(self):
        arguments = (self.rows(), (0, 4, 16, 32), (7, 42, 137), 20260910, 0.4, 0.2)
        first = build_partitions(*arguments)
        second = build_partitions(*arguments)
        self.assertEqual(first, second)
        audit_partitions(first)

        for record in first["architectures"].values():
            self.assertEqual(len(record["query"]), 40)
            self.assertEqual(len(record["validation"]), 20)
            self.assertEqual(len(record["support_pool"]), 40)
            for draw in record["draws"].values():
                self.assertEqual(
                    draw["support"]["32"], draw["support"]["16"] + draw["support_order"][16:32]
                )


class E5AffineTests(unittest.TestCase):
    def test_rank_deficient_support_is_safe(self):
        prediction = np.ones((4, len(LABEL_KEYS))) * 3
        target = np.ones_like(prediction) * 5
        coefficients = fit_affine_coefficients(
            prediction, target, prediction, target, [0.0, 0.01, 1.0]
        )
        self.assertEqual(len(coefficients), len(LABEL_KEYS))
        self.assertTrue(all(np.isfinite(row["slope"]) for row in coefficients))

    def test_recovers_exact_log_affine_mapping(self):
        source = np.arange(1, 9, dtype=float)[:, None]
        source = np.repeat(source, len(LABEL_KEYS), axis=1)
        target = np.expm1(0.3 + 1.2 * np.log1p(source))
        coefficients = fit_affine_coefficients(
            source[:4], target[:4], source[4:], target[4:], [0.0]
        )
        calibrated = _apply_affine(
            torch.tensor(source, dtype=torch.float32),
            {"coefficients": coefficients},
        ).numpy()
        np.testing.assert_allclose(calibrated, target, rtol=1e-5, atol=1e-5)


class FusionEncodingTests(unittest.TestCase):
    def test_encode_is_the_exact_pre_head_forward_path(self):
        model = HierarchicalHighLevelFusion(
            instruction_vocab_size=5,
            edge_pos_vocab_size=3,
            high_level_input_dim=9,
            y_means=torch.zeros(6),
            y_stds=torch.ones(6),
            hidden_dim=8,
            num_layers=1,
            dropout=0.0,
        )
        cdfg = torch.randn(2, model.cdfg_encoder.output_dim)
        high_level = torch.randn(2, model.high_level_encoder.output_dim)

        class Store:
            x = torch.zeros(2, 9)
            batch = torch.tensor([0, 1])
            edge_index = torch.empty((2, 0), dtype=torch.long)

        class Example:
            num_graphs = 2
            high_level_strategy = torch.zeros(2, dtype=torch.long)
            high_level_io_type = torch.zeros(2, dtype=torch.long)

            def __getitem__(self, key):
                return Store()

        with mock.patch.object(model.cdfg_encoder, "encode", return_value=cdfg), mock.patch.object(
            model.high_level_encoder, "forward", return_value=high_level
        ):
            encoded = model.encode(Example())
        self.assertTrue(torch.equal(encoded, torch.cat([cdfg, high_level], dim=-1)))

        with mock.patch.object(model, "encode", return_value=encoded):
            self.assertTrue(torch.equal(model(Example()), model.classifier(encoded)))


class E5ReportingTests(unittest.TestCase):
    def test_markdown_does_not_require_optional_tabulate(self):
        table = _markdown_table(pd.DataFrame({"method": ["head"], "score": [1.25]}))
        self.assertIn("| method | score |", table)
        self.assertIn("| head | 1.25 |", table)

    def test_analysis_packages_raw_runs_without_optional_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            partitions_path = root / "partitions.json"
            metadata_path = root / "study_metadata.json"
            architectures = ("arch-a", "arch-b")
            draws = (7, 42, 137)
            partitions_path.write_text(json.dumps({
                "architectures": {architecture: {} for architecture in architectures},
                "draw_seeds": list(draws),
            }))
            metadata = {
                "output_dir": str(root),
                "partitions_path": str(partitions_path),
                "metadata_path": str(metadata_path),
                "bootstrap_replicates": 100,
            }
            metadata_path.write_text(json.dumps(metadata))

            def rows(split, adapted_for, method, draw, offset):
                result = []
                for sample in range(3):
                    row = {
                        "split": split,
                        "adapted_for_architecture": adapted_for,
                        "architecture_id": adapted_for,
                        "draw_seed": draw,
                        "method": method,
                        "budget": 0 if method == "zero_shot" else 4,
                    }
                    for target in LABEL_KEYS:
                        row[f"target_{target}"] = float(sample + 2)
                        row[f"prediction_{target}"] = float(sample + 2 + offset)
                    result.append(row)
                return result

            zero_rows = []
            for architecture in architectures:
                zero_rows.extend(rows("query", architecture, "zero_shot", None, 1.0))
            zero_rows.extend(rows("source_test", "all", "zero_shot", None, 1.0))
            zero_dir = root / "runs/zero_shot"
            zero_dir.mkdir(parents=True)
            pd.DataFrame(zero_rows).to_csv(zero_dir / "predictions.csv", index=False)

            for architecture in architectures:
                for draw in draws:
                    run_dir = _run_dir(metadata, architecture, draw, "affine", 4)
                    run_dir.mkdir(parents=True)
                    prediction_rows = rows("query", architecture, "affine", draw, 0.5)
                    prediction_rows.extend(
                        rows("source_test", architecture, "affine", draw, 1.25)
                    )
                    pd.DataFrame(prediction_rows).to_csv(
                        run_dir / "predictions.csv", index=False
                    )
                    (run_dir / "fit_summary.json").write_text(json.dumps({
                        "wall_seconds": 1.0,
                        "trainable_parameters": 12,
                    }))

            _analyze(metadata, ("affine",), (0, 4))
            analysis = root / "analysis"
            self.assertTrue((analysis / "REPORT.md").is_file())
            self.assertTrue((analysis / "artifact_inventory.csv").is_file())
            summary = pd.read_csv(analysis / "equal_architecture_summary.csv")
            self.assertTrue(set(LABEL_KEYS) <= set(summary["scope"]))


if __name__ == "__main__":
    unittest.main()
