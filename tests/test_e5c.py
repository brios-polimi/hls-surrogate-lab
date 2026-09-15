"""Focused protocol and numerical tests for E5c."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

import scripts.run_e5c as e5c
from ll_hls4ml.io.schema import LABEL_KEYS
from scripts.run_e5c import (
    BUDGETS,
    DRAW_SEEDS,
    _build_partitions,
    _ridge_solution,
    _sign_flip_p,
    _specs,
    _standardized_cache_rows,
)


class E5cProtocolTests(unittest.TestCase):
    def _old_partitions(self):
        paths = [f"p{index}" for index in range(80)]
        return {
            "partition_seed": 123,
            "architectures": {
                "arch": {
                    "n_total": 80,
                    "query": paths[:10],
                    "validation": paths[10:30],
                    "support_pool": paths[30:],
                    "architecture_summary": {},
                    "topology_id": "t",
                }
            },
        }

    def test_query_is_unchanged_and_total_budgets_are_exact_nested(self):
        old = self._old_partitions()
        new = _build_partitions(old)
        record = new["architectures"]["arch"]
        self.assertEqual(record["query"], old["architectures"]["arch"]["query"])
        self.assertEqual(len(record["adaptation_pool"]), 70)
        for draw in DRAW_SEEDS:
            previous = set()
            for budget in BUDGETS:
                support = record["draws"][str(draw)]["support"][str(budget)]
                self.assertEqual(len(support), budget)
                self.assertTrue(previous <= set(support))
                previous = set(support)

    def test_full_schedule_contains_paired_five_replicate_heads(self):
        specs = _specs(BUDGETS)
        for budget in BUDGETS:
            pretrained = [s for s in specs if s.method == "fresh_head_standardized_pretrained_encoder" and s.budget == budget]
            random = [s for s in specs if s.method == "fresh_head_standardized_random_encoder" and s.budget == budget]
            self.assertEqual([s.replicate for s in pretrained], [0, 1, 2, 3, 4])
            self.assertEqual([s.replicate for s in random], [0, 1, 2, 3, 4])
        graph = {(s.method, s.budget) for s in specs if s.method in {"scratch_full", "pretrained_full_tune"}}
        self.assertEqual(graph, {
            ("scratch_full", 32), ("scratch_full", 64),
            ("pretrained_full_tune", 32), ("pretrained_full_tune", 64),
        })
        random_ridge = [s for s in specs if s.method == "random_encoder_residual_ridge"]
        self.assertEqual(len(random_ridge), len(BUDGETS) * 5)

    def test_feature_standardization_uses_only_reference_paths(self):
        cache = {
            "paths": ["s0", "s1", "q"],
            "features": torch.tensor([[1.0, 2.0], [3.0, 2.0], [5.0, 9.0]]),
            "source_predictions": torch.zeros(3, 6),
        }
        standardized, _, digest = _standardized_cache_rows(cache, ["s0", "s1"], ["s0", "s1"])
        np.testing.assert_allclose(standardized[:, 0].numpy(), [-1.0, 1.0])
        np.testing.assert_allclose(standardized[:, 1].numpy(), [0.0, 0.0])
        self.assertEqual(len(digest), 64)

    def test_dual_ridge_matches_primal_solution(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(7, 12))
        x[:, 0] = 1
        y = rng.normal(size=7)
        prior = rng.normal(size=12)
        alpha = 0.3
        actual = _ridge_solution(x, y, alpha, prior)
        penalty = alpha * np.eye(x.shape[1])
        penalty[0, 0] = alpha * 0.01
        expected = np.linalg.solve(x.T @ x + penalty, x.T @ y + penalty @ prior)
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)

    def test_exact_sign_flip_p_value(self):
        self.assertAlmostEqual(_sign_flip_p(np.ones(7)), 2 / 128)

    def test_analysis_packages_registered_contrasts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            architectures = [f"arch-{index}" for index in range(7)]
            partitions = root / "partitions.json"
            partitions.write_text(json.dumps({
                "architectures": {architecture: {} for architecture in architectures}
            }))
            prediction_rows = []
            for sample in range(3):
                row = {}
                for target in LABEL_KEYS:
                    row[f"target_{target}"] = float(sample + 2)
                    row[f"prediction_{target}"] = float(sample + 2.1)
                prediction_rows.append(row)
            prediction_frame = pd.DataFrame(prediction_rows)
            metrics = e5c._metric_record(prediction_frame)
            zero = pd.DataFrame([
                {
                    "architecture_id": architecture, "draw_seed": -1,
                    "method": "zero_shot", "budget": 0, "replicate": 0,
                    **metrics,
                }
                for architecture in architectures
            ])
            metadata = {
                "output_dir": str(root), "partitions_path": str(partitions),
                "bootstrap_replicates": 20,
            }
            with (
                mock.patch.object(e5c, "BUDGETS", (4,)),
                mock.patch.object(e5c, "PRIMARY_BUDGETS", (4,)),
                mock.patch.object(e5c, "DRAW_SEEDS", (7,)),
                mock.patch.object(e5c, "HEAD_REPLICATES", (0,)),
                mock.patch.object(e5c, "_zero_metrics", return_value=zero),
                mock.patch.object(e5c.pd, "read_csv", return_value=prediction_frame),
            ):
                e5c._analyze(metadata)
            analysis = root / "analysis"
            self.assertTrue((analysis / "primary_curve_contrasts.csv").is_file())
            self.assertTrue((analysis / "primary_contrasts.csv").is_file())
            self.assertTrue((analysis / "secondary_contrasts.csv").is_file())
            self.assertTrue((analysis / "architecture_contrasts.csv").is_file())
            self.assertTrue((analysis / "report.md").is_file())


if __name__ == "__main__":
    unittest.main()
