import unittest

import pandas as pd

from scripts.analyze_e2_hierarchy import _cross_seed_descriptive


class E2HierarchyAnalysisTests(unittest.TestCase):
    def test_cross_seed_summary_counts_interval_directions(self):
        frame = pd.DataFrame(
            {
                "training_seed": [7, 42, 137],
                "candidate": ["control"] * 3,
                "reference": ["h0"] * 3,
                "split": ["test"] * 3,
                "delta_macro_smape_candidate_minus_reference": [1.0, 2.0, -0.5],
                "cluster_bootstrap_ci95_low": [0.2, 1.1, -1.0],
                "cluster_bootstrap_ci95_high": [1.8, 2.9, -0.1],
            }
        )
        summary = _cross_seed_descriptive(
            frame, ["candidate", "reference", "split"]
        ).iloc[0]
        self.assertEqual(summary["seeds"], 3)
        self.assertEqual(summary["intervals_above_zero"], 2)
        self.assertEqual(summary["intervals_below_zero"], 1)
        self.assertAlmostEqual(summary["delta_smape_mean"], 2.5 / 3)


if __name__ == "__main__":
    unittest.main()
