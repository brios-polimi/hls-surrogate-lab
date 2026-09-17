import unittest

from scripts.analyze_e7_screen import _promotion_rows
from scripts.run_e7_message_operators import SCREEN_CANDIDATES, candidate_track


def _row(candidate, seed, level, delta):
    return {
        "candidate": candidate,
        "seed": seed,
        "dimension": "scope",
        "level": level,
        "delta_smape": delta,
        "candidate_parameters": 250000,
        "parameter_matched": False,
    }


class PromotionRuleTests(unittest.TestCase):
    def test_candidate_catalog_is_explicit_and_unique(self):
        self.assertEqual(len(SCREEN_CANDIDATES), 23)
        self.assertEqual(len(SCREEN_CANDIDATES), len(set(SCREEN_CANDIDATES)))
        self.assertEqual(candidate_track("pna"), "mechanism")
        self.assertEqual(candidate_track("pna_wide"), "performance")
        self.assertEqual(candidate_track("attn_dual"), "attention")
        self.assertEqual(candidate_track("variable_route_all"), "variable")

    def test_consistent_small_improvement_advances(self):
        rows = [
            _row("receiver_gate", 7, "overall", -0.4),
            _row("receiver_gate", 42, "overall", -0.3),
            _row("receiver_gate", 7, "resource", -0.2),
            _row("receiver_gate", 42, "resource", -0.1),
            _row("receiver_gate", 7, "timing", -0.8),
            _row("receiver_gate", 42, "timing", -0.7),
        ]
        result = _promotion_rows(rows)[0]
        self.assertTrue(result["ordinary_rule"])
        self.assertTrue(result["promotion_recommendation"])

    def test_repeated_scope_regression_blocks_promotion(self):
        rows = [
            _row("bidirectional", 7, "overall", -1.2),
            _row("bidirectional", 42, "overall", -0.1),
            _row("bidirectional", 7, "resource", 1.2),
            _row("bidirectional", 42, "resource", 1.1),
            _row("bidirectional", 7, "timing", -5.0),
            _row("bidirectional", 42, "timing", -2.5),
        ]
        result = _promotion_rows(rows)[0]
        self.assertTrue(result["scope_guardrail_failure"])
        self.assertFalse(result["promotion_recommendation"])

    def test_one_unstable_win_does_not_advance(self):
        rows = [
            _row("relation_mixer", 7, "overall", -1.5),
            _row("relation_mixer", 42, "overall", 0.5),
            _row("relation_mixer", 7, "resource", -1.0),
            _row("relation_mixer", 42, "resource", 0.2),
            _row("relation_mixer", 7, "timing", -2.5),
            _row("relation_mixer", 42, "timing", 0.8),
        ]
        result = _promotion_rows(rows)[0]
        self.assertFalse(result["ordinary_rule"])
        self.assertFalse(result["strong_one_rule"])
        self.assertFalse(result["promotion_recommendation"])


if __name__ == "__main__":
    unittest.main()
