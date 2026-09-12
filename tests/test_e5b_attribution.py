import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_e5b import (
    PRIMARY_BUDGETS,
    PRIMARY_DRAW_SEEDS,
    _analysis_dir,
    _evaluation_complete,
    _fit_complete,
    _run_dir,
    _run_seeds,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class E5bProtocolTests(unittest.TestCase):
    def test_primary_protocol_requires_seven_runs_not_forty_two(self):
        self.assertEqual(PRIMARY_BUDGETS, (32,))
        self.assertEqual(PRIMARY_DRAW_SEEDS, (42,))
        self.assertEqual(7 * len(PRIMARY_BUDGETS) * len(PRIMARY_DRAW_SEEDS), 7)

    def test_k16_and_k32_share_initialization_but_not_training_seed(self):
        init16, train16 = _run_seeds("architecture-a", 42, 16)
        init32, train32 = _run_seeds("architecture-a", 42, 32)
        self.assertEqual(init16, init32)
        self.assertNotEqual(train16, train32)

    def test_expanded_analysis_cannot_overwrite_primary_analysis(self):
        metadata = {"output_dir": "/tmp/e5b-test"}
        primary = _analysis_dir(metadata, (42,), (32,))
        expanded = _analysis_dir(metadata, (7, 42, 137), (16, 32))
        self.assertEqual(primary.name, "analysis")
        self.assertNotEqual(primary, expanded)


class E5bCompletionTests(unittest.TestCase):
    def test_completion_markers_are_content_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            metadata = {"output_dir": temporary}
            run_dir = _run_dir(metadata, "architecture-a", 42, 32)
            checkpoint = (
                run_dir / "checkpoints"
                / "e5b_scratch_architecture-a_draw42_k32_checkpoint.pt"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            (run_dir / "fit_summary.json").write_text(json.dumps({
                "checkpoint_sha256": sha256(checkpoint),
            }))
            self.assertTrue(_fit_complete(metadata, "architecture-a", 42, (32,)))
            checkpoint.write_bytes(b"corrupt")
            self.assertFalse(_fit_complete(metadata, "architecture-a", 42, (32,)))

            predictions = run_dir / "predictions.csv"
            predictions.write_text("prediction\n1\n")
            (run_dir / "evaluation_summary.json").write_text(json.dumps({
                "predictions_sha256": sha256(predictions),
            }))
            self.assertTrue(
                _evaluation_complete(metadata, "architecture-a", 42, (32,))
            )
            predictions.write_text("prediction\n2\n")
            self.assertFalse(
                _evaluation_complete(metadata, "architecture-a", 42, (32,))
            )


if __name__ == "__main__":
    unittest.main()
