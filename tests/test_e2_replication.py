import json
import tempfile
import unittest
from pathlib import Path

import torch

from ll_hls4ml.data.signatures import (
    HIGH_LEVEL_ARCHITECTURE_SIGNATURE_VERSION,
    HIGH_LEVEL_TOPOLOGY_SIGNATURE_VERSION,
)
from ll_hls4ml.models.hierarchical import BlockFlowLayer, InstructionFlowLayer
from ll_hls4ml.models.registry import build


class NoLocalMessageTests(unittest.TestCase):
    def test_control_preserves_parameter_scaffold(self):
        common = {
            "instruction_vocab_size": 5,
            "edge_pos_vocab_size": 3,
            "y_means": torch.zeros(6),
            "y_stds": torch.ones(6),
            "hidden_dim": 8,
            "num_layers": 2,
            "dropout": 0.0,
            "use_global_features": True,
            "use_context": True,
            "split_heads": True,
            "hurdle_heads": True,
        }
        intact = build("hierarchical", **common)
        control = build("hierarchical_no_local_message", **common)

        self.assertTrue(intact.use_local_messages)
        self.assertFalse(control.use_local_messages)
        self.assertEqual(intact.state_dict().keys(), control.state_dict().keys())
        self.assertEqual(
            sum(parameter.numel() for parameter in intact.parameters()),
            sum(parameter.numel() for parameter in control.parameters()),
        )

    def test_instruction_layer_bypasses_only_message_values(self):
        torch.manual_seed(3)
        layer = InstructionFlowLayer(hidden_dim=4, dropout=0.0)
        state = torch.randn(3, 4)
        first_edges = torch.tensor([[0, 1, 2], [1, 2, 0]])
        second_edges = torch.tensor([[0, 1, 2], [2, 0, 1]])
        first_features = torch.randn(3, 4)
        second_features = torch.randn(3, 4)

        first = layer(
            state, first_edges, second_edges, first_features,
            use_messages=False,
        )
        second = layer(
            state, second_edges, first_edges, second_features,
            use_messages=False,
        )
        self.assertTrue(torch.equal(first, second))
        first.sum().backward()
        self.assertIsNotNone(layer.control.weight.grad)
        self.assertIsNotNone(layer.def_use.weight.grad)
        self.assertEqual(torch.count_nonzero(layer.control.weight.grad), 0)
        self.assertEqual(torch.count_nonzero(layer.def_use.weight.grad), 0)

    def test_block_layer_bypasses_cfg_messages(self):
        torch.manual_seed(5)
        layer = BlockFlowLayer(hidden_dim=4, dropout=0.0)
        state = torch.randn(3, 4)
        first_edges = torch.tensor([[0, 1, 2], [1, 2, 0]])
        second_edges = torch.tensor([[0, 1, 2], [2, 0, 1]])

        first = layer(state, first_edges, use_messages=False)
        second = layer(state, second_edges, use_messages=False)
        self.assertTrue(torch.equal(first, second))
        first.sum().backward()
        self.assertIsNotNone(layer.message.weight.grad)
        self.assertEqual(torch.count_nonzero(layer.message.weight.grad), 0)


class E2ManifestValidationTests(unittest.TestCase):
    def test_rejects_architecture_leakage(self):
        from scripts.run_e2 import FAMILIES, _validate_manifest

        def row(family, architecture_id, suffix):
            return {
                "tensor_path": f"{suffix}.pt",
                "kernel_family": family,
                "architecture_id": architecture_id,
                "architecture_signature_version": (
                    HIGH_LEVEL_ARCHITECTURE_SIGNATURE_VERSION
                ),
                "topology_signature_version": HIGH_LEVEL_TOPOLOGY_SIGNATURE_VERSION,
            }

        manifest = {
            split: [
                row(family, f"{split}-{family}", f"{split}-{index}")
                for index, family in enumerate(sorted(FAMILIES))
            ]
            for split in ("train", "validation", "test")
        }
        manifest["exemplar"] = [row("exemplar", "exemplar-0", "exemplar-0")]
        manifest["test"][0]["architecture_id"] = manifest["train"][0][
            "architecture_id"
        ]

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "architecture IDs overlap"):
                _validate_manifest(path, expected_hash=None)


if __name__ == "__main__":
    unittest.main()
