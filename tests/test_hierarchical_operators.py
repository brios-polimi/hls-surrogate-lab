import unittest

import torch

from ll_hls4ml.models.hierarchical import BlockFlowLayer, InstructionFlowLayer
from ll_hls4ml.models.hierarchical_operators import (
    OPERATOR_PROFILES,
    OperatorBlockLayer,
    OperatorInstructionLayer,
    operator_spec,
)
from ll_hls4ml.models.registry import build


class OperatorLayerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.state = torch.randn(5, 8)
        self.control = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        self.def_use = torch.tensor([[0, 0, 2], [2, 3, 4]])
        self.edge_features = torch.randn(4, 8)

    def test_generic_forward_mean_matches_canonical_instruction_layer(self):
        canonical = InstructionFlowLayer(8, 0.0).eval()
        generic = OperatorInstructionLayer(
            8, 0.0, operator_spec("forward_mean")
        ).eval()
        with torch.no_grad():
            generic.control_forward.source.weight.copy_(canonical.control.weight)
            generic.def_use_forward.source.weight.copy_(canonical.def_use.weight)
            generic.update.norm.load_state_dict(canonical.norm.state_dict())
        expected = canonical(
            self.state,
            self.control,
            self.def_use,
            self.edge_features,
        )
        actual = generic(
            self.state,
            self.control,
            self.def_use,
            self.edge_features,
        )
        self.assertTrue(torch.allclose(expected, actual, atol=1e-7))

    def test_generic_forward_mean_matches_canonical_block_layer(self):
        canonical = BlockFlowLayer(8, 0.0).eval()
        generic = OperatorBlockLayer(
            8, 0.0, operator_spec("forward_mean")
        ).eval()
        with torch.no_grad():
            generic.forward_flow.source.weight.copy_(canonical.message.weight)
            generic.update.norm.load_state_dict(canonical.norm.state_dict())
        expected = canonical(self.state, self.control)
        actual = generic(self.state, self.control)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-7))

    def test_every_profile_has_finite_gradients_with_empty_relations(self):
        empty = torch.empty((2, 0), dtype=torch.long)
        empty_features = torch.empty((0, 8))
        for name, spec in OPERATOR_PROFILES.items():
            with self.subTest(profile=name):
                state = self.state.detach().clone().requires_grad_(True)
                instruction = OperatorInstructionLayer(8, 0.0, spec)
                block = OperatorBlockLayer(8, 0.0, spec)
                output = instruction(
                    state, empty, empty, empty_features
                ) + block(state, empty)
                self.assertTrue(torch.isfinite(output).all())
                output.sum().backward()
                missing = [
                    key
                    for key, parameter in [
                        *instruction.named_parameters(),
                        *block.named_parameters(),
                    ]
                    if parameter.grad is None
                ]
                self.assertEqual(missing, [])

    def test_bidirectional_profile_responds_to_edge_reversal(self):
        layer = OperatorBlockLayer(
            8, 0.0, operator_spec("bidirectional")
        ).eval()
        forward = layer(self.state, self.control)
        reversed_edges = layer(self.state, self.control.flip(0))
        self.assertFalse(torch.allclose(forward, reversed_edges))


class OperatorModelTests(unittest.TestCase):
    def test_registry_build_replaces_only_local_layer_classes(self):
        common = {
            "instruction_vocab_size": 8,
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
        canonical = build("hierarchical", **common)
        candidate = build(
            "hierarchical_operator",
            operator_profile="gate_bidir",
            **common,
        )
        self.assertEqual(candidate.operator_profile, "gate_bidir")
        self.assertIsInstance(
            candidate.instruction_layers[0], OperatorInstructionLayer
        )
        self.assertIsInstance(candidate.block_layers[0], OperatorBlockLayer)
        for attribute in (
            "input_proj",
            "callee_proj",
            "instruction_input",
            "block_input",
            "function_input",
            "root_readout",
            "global_features",
            "context_encoder",
            "classifier",
        ):
            self.assertEqual(
                type(getattr(canonical, attribute)),
                type(getattr(candidate, attribute)),
            )


if __name__ == "__main__":
    unittest.main()
