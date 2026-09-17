import unittest

import torch

from ll_hls4ml.models.hierarchical_attention import (
    ATTENTION_PROFILES,
    CDFGHierarchicalAttention,
    GroupedSelfAttentionRefiner,
)
from ll_hls4ml.models.registry import build
from tests.test_hierarchy_ablation import _tiny_hierarchy


class GroupedAttentionTests(unittest.TestCase):
    def test_groups_do_not_exchange_information(self):
        torch.manual_seed(31)
        refiner = GroupedSelfAttentionRefiner(
            hidden_dim=8,
            heads=2,
            layers=1,
            dropout=0.0,
            pair_budget=64,
        ).eval()
        state = torch.randn(6, 8)
        groups = torch.tensor([0, 0, 0, 1, 1, 1])
        changed = state.clone()
        changed[3:] += 100
        with torch.no_grad():
            expected = refiner(state, groups, 2)
            actual = refiner(changed, groups, 2)
        self.assertTrue(torch.allclose(expected[:3], actual[:3], atol=1e-6))
        self.assertFalse(torch.allclose(expected[3:], actual[3:]))

    def test_all_attention_profiles_build(self):
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
            "attention_heads": 2,
            "attention_layers": 1,
        }
        for name, profile in ATTENTION_PROFILES.items():
            with self.subTest(profile=name):
                model = build("hierarchical_attention", **profile, **common)
                self.assertIsInstance(model, CDFGHierarchicalAttention)
                self.assertEqual(model.attention_scope, profile["attention_scope"])
                self.assertEqual(model.operator_profile, profile["operator_profile"])

    def test_all_attention_profiles_complete_hierarchical_backward(self):
        common = {
            "instruction_vocab_size": 8,
            "edge_pos_vocab_size": 3,
            "y_means": torch.zeros(6),
            "y_stds": torch.ones(6),
            "hidden_dim": 8,
            "num_layers": 2,
            "dropout": 0.0,
            "use_global_features": False,
            "use_context": False,
            "split_heads": True,
            "hurdle_heads": False,
            "attention_heads": 2,
            "attention_layers": 1,
        }
        for name, profile in ATTENTION_PROFILES.items():
            with self.subTest(profile=name):
                model = build("hierarchical_attention", **profile, **common)
                data = _tiny_hierarchy()
                data.num_graphs = 1
                model(data).sum().backward()
                missing = [
                    key
                    for key, parameter in model.named_parameters()
                    if parameter.grad is None
                ]
                self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
