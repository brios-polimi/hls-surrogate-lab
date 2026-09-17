import unittest

import torch

from ll_hls4ml.models.hierarchical_edge_attention import (
    EDGE_ATTENTION_PROFILES,
    EdgeAttentionBlockLayer,
)
from ll_hls4ml.models.registry import build
from tests.test_hierarchy_ablation import _tiny_hierarchy


class EdgeAttentionTests(unittest.TestCase):
    def test_direction_changes_sparse_attention_result(self):
        torch.manual_seed(41)
        layer = EdgeAttentionBlockLayer(8, 0.0, heads=2, bidirectional=False)
        state = torch.randn(4, 8)
        edges = torch.tensor([[0, 0, 1], [1, 2, 3]])
        self.assertFalse(
            torch.allclose(layer(state, edges), layer(state, edges.flip(0)))
        )

    def test_profiles_complete_hierarchical_backward(self):
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
            "edge_attention_heads": 2,
        }
        for name, profile in EDGE_ATTENTION_PROFILES.items():
            with self.subTest(profile=name):
                model = build(
                    "hierarchical_edge_attention", **profile, **common
                )
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
