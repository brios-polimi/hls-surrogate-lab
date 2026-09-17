import unittest

import torch

from ll_hls4ml.models.hierarchical_variables import VARIABLE_ROUTE_PROFILES
from ll_hls4ml.models.registry import build
from tests.test_hierarchy_ablation import _tiny_hierarchy


class VariableRouteTests(unittest.TestCase):
    def test_all_profiles_complete_hierarchical_backward(self):
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
        }
        for name, profile in VARIABLE_ROUTE_PROFILES.items():
            with self.subTest(profile=name):
                model = build(
                    "hierarchical_variable_route", **profile, **common
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

    def test_all_variable_route_updates_are_registered(self):
        model = build(
            "hierarchical_variable_route",
            **VARIABLE_ROUTE_PROFILES["variable_route_all_2round"],
            instruction_vocab_size=8,
            edge_pos_vocab_size=3,
            y_means=torch.zeros(6),
            y_stds=torch.ones(6),
            hidden_dim=8,
            num_layers=2,
            dropout=0.0,
            use_global_features=False,
            use_context=False,
            split_heads=True,
            hurdle_heads=False,
        )
        self.assertEqual(len(model.variable_exchange), 2)
        self.assertEqual(model.variable_route, "all")


if __name__ == "__main__":
    unittest.main()
