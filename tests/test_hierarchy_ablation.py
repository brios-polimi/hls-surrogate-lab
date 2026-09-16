import copy
import unittest

import torch
from torch_geometric.data import HeteroData

from ll_hls4ml.data.tensorize import EMBED_SIZE
from ll_hls4ml.io.schema import (
    BLOCK_FEATURE_SIZE,
    DERIVED_DEF_USE_EDGE,
    EDGE_TYPES,
    FUNCTION_FEATURE_SIZE,
    PRAGMA_ARGUMENT_SIZE,
)
from ll_hls4ml.models.registry import build


def _empty_edges():
    return torch.empty((2, 0), dtype=torch.long)


def _tiny_hierarchy() -> HeteroData:
    data = HeteroData()
    data["instruction"].x = torch.tensor([[1], [2], [3], [1], [2], [3]])
    data["variable"].x = torch.randn(2, EMBED_SIZE)
    data["constant"].x = torch.randn(2, EMBED_SIZE)
    data["pragma"].x = torch.zeros(3, 1 + PRAGMA_ARGUMENT_SIZE)
    data["pragma"].x[:, 0] = torch.tensor([1, 2, 3])
    data["block"].x = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )[:, :BLOCK_FEATURE_SIZE]
    data["function"].x = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]]
    )[:, :FUNCTION_FEATURE_SIZE]

    for edge_type in EDGE_TYPES:
        data[edge_type].edge_index = _empty_edges()
    data[("instruction", "control", "instruction")].edge_index = torch.tensor(
        [[0, 1, 3, 4], [1, 2, 4, 5]]
    )
    data[("instruction", "defines", "variable")].edge_index = torch.tensor(
        [[0, 3], [0, 1]]
    )
    data[("variable", "operand", "instruction")].edge_index = torch.tensor(
        [[0, 1], [1, 4]]
    )
    data[("constant", "operand", "instruction")].edge_index = torch.tensor(
        [[0, 1], [2, 5]]
    )
    data[("instruction", "calls", "function")].edge_index = torch.tensor(
        [[2], [1]]
    )
    data[("pragma", "applies_to", "instruction")].edge_index = torch.tensor(
        [[0], [0]]
    )
    data[("pragma", "applies_to", "block")].edge_index = torch.tensor(
        [[1], [0]]
    )
    data[("pragma", "applies_to", "function")].edge_index = torch.tensor(
        [[2], [0]]
    )
    data[("block", "control", "block")].edge_index = torch.tensor(
        [[0, 2], [1, 3]]
    )
    data[("block", "contains", "instruction")].edge_index = torch.tensor(
        [[0, 0, 1, 2, 2, 3], [0, 1, 2, 3, 4, 5]]
    )
    data[("function", "contains", "block")].edge_index = torch.tensor(
        [[0, 0, 1, 1], [0, 1, 2, 3]]
    )
    data[DERIVED_DEF_USE_EDGE].edge_index = torch.tensor(
        [[0, 1, 3, 4], [1, 2, 4, 5]]
    )

    for edge_type in (
        ("instruction", "control", "instruction"),
        ("variable", "operand", "instruction"),
        ("constant", "operand", "instruction"),
    ):
        count = data[edge_type].edge_index.size(1)
        data[edge_type].edge_attr = torch.zeros((count, 1), dtype=torch.long)

    data["instruction"].call_depth = torch.tensor([1, 1, 1, 0, 0, 0])
    data["block"].call_depth = torch.tensor([1, 1, 0, 0])
    data["function"].call_depth = torch.tensor([1, 0])
    data["function"].is_reachable = torch.tensor([True, True])
    data["function"].is_entry = torch.tensor([True, False])
    data.hierarchy_schema_version = 2
    return data


def _model(name: str):
    return build(
        name,
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


class HierarchyAblationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_controls_preserve_parameter_scaffold(self):
        intact = _model("hierarchical")
        names = (
            "hierarchical_orderless",
            "hierarchical_no_block_cfg",
            "hierarchical_no_callee",
        )
        for name in names:
            control = _model(name)
            self.assertEqual(intact.state_dict().keys(), control.state_dict().keys())
            self.assertEqual(
                sum(parameter.numel() for parameter in intact.parameters()),
                sum(parameter.numel() for parameter in control.parameters()),
            )
        self.assertEqual(_model("hierarchical_orderless").hierarchy_mode, "orderless")
        self.assertFalse(_model("hierarchical_no_block_cfg").use_block_messages)
        self.assertFalse(_model("hierarchical_no_callee").use_callee_messages)

    def test_orderless_control_uses_every_parameter(self):
        model = _model("hierarchical_orderless")
        model(_tiny_hierarchy()).sum().backward()
        missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
        self.assertEqual(missing, [])

    def test_orderless_ignores_upper_structure_but_uses_instruction_edges(self):
        model = _model("hierarchical_orderless").eval()
        original = _tiny_hierarchy()
        rewired = copy.deepcopy(original)
        rewired[("instruction", "calls", "function")].edge_index = torch.tensor(
            [[1], [1]]
        )
        rewired[("block", "control", "block")].edge_index = torch.tensor(
            [[0, 2], [0, 2]]
        )
        rewired[("block", "contains", "instruction")].edge_index = torch.tensor(
            [[1, 1, 0, 3, 3, 2], [0, 1, 2, 3, 4, 5]]
        )
        rewired[("function", "contains", "block")].edge_index = torch.tensor(
            [[1, 0, 0, 1], [0, 1, 2, 3]]
        )
        with torch.no_grad():
            expected = model(original)
            actual = model(rewired)
        self.assertTrue(torch.equal(expected, actual))

        instruction_rewired = copy.deepcopy(original)
        instruction_rewired[("instruction", "control", "instruction")].edge_index = (
            torch.tensor([[0, 1, 3, 4], [2, 0, 5, 3]])
        )
        with torch.no_grad():
            changed = model(instruction_rewired)
        self.assertFalse(torch.allclose(expected, changed))


if __name__ == "__main__":
    unittest.main()
