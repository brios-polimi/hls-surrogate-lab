import unittest

import torch
from torch_geometric.data import HeteroData

from ll_hls4ml.data.augmentations.topology import PermuteDestinationsWithinFunction
from ll_hls4ml.io.schema import DERIVED_DEF_USE_EDGE


class TopologyAugmentationTests(unittest.TestCase):
    @staticmethod
    def graph():
        graph = HeteroData()
        graph.graph_id = "example"
        graph["instruction"].x = torch.zeros((6, 1), dtype=torch.long)
        graph["block"].x = torch.zeros((4, 3))
        graph["function"].x = torch.zeros((2, 2))
        graph[("function", "contains", "block")].edge_index = torch.tensor(
            [[0, 0, 1, 1], [0, 1, 2, 3]]
        )
        graph[("block", "contains", "instruction")].edge_index = torch.tensor(
            [[0, 0, 1, 2, 2, 3], [0, 1, 2, 3, 4, 5]]
        )
        graph[("instruction", "control", "instruction")].edge_index = torch.tensor(
            [[0, 0, 1, 3, 3, 4], [1, 2, 2, 4, 5, 5]]
        )
        graph[DERIVED_DEF_USE_EDGE].edge_index = torch.tensor(
            [[0, 1, 1, 3, 4], [2, 2, 3, 5, 5]]
        )
        graph[("block", "control", "block")].edge_index = torch.tensor(
            [[0, 0, 2, 2], [0, 1, 2, 3]]
        )
        return graph

    def test_is_deterministic_and_preserves_degree_marginals(self):
        original = self.graph()
        transform = PermuteDestinationsWithinFunction(seed=19)
        left = transform(original)
        right = transform(original)
        for edge_type in (
            ("instruction", "control", "instruction"),
            DERIVED_DEF_USE_EDGE,
            ("block", "control", "block"),
        ):
            old = original[edge_type].edge_index
            new = left[edge_type].edge_index
            self.assertTrue(torch.equal(new, right[edge_type].edge_index))
            self.assertTrue(torch.equal(old[0], new[0]))
            self.assertTrue(torch.equal(torch.sort(old[1]).values, torch.sort(new[1]).values))
        old_def_use = original[DERIVED_DEF_USE_EDGE].edge_index
        new_def_use = left[DERIVED_DEF_USE_EDGE].edge_index
        cross_function_position = 2
        self.assertTrue(torch.equal(
            old_def_use[:, cross_function_position],
            new_def_use[:, cross_function_position],
        ))
        self.assertTrue(any(
            not torch.equal(
                original[edge_type].edge_index,
                left[edge_type].edge_index,
            )
            for edge_type in (
                ("instruction", "control", "instruction"),
                DERIVED_DEF_USE_EDGE,
                ("block", "control", "block"),
            )
        ))
        self.assertTrue(torch.equal(
            original[("function", "contains", "block")].edge_index,
            left[("function", "contains", "block")].edge_index,
        ))


if __name__ == "__main__":
    unittest.main()
