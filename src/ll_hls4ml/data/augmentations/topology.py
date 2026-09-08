"""Deterministic topology controls for structural-necessity experiments."""

from __future__ import annotations

import hashlib

import torch

from ll_hls4ml.data.augmentations.base import register
from ll_hls4ml.io.schema import DERIVED_DEF_USE_EDGE


TOPOLOGY_DESTINATION_PERMUTE_VERSION = "within_function_destination_v2"
_CORRUPTED_EDGES = (
    ("instruction", "control", "instruction"),
    DERIVED_DEF_USE_EDGE,
    ("block", "control", "block"),
)


def _seed(seed: int, graph_key: str, edge_type: tuple[str, str, str]) -> int:
    value = f"{seed}\0{graph_key}\0{'/'.join(edge_type)}"
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")


def _owners(graph):
    block_owner = torch.full((graph["block"].num_nodes,), -1, dtype=torch.long)
    function_block = graph[("function", "contains", "block")].edge_index.cpu()
    block_owner[function_block[1]] = function_block[0]
    instruction_block = torch.full((graph["instruction"].num_nodes,), -1, dtype=torch.long)
    block_instruction = graph[("block", "contains", "instruction")].edge_index.cpu()
    instruction_block[block_instruction[1]] = block_instruction[0]
    if (block_owner < 0).any() or (instruction_block < 0).any():
        raise ValueError("Topology corruption requires complete hierarchy containment")
    return {"block": block_owner, "instruction": block_owner[instruction_block]}


class PermuteDestinationsWithinFunction:
    """Preserve edge counts and degree marginals while destroying adjacency."""

    def __init__(self, seed: int = 42, graph_key: str | None = None):
        self.seed = int(seed)
        self.graph_key = graph_key

    def __call__(self, graph):
        result = graph.clone()
        owners = _owners(result)
        graph_key = self.graph_key or str(getattr(result, "graph_id", "graph"))
        for edge_type in _CORRUPTED_EDGES:
            if edge_type not in result.edge_types:
                continue
            edge_index = result[edge_type].edge_index
            if edge_index.numel() == 0:
                continue
            source_owner = owners[edge_type[0]].to(edge_index.device)[edge_index[0]]
            destination_owner = owners[edge_type[2]].to(edge_index.device)[edge_index[1]]
            destinations = edge_index[1].clone()
            generator = torch.Generator(device=edge_index.device)
            generator.manual_seed(_seed(self.seed, graph_key, edge_type))
            # Only corrupt intra-function topology. Cross-function def-use/control
            # edges remain intact instead of becoming a second intervention.
            intrafunction = source_owner == destination_owner
            for owner in torch.unique(destination_owner[intrafunction], sorted=True):
                positions = torch.nonzero(
                    intrafunction & (destination_owner == owner), as_tuple=False
                ).flatten()
                permutation = torch.randperm(len(positions), generator=generator, device=edge_index.device)
                destinations[positions] = destinations[positions[permutation]]
            result[edge_type].edge_index = torch.stack((edge_index[0], destinations))
        result.topology_transform = TOPOLOGY_DESTINATION_PERMUTE_VERSION
        result.topology_transform_seed = self.seed
        return result


@register("permute_destinations_within_function_v2")
def build_permute_destinations_within_function(seed: int = 42, **_kwargs):
    return PermuteDestinationsWithinFunction(seed=seed)
