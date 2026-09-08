"""Model registry for experiment notebooks."""

from __future__ import annotations

import torch.nn as nn

from ll_hls4ml.models.hetero_gat import CDFGHeteroGAT
from ll_hls4ml.models.hetero_relational import CDFGHeteroRelational
from ll_hls4ml.models.hierarchical import CDFGHierarchical
from ll_hls4ml.models.hierarchical_experimental import (
    CDFGHierarchicalBlockAttention,
    CDFGHierarchicalMemoryDual,
    CDFGHierarchicalSequence,
)
from ll_hls4ml.models.fusion import HierarchicalHighLevelFusion
from ll_hls4ml.models.paper_high_level import (
    PaperHighLevelGATv2,
    PaperTransformerRegressor,
)
from ll_hls4ml.models.mlp import MLP

MODELS: dict[str, type[nn.Module]] = {
    "hetero_gat": CDFGHeteroGAT,
    "hetero_relational": CDFGHeteroRelational,
    "hierarchical": CDFGHierarchical,
    "hierarchical_topology_destroyed": CDFGHierarchical,
    "hierarchical_sequence": CDFGHierarchicalSequence,
    "hierarchical_block_attention": CDFGHierarchicalBlockAttention,
    "hierarchical_memory_dual": CDFGHierarchicalMemoryDual,
    "hierarchical_high_level_fusion": HierarchicalHighLevelFusion,
    "paper_high_level_gatv2": PaperHighLevelGATv2,
    "paper_transformer": PaperTransformerRegressor,
    # Compatibility alias for existing configs and recorded experiments.
    "rgcn": CDFGHeteroGAT,
    "mlp": MLP,
    "pooled_control": MLP,
}


def list_models() -> list[str]:
    return sorted(MODELS.keys())


def build(name: str, **kwargs) -> nn.Module:
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Available: {list_models()}")
    return MODELS[name](**kwargs)
