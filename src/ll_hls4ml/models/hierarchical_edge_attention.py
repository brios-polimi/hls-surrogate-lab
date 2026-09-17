"""Relation-specific sparse edge attention for the canonical hierarchy."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter, softmax

from ll_hls4ml.models.hierarchical import CDFGHierarchical


EDGE_ATTENTION_PROFILES = {
    "edge_attention": {"edge_attention_bidirectional": False},
    "edge_attention_bidir": {"edge_attention_bidirectional": True},
}


class EdgeAttentionFlow(nn.Module):
    """Multi-head attention along one directed edge relation."""

    def __init__(self, hidden_dim: int, heads: int):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by edge_attention_heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        state: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self.query(state).view(-1, self.heads, self.head_dim)
        key = self.key(state).view(-1, self.heads, self.head_dim)
        value = self.value(state).view(-1, self.heads, self.head_dim)
        if edge_index.numel() == 0:
            zero = (query.sum() + key.sum() + value.sum()) * 0
            if edge_features is not None:
                zero = zero + edge_features.sum() * 0
            return self.output(state * 0 + zero)

        source, target = edge_index
        edge_key = key[source]
        edge_value = value[source]
        if edge_features is not None:
            edge = edge_features.view(-1, self.heads, self.head_dim)
            edge_key = edge_key + edge
            edge_value = edge_value + edge
        logits = (query[target] * edge_key).sum(dim=-1) / math.sqrt(self.head_dim)
        weights = softmax(logits, target, num_nodes=state.size(0))
        aggregated = scatter(
            edge_value * weights.unsqueeze(-1),
            target,
            dim=0,
            dim_size=state.size(0),
            reduce="sum",
        )
        return self.output(aggregated.reshape(state.size(0), -1))


class EdgeAttentionUpdate(nn.Module):
    def __init__(self, hidden_dim: int, flow_count: int, dropout: float):
        super().__init__()
        self.merge = nn.Linear(flow_count * hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, state: torch.Tensor, messages: list[torch.Tensor]
    ) -> torch.Tensor:
        update = self.dropout(F.gelu(self.merge(torch.cat(messages, dim=-1))))
        return self.norm(state + update)


class EdgeAttentionInstructionLayer(nn.Module):
    def __init__(
        self, hidden_dim: int, dropout: float, heads: int, bidirectional: bool
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.control_forward = EdgeAttentionFlow(hidden_dim, heads)
        self.def_use_forward = EdgeAttentionFlow(hidden_dim, heads)
        self.control_reverse = (
            EdgeAttentionFlow(hidden_dim, heads) if bidirectional else None
        )
        self.def_use_reverse = (
            EdgeAttentionFlow(hidden_dim, heads) if bidirectional else None
        )
        self.update = EdgeAttentionUpdate(
            hidden_dim, 4 if bidirectional else 2, dropout
        )

    def forward(
        self,
        state: torch.Tensor,
        control_edges: torch.Tensor,
        def_use_edges: torch.Tensor,
        control_features: torch.Tensor | None,
        use_messages: bool = True,
    ) -> torch.Tensor:
        messages = [
            self.control_forward(state, control_edges, control_features),
            self.def_use_forward(state, def_use_edges),
        ]
        if self.bidirectional:
            messages.extend(
                [
                    self.control_reverse(
                        state, control_edges.flip(0), control_features
                    ),
                    self.def_use_reverse(state, def_use_edges.flip(0)),
                ]
            )
        if not use_messages:
            messages = [message * 0 for message in messages]
        return self.update(state, messages)


class EdgeAttentionBlockLayer(nn.Module):
    def __init__(
        self, hidden_dim: int, dropout: float, heads: int, bidirectional: bool
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.forward_flow = EdgeAttentionFlow(hidden_dim, heads)
        self.reverse_flow = (
            EdgeAttentionFlow(hidden_dim, heads) if bidirectional else None
        )
        self.update = EdgeAttentionUpdate(
            hidden_dim, 2 if bidirectional else 1, dropout
        )

    def forward(
        self,
        state: torch.Tensor,
        cfg_edges: torch.Tensor,
        use_messages: bool = True,
    ) -> torch.Tensor:
        messages = [self.forward_flow(state, cfg_edges)]
        if self.bidirectional:
            messages.append(self.reverse_flow(state, cfg_edges.flip(0)))
        if not use_messages:
            messages = [message * 0 for message in messages]
        return self.update(state, messages)


class CDFGHierarchicalEdgeAttention(CDFGHierarchical):
    """Canonical hierarchy with sparse relation-specific local attention."""

    def __init__(
        self,
        *args,
        edge_attention_bidirectional: bool = False,
        edge_attention_heads: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        instruction_count = len(self.instruction_layers)
        block_count = len(self.block_layers)
        hidden_dim = int(self.instruction_layers[0].norm.normalized_shape[0])
        dropout = float(self.instruction_layers[0].dropout.p)
        self.edge_attention_bidirectional = edge_attention_bidirectional
        self.edge_attention_heads = edge_attention_heads
        self.instruction_layers = nn.ModuleList(
            EdgeAttentionInstructionLayer(
                hidden_dim,
                dropout,
                edge_attention_heads,
                edge_attention_bidirectional,
            )
            for _ in range(instruction_count)
        )
        self.block_layers = nn.ModuleList(
            EdgeAttentionBlockLayer(
                hidden_dim,
                dropout,
                edge_attention_heads,
                edge_attention_bidirectional,
            )
            for _ in range(block_count)
        )
