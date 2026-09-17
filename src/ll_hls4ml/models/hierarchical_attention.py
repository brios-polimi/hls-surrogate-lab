"""Containment-bounded attention refiners for the canonical hierarchy."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from ll_hls4ml.models.hierarchical_operators import CDFGHierarchicalOperator


ATTENTION_PROFILES = {
    "attn_instruction": {
        "attention_scope": "instruction",
        "operator_profile": "forward_mean",
    },
    "attn_block": {
        "attention_scope": "block",
        "operator_profile": "forward_mean",
    },
    "attn_dual": {
        "attention_scope": "dual",
        "operator_profile": "forward_mean",
    },
    "attn_dual_wide_operator": {
        "attention_scope": "dual",
        "operator_profile": "wide_full_operator",
    },
}


def _length_buckets(lengths: torch.Tensor, pair_budget: int) -> list[list[int]]:
    values = lengths.detach().cpu().tolist()
    ordered = sorted(
        (index for index, length in enumerate(values) if length),
        key=values.__getitem__,
        reverse=True,
    )
    buckets = []
    cursor = 0
    while cursor < len(ordered):
        maximum = int(values[ordered[cursor]]) + 1  # account for padding safety
        width = max(1, pair_budget // max(maximum * maximum, 1))
        buckets.append(ordered[cursor : cursor + width])
        cursor += width
    return buckets


def _dense_group_chunk(
    state: torch.Tensor,
    group: torch.Tensor,
    group_ids: list[int],
    total_groups: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected_groups = torch.as_tensor(group_ids, device=group.device)
    remap = torch.full((total_groups,), -1, dtype=torch.long, device=group.device)
    remap[selected_groups] = torch.arange(len(group_ids), device=group.device)
    local_group = remap[group]
    selected = (local_group >= 0).nonzero(as_tuple=False).flatten()
    order = torch.argsort(local_group[selected], stable=True)
    selected = selected[order]
    dense, mask = to_dense_batch(
        state[selected],
        local_group[selected],
        batch_size=len(group_ids),
    )
    return dense, mask, selected


def _sinusoidal_positions(length: int, width: int, reference: torch.Tensor):
    position = torch.arange(length, device=reference.device, dtype=torch.float32)
    frequency = torch.exp(
        torch.arange(0, width, 2, device=reference.device, dtype=torch.float32)
        * (-math.log(10_000.0) / max(width, 1))
    )
    encoding = torch.zeros((length, width), device=reference.device)
    encoding[:, 0::2] = torch.sin(position[:, None] * frequency)
    encoding[:, 1::2] = torch.cos(
        position[:, None] * frequency[: encoding[:, 1::2].shape[1]]
    )
    return encoding.to(reference.dtype)


class GroupedSelfAttentionRefiner(nn.Module):
    """Full self-attention independently inside known containment groups."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        layers: int,
        dropout: float,
        feedforward_multiplier: int = 4,
        pair_budget: int = 131_072,
    ):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if layers < 1 or feedforward_multiplier < 1 or pair_budget < 1:
            raise ValueError("Attention layers, multiplier, and budget must be positive")
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=feedforward_multiplier * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.pair_budget = pair_budget

    def forward(
        self,
        state: torch.Tensor,
        group: torch.Tensor,
        group_count: int,
    ) -> torch.Tensor:
        lengths = torch.bincount(group, minlength=group_count)
        context = torch.zeros_like(state)
        for group_ids in _length_buckets(lengths, self.pair_budget):
            dense, mask, selected = _dense_group_chunk(
                state, group, group_ids, group_count
            )
            positioned = dense + _sinusoidal_positions(
                dense.size(1), dense.size(2), dense
            )
            encoded = self.encoder(positioned, src_key_padding_mask=~mask)
            context[selected] = (encoded - positioned)[mask].to(context.dtype)
        gate = torch.sigmoid(self.gate(torch.cat([state, context], dim=-1)))
        return self.norm(state + gate * context)


class CDFGHierarchicalAttention(CDFGHierarchicalOperator):
    """Canonical hierarchy plus full attention within blocks and/or functions."""

    def __init__(
        self,
        *args,
        attention_scope: str,
        operator_profile: str = "forward_mean",
        attention_heads: int = 4,
        attention_layers: int = 2,
        attention_feedforward_multiplier: int = 4,
        attention_pair_budget: int = 131_072,
        **kwargs,
    ):
        if attention_scope not in {"instruction", "block", "dual"}:
            raise ValueError(
                "attention_scope must be 'instruction', 'block', or 'dual'"
            )
        super().__init__(
            *args,
            operator_profile=operator_profile,
            **kwargs,
        )
        hidden_dim = int(self.instruction_layers[0].update.norm.normalized_shape[0])
        dropout = float(self.instruction_layers[0].update.dropout.p)
        refiner_kwargs = {
            "hidden_dim": hidden_dim,
            "heads": attention_heads,
            "layers": attention_layers,
            "dropout": dropout,
            "feedforward_multiplier": attention_feedforward_multiplier,
            "pair_budget": attention_pair_budget,
        }
        self.attention_scope = attention_scope
        self.instruction_attention = (
            GroupedSelfAttentionRefiner(**refiner_kwargs)
            if attention_scope in {"instruction", "dual"}
            else None
        )
        self.block_attention = (
            GroupedSelfAttentionRefiner(**refiner_kwargs)
            if attention_scope in {"block", "dual"}
            else None
        )

    def _refine_instruction_state(
        self,
        state: torch.Tensor,
        instruction_block: torch.Tensor,
        block_count: int,
    ) -> torch.Tensor:
        if self.instruction_attention is None:
            return state
        return self.instruction_attention(state, instruction_block, block_count)

    def _refine_block_state(
        self,
        state: torch.Tensor,
        block_function: torch.Tensor,
        function_count: int,
    ) -> torch.Tensor:
        if self.block_attention is None:
            return state
        return self.block_attention(state, block_function, function_count)
