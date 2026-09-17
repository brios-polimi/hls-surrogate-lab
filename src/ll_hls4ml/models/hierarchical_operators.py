"""Controlled local-operator variants for the canonical hierarchy.

Only instruction and block message/update layers change here. The surrounding
hierarchy, pooling, call composition, readout, context, and heads are inherited
unchanged from :mod:`ll_hls4ml.models.hierarchical`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

from ll_hls4ml.models.hierarchical import CDFGHierarchical


@dataclass(frozen=True)
class OperatorSpec:
    """Independent axes of a local message/update operator."""

    receiver_gate: bool = False
    bidirectional: bool = False
    aggregation: str = "mean"
    relation_mixer: str = "sum"
    update: str = "residual"

    def validate(self) -> None:
        if self.aggregation not in {"mean", "mean_max"}:
            raise ValueError(f"Unsupported aggregation: {self.aggregation}")
        if self.relation_mixer not in {"sum", "concat"}:
            raise ValueError(f"Unsupported relation mixer: {self.relation_mixer}")
        if self.update not in {"residual", "gru"}:
            raise ValueError(f"Unsupported update: {self.update}")


OPERATOR_PROFILES: dict[str, OperatorSpec] = {
    # Internal reference: functionally the canonical forward/mean/sum/residual
    # operator, represented with the generic implementation.
    "forward_mean": OperatorSpec(),
    "relation_mixer": OperatorSpec(relation_mixer="concat"),
    "receiver_gate": OperatorSpec(receiver_gate=True),
    "bidirectional": OperatorSpec(bidirectional=True),
    "gate_bidir": OperatorSpec(receiver_gate=True, bidirectional=True),
    "gate_bidir_meanmax": OperatorSpec(
        receiver_gate=True,
        bidirectional=True,
        aggregation="mean_max",
    ),
    "gate_bidir_gru": OperatorSpec(
        receiver_gate=True,
        bidirectional=True,
        update="gru",
    ),
    "full_operator": OperatorSpec(
        receiver_gate=True,
        bidirectional=True,
        aggregation="mean_max",
        update="gru",
    ),
}


def operator_spec(name: str) -> OperatorSpec:
    try:
        spec = OPERATOR_PROFILES[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown operator profile {name!r}; "
            f"available={sorted(OPERATOR_PROFILES)}"
        ) from error
    spec.validate()
    return spec


class _FlowBranch(nn.Module):
    """One directed, relation-specific message flow."""

    def __init__(self, hidden_dim: int, spec: OperatorSpec, rank: int = 8):
        super().__init__()
        self.spec = spec
        self.source = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.aggregate_correction = (
            _LowRankProjection(
                2 * hidden_dim, hidden_dim, rank, output_bias=False
            )
            if spec.aggregation == "mean_max"
            else None
        )
        self.gate = (
            _LowRankProjection(2 * hidden_dim, hidden_dim, rank)
            if spec.receiver_gate
            else None
        )
        if spec.bidirectional:
            self.reverse_scale = nn.Parameter(torch.empty(hidden_dim))
            nn.init.normal_(self.reverse_scale, mean=1.0, std=0.02)
            self.reverse_shift = nn.Parameter(torch.zeros(hidden_dim))
        else:
            self.register_parameter("reverse_scale", None)
            self.register_parameter("reverse_shift", None)

    def forward(
        self,
        state: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor | None,
        reverse: bool,
    ) -> torch.Tensor:
        projected = self.source(state)
        if reverse:
            if self.reverse_scale is None:
                raise ValueError("Reverse flow requested for a forward-only branch")
            projected = projected * self.reverse_scale + self.reverse_shift
        source_index = edge_index[1 if reverse else 0]
        target_index = edge_index[0 if reverse else 1]
        if edge_index.numel() == 0:
            zero = projected.sum(dim=0, keepdim=True) * 0
            if edge_features is not None:
                zero = zero + edge_features.sum(dim=0, keepdim=True) * 0
            mean = zero.expand(state.size(0), -1)
            maximum = mean
        else:
            values = projected[source_index]
            if edge_features is not None:
                values = values + edge_features
            mean = scatter(
                values,
                target_index,
                dim=0,
                dim_size=state.size(0),
                reduce="mean",
            )
            maximum = scatter(
                values,
                target_index,
                dim=0,
                dim_size=state.size(0),
                reduce="max",
            )
        message = mean
        if self.aggregate_correction is not None:
            message = message + self.aggregate_correction(
                torch.cat([mean, maximum], dim=-1)
            )
        if self.gate is not None:
            message = message * torch.sigmoid(
                self.gate(torch.cat([state, message], dim=-1))
            )
        return message


class _LowRankProjection(nn.Module):
    """Nonlinear low-rank map used to limit capacity confounding."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rank: int = 8,
        output_bias: bool = True,
    ):
        super().__init__()
        self.down = nn.Linear(input_dim, rank, bias=False)
        self.up = nn.Linear(rank, output_dim, bias=output_bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.up(F.gelu(self.down(value)))


class _LowRankGRUUpdate(nn.Module):
    """GRU-style state update without three full hidden-width matrices."""

    def __init__(self, hidden_dim: int, rank: int = 8):
        super().__init__()
        pair_dim = 2 * hidden_dim
        self.update_gate = _LowRankProjection(pair_dim, hidden_dim, rank)
        self.reset_gate = _LowRankProjection(pair_dim, hidden_dim, rank)
        self.candidate = _LowRankProjection(pair_dim, hidden_dim, rank)

    def forward(self, message: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([state, message], dim=-1)
        update = torch.sigmoid(self.update_gate(pair))
        reset = torch.sigmoid(self.reset_gate(pair))
        candidate = torch.tanh(
            message
            + self.candidate(torch.cat([reset * state, message], dim=-1))
        )
        return (1 - update) * state + update * candidate


class _OperatorUpdate(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        flow_count: int,
        spec: OperatorSpec,
        dropout: float,
        rank: int = 8,
    ):
        super().__init__()
        self.spec = spec
        self.mixer = (
            _LowRankProjection(
                flow_count * hidden_dim,
                hidden_dim,
                rank,
                output_bias=False,
            )
            if spec.relation_mixer == "concat" and flow_count > 1
            else None
        )
        self.gru = (
            _LowRankGRUUpdate(hidden_dim, rank)
            if spec.update == "gru"
            else None
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, state: torch.Tensor, messages: list[torch.Tensor]) -> torch.Tensor:
        mixed = torch.stack(messages, dim=0).sum(dim=0)
        if self.mixer is not None:
            mixed = mixed + self.mixer(torch.cat(messages, dim=-1))
        proposal = self.dropout(F.relu(mixed))
        if self.gru is not None:
            return self.norm(self.gru(proposal, state))
        return self.norm(state + proposal)


class OperatorInstructionLayer(nn.Module):
    """Relation-specific instruction flows with configurable update axes."""

    def __init__(self, hidden_dim: int, dropout: float, spec: OperatorSpec):
        super().__init__()
        self.spec = spec
        self.control_forward = _FlowBranch(hidden_dim, spec)
        self.def_use_forward = _FlowBranch(hidden_dim, spec)
        self.update = _OperatorUpdate(
            hidden_dim,
            flow_count=4 if spec.bidirectional else 2,
            spec=spec,
            dropout=dropout,
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
            self.control_forward(state, control_edges, control_features, False),
            self.def_use_forward(state, def_use_edges, None, False),
        ]
        if self.spec.bidirectional:
            messages.extend(
                [
                    self.control_forward(
                        state, control_edges, control_features, True
                    ),
                    self.def_use_forward(state, def_use_edges, None, True),
                ]
            )
        if not use_messages:
            messages = [message * 0 for message in messages]
        return self.update(state, messages)


class OperatorBlockLayer(nn.Module):
    """Configurable predecessor/successor flow on the block CFG."""

    def __init__(self, hidden_dim: int, dropout: float, spec: OperatorSpec):
        super().__init__()
        self.spec = spec
        self.forward_flow = _FlowBranch(hidden_dim, spec)
        self.update = _OperatorUpdate(
            hidden_dim,
            flow_count=2 if spec.bidirectional else 1,
            spec=spec,
            dropout=dropout,
        )

    def forward(
        self,
        state: torch.Tensor,
        cfg_edges: torch.Tensor,
        use_messages: bool = True,
    ) -> torch.Tensor:
        messages = [self.forward_flow(state, cfg_edges, None, False)]
        if self.spec.bidirectional:
            messages.append(self.forward_flow(state, cfg_edges, None, True))
        if not use_messages:
            messages = [message * 0 for message in messages]
        return self.update(state, messages)


class CDFGHierarchicalOperator(CDFGHierarchical):
    """Canonical hierarchy with only its local operators replaced."""

    def __init__(self, *args, operator_profile: str, **kwargs):
        super().__init__(*args, **kwargs)
        spec = operator_spec(operator_profile)
        instruction_count = len(self.instruction_layers)
        block_count = len(self.block_layers)
        if not instruction_count or not block_count:
            raise ValueError("Operator variants require positive layer counts")
        hidden_dim = int(self.instruction_layers[0].norm.normalized_shape[0])
        dropout = float(self.instruction_layers[0].dropout.p)
        self.operator_profile = operator_profile
        self.operator_spec = asdict(spec)
        self.instruction_layers = nn.ModuleList(
            OperatorInstructionLayer(hidden_dim, dropout, spec)
            for _ in range(instruction_count)
        )
        self.block_layers = nn.ModuleList(
            OperatorBlockLayer(hidden_dim, dropout, spec)
            for _ in range(block_count)
        )
