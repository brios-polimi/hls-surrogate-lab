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
    rank: int | None = 8

    def validate(self) -> None:
        if self.aggregation not in {"mean", "mean_max", "pna"}:
            raise ValueError(f"Unsupported aggregation: {self.aggregation}")
        if self.relation_mixer not in {"sum", "concat"}:
            raise ValueError(f"Unsupported relation mixer: {self.relation_mixer}")
        if self.update not in {"residual", "gru"}:
            raise ValueError(f"Unsupported update: {self.update}")
        if self.rank is not None and self.rank < 1:
            raise ValueError("Operator rank must be positive or None for full width")


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
    "pna": OperatorSpec(aggregation="pna"),
    "pna_gate_bidir": OperatorSpec(
        receiver_gate=True,
        bidirectional=True,
        aggregation="pna",
    ),
    # Capacity-seeking track. ``rank=None`` uses full-width linear maps and a
    # standard GRUCell instead of the rank-8 mechanism probes above.
    "wide_receiver_gate": OperatorSpec(receiver_gate=True, rank=None),
    "wide_gate_bidir": OperatorSpec(
        receiver_gate=True,
        bidirectional=True,
        rank=None,
    ),
    "wide_gate_bidir_meanmax": OperatorSpec(
        receiver_gate=True,
        bidirectional=True,
        aggregation="mean_max",
        rank=None,
    ),
    "wide_full_operator": OperatorSpec(
        receiver_gate=True,
        bidirectional=True,
        aggregation="mean_max",
        update="gru",
        rank=None,
    ),
    "pna_wide": OperatorSpec(aggregation="pna", rank=None),
    "pna_wide_full": OperatorSpec(
        receiver_gate=True,
        bidirectional=True,
        aggregation="pna",
        update="gru",
        rank=None,
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

    def __init__(
        self,
        hidden_dim: int,
        spec: OperatorSpec,
        pna_avg_log_degree: tuple[float, float] | None = None,
    ):
        super().__init__()
        self.spec = spec
        rank = spec.rank
        self.source = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.aggregate_correction = (
            _LowRankProjection(
                2 * hidden_dim, hidden_dim, rank, output_bias=False
            )
            if spec.aggregation == "mean_max"
            else None
        )
        self.pna_correction = (
            _LowRankProjection(
                12 * hidden_dim, hidden_dim, rank, output_bias=False
            )
            if spec.aggregation == "pna"
            else None
        )
        if spec.aggregation == "pna":
            if pna_avg_log_degree is None:
                raise ValueError("PNA aggregation requires forward/reverse degree statistics")
            self.register_buffer(
                "pna_avg_log_degree",
                torch.tensor(pna_avg_log_degree, dtype=torch.float32),
            )
        else:
            self.register_buffer("pna_avg_log_degree", None)
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
            minimum = scatter(
                values,
                target_index,
                dim=0,
                dim_size=state.size(0),
                reduce="min",
            )
            mean_square = scatter(
                values.square(),
                target_index,
                dim=0,
                dim_size=state.size(0),
                reduce="mean",
            )
            standard_deviation = (
                mean_square - mean.square()
            ).clamp_min(0).add(1e-6).sqrt()
        if edge_index.numel() == 0:
            minimum = mean
            standard_deviation = mean
        message = mean
        if self.aggregate_correction is not None:
            message = message + self.aggregate_correction(
                torch.cat([mean, maximum], dim=-1)
            )
        if self.pna_correction is not None:
            degree = torch.bincount(
                target_index, minlength=state.size(0)
            ).to(dtype=state.dtype, device=state.device)
            log_degree = torch.log1p(degree).unsqueeze(-1)
            average = self.pna_avg_log_degree[1 if reverse else 0].to(
                dtype=state.dtype, device=state.device
            ).clamp_min(1e-3)
            amplification = log_degree / average
            attenuation = average / log_degree.clamp_min(1e-3)
            aggregators = torch.cat(
                [mean, maximum, minimum, standard_deviation], dim=-1
            )
            aggregators = aggregators * (degree > 0).unsqueeze(-1)
            scaled = torch.cat(
                [aggregators, aggregators * amplification, aggregators * attenuation],
                dim=-1,
            )
            message = message + self.pna_correction(scaled)
        if self.gate is not None:
            message = message * torch.sigmoid(
                self.gate(torch.cat([state, message], dim=-1))
            )
        return message


class _LowRankProjection(nn.Module):
    """Low-rank nonlinear map, or one full-width linear map when rank is None."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rank: int | None = 8,
        output_bias: bool = True,
    ):
        super().__init__()
        self.direct = (
            nn.Linear(input_dim, output_dim, bias=output_bias)
            if rank is None
            else None
        )
        self.down = (
            nn.Linear(input_dim, rank, bias=False)
            if rank is not None
            else None
        )
        self.up = (
            nn.Linear(rank, output_dim, bias=output_bias)
            if rank is not None
            else None
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.direct is not None:
            return self.direct(value)
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
        rank: int | None = 8,
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
        self.gru = None
        if spec.update == "gru":
            self.gru = (
                nn.GRUCell(hidden_dim, hidden_dim)
                if rank is None
                else _LowRankGRUUpdate(hidden_dim, rank)
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

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        spec: OperatorSpec,
        pna_avg_log_degrees: dict[str, float] | None = None,
    ):
        super().__init__()
        self.spec = spec
        rank = spec.rank
        pna = pna_avg_log_degrees or {}
        self.control_forward = _FlowBranch(
            hidden_dim,
            spec,
            (
                pna.get("instruction_control_forward"),
                pna.get("instruction_control_reverse"),
            ) if spec.aggregation == "pna" else None,
        )
        self.def_use_forward = _FlowBranch(
            hidden_dim,
            spec,
            (
                pna.get("instruction_def_use_forward"),
                pna.get("instruction_def_use_reverse"),
            ) if spec.aggregation == "pna" else None,
        )
        self.update = _OperatorUpdate(
            hidden_dim,
            flow_count=4 if spec.bidirectional else 2,
            spec=spec,
            dropout=dropout,
            rank=rank,
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

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        spec: OperatorSpec,
        pna_avg_log_degrees: dict[str, float] | None = None,
    ):
        super().__init__()
        self.spec = spec
        rank = spec.rank
        pna = pna_avg_log_degrees or {}
        self.forward_flow = _FlowBranch(
            hidden_dim,
            spec,
            (
                pna.get("block_cfg_forward"),
                pna.get("block_cfg_reverse"),
            ) if spec.aggregation == "pna" else None,
        )
        self.update = _OperatorUpdate(
            hidden_dim,
            flow_count=2 if spec.bidirectional else 1,
            spec=spec,
            dropout=dropout,
            rank=rank,
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

    def __init__(
        self,
        *args,
        operator_profile: str,
        pna_avg_log_degrees: dict[str, float] | None = None,
        **kwargs,
    ):
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
            OperatorInstructionLayer(
                hidden_dim, dropout, spec, pna_avg_log_degrees
            )
            for _ in range(instruction_count)
        )
        self.block_layers = nn.ModuleList(
            OperatorBlockLayer(hidden_dim, dropout, spec, pna_avg_log_degrees)
            for _ in range(block_count)
        )
