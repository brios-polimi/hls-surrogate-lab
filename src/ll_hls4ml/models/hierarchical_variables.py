"""First-class variable-state routes on the frozen heterogeneous graph."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ll_hls4ml.models.hierarchical import _messages
from ll_hls4ml.models.hierarchical_operators import CDFGHierarchicalOperator


VARIABLE_ROUTE_PROFILES = {
    "variable_route_all": {
        "variable_route": "all",
        "variable_rounds": 1,
        "operator_profile": "forward_mean",
    },
    "variable_route_memory": {
        "variable_route": "memory",
        "variable_rounds": 1,
        "operator_profile": "forward_mean",
    },
    "variable_route_all_2round": {
        "variable_route": "all",
        "variable_rounds": 2,
        "operator_profile": "forward_mean",
    },
    "variable_route_wide_operator": {
        "variable_route": "all",
        "variable_rounds": 1,
        "operator_profile": "wide_full_operator",
    },
}


class VariableExchangeLayer(nn.Module):
    """Alternating instruction-to-variable-to-instruction update."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.producer_to_variable = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.consumer_to_variable = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.variable_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.variable_to_consumer = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.variable_to_producer = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.consumer_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.producer_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.instruction_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _zero(self, state: torch.Tensor) -> torch.Tensor:
        zero = state.sum() * 0
        for parameter in self.parameters():
            zero = zero + parameter.sum() * 0
        return state + zero

    def forward(
        self,
        instruction_state: torch.Tensor,
        variable_state: torch.Tensor,
        variable_local: torch.Tensor,
        defines: torch.Tensor,
        operands: torch.Tensor,
        operand_features: torch.Tensor,
        instruction_local: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not variable_state.numel():
            return self._zero(instruction_state), variable_state
        local_defines = torch.stack(
            [instruction_local[defines[0]], variable_local[defines[1]]]
        )
        local_operands_to_variable = torch.stack(
            [instruction_local[operands[1]], variable_local[operands[0]]]
        )
        producer = _messages(
            self.producer_to_variable(instruction_state),
            local_defines,
            variable_state.size(0),
        )
        consumer = _messages(
            self.consumer_to_variable(instruction_state),
            local_operands_to_variable,
            variable_state.size(0),
        )
        updated_variable = self.variable_update(
            F.gelu(producer + consumer),
            variable_state,
        )
        local_operands_to_instruction = torch.stack(
            [variable_local[operands[0]], instruction_local[operands[1]]]
        )
        consumer_message = _messages(
            self.variable_to_consumer(updated_variable),
            local_operands_to_instruction,
            instruction_state.size(0),
            operand_features,
        )
        local_defines_to_instruction = torch.stack(
            [variable_local[defines[1]], instruction_local[defines[0]]]
        )
        producer_message = _messages(
            self.variable_to_producer(updated_variable),
            local_defines_to_instruction,
            instruction_state.size(0),
        )
        consumer_message = consumer_message * torch.sigmoid(
            self.consumer_gate(
                torch.cat([instruction_state, consumer_message], dim=-1)
            )
        )
        producer_message = producer_message * torch.sigmoid(
            self.producer_gate(
                torch.cat([instruction_state, producer_message], dim=-1)
            )
        )
        proposal = self.dropout(F.gelu(consumer_message + producer_message))
        return (
            self.norm(self.instruction_update(proposal, instruction_state)),
            updated_variable,
        )


class CDFGHierarchicalVariableRoute(CDFGHierarchicalOperator):
    """Canonical hierarchy with recurrent variable nodes kept first class."""

    def __init__(
        self,
        *args,
        variable_route: str,
        variable_rounds: int = 1,
        operator_profile: str = "forward_mean",
        **kwargs,
    ):
        if variable_route not in {"all", "memory"}:
            raise ValueError("variable_route must be 'all' or 'memory'")
        if variable_rounds < 1:
            raise ValueError("variable_rounds must be positive")
        super().__init__(*args, operator_profile=operator_profile, **kwargs)
        hidden_dim = int(self.instruction_layers[0].update.norm.normalized_shape[0])
        dropout = float(self.instruction_layers[0].update.dropout.p)
        self.variable_route = variable_route
        self.variable_rounds = variable_rounds
        self.variable_exchange = nn.ModuleList(
            VariableExchangeLayer(hidden_dim, dropout)
            for _ in range(variable_rounds)
        )

    def _exchange_variable_state(
        self,
        state: torch.Tensor,
        variable_state: torch.Tensor,
        data,
        instruction_ids: torch.Tensor,
        instruction_local: torch.Tensor,
        operand_edge_features: torch.Tensor,
    ) -> torch.Tensor:
        defines = data[("instruction", "defines", "variable")].edge_index
        operands = data[("variable", "operand", "instruction")].edge_index
        define_mask = instruction_local[defines[0]] >= 0
        operand_mask = instruction_local[operands[1]] >= 0
        if self.variable_route == "memory":
            memory_like = data["variable"].x[:, 5:10].bool().any(dim=-1)
            define_mask &= memory_like[defines[1]]
            operand_mask &= memory_like[operands[0]]
        local_defines = defines[:, define_mask]
        local_operands = operands[:, operand_mask]
        variable_ids = torch.unique(
            torch.cat([local_defines[1], local_operands[0]])
        )
        variable_local = torch.full(
            (data["variable"].num_nodes,),
            -1,
            dtype=torch.long,
            device=state.device,
        )
        variable_local[variable_ids] = torch.arange(
            variable_ids.numel(), device=state.device
        )
        local_operand_features = operand_edge_features[operand_mask]
        current_variable = variable_state[variable_ids]
        for layer in self.variable_exchange:
            state, current_variable = layer(
                state,
                current_variable,
                variable_local,
                local_defines,
                local_operands,
                local_operand_features,
                instruction_local,
            )
        return state
