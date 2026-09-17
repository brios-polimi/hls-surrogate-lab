"""Strict leaf-to-root hierarchical encoder for canonical LLVM CDFGs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

from ll_hls4ml.io.schema import DERIVED_DEF_USE_EDGE, LABEL_KEYS, NODE_TYPES
from ll_hls4ml.models.input_projection import CDFGInputProjection
from ll_hls4ml.models.readout import (
    GlobalFeatureEncoder,
    GraphContextEncoder,
    SplitRegressionHead,
    multi_pool,
)


def _messages(
    source: torch.Tensor,
    edge_index: torch.Tensor,
    target_count: int,
    edge_features: torch.Tensor | None = None,
) -> torch.Tensor:
    if edge_index.numel() == 0:
        # Sparse graph batches can omit an otherwise valid relation on one DDP
        # rank. Preserve a zero-gradient dependency on the source projection so
        # every rank reduces the same parameters without changing the message.
        zero = source.sum(dim=0, keepdim=True)
        if edge_features is not None:
            zero = zero + edge_features.sum(dim=0, keepdim=True)
        return zero.expand(target_count, -1) * 0
    values = source[edge_index[0]]
    if edge_features is not None:
        values = values + edge_features
    return scatter(
        values,
        edge_index[1],
        dim=0,
        dim_size=target_count,
        reduce="mean",
    )


def _reverse_messages(
    target: torch.Tensor,
    edge_index: torch.Tensor,
    source_count: int,
) -> torch.Tensor:
    if edge_index.numel() == 0:
        return target.sum(dim=0, keepdim=True).expand(
            source_count, -1
        ) * 0
    return scatter(
        target[edge_index[1]],
        edge_index[0],
        dim=0,
        dim_size=source_count,
        reduce="mean",
    )


class InstructionFlowLayer(nn.Module):
    """Update ready instructions over control and SSA def-use adjacency."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.control = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.def_use = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        state: torch.Tensor,
        control_edges: torch.Tensor,
        def_use_edges: torch.Tensor,
        control_features: torch.Tensor | None,
        use_messages: bool = True,
    ) -> torch.Tensor:
        control_source = self.control(state)
        data_source = self.def_use(state)
        if use_messages:
            control = _messages(
                control_source,
                control_edges,
                state.size(0),
                control_features,
            )
            data = _messages(
                data_source,
                def_use_edges,
                state.size(0),
            )
        else:
            # Keep the exact layer and parameter scaffold while removing only
            # adjacency-derived contributions. The zero dependencies also keep
            # all parameters participating in distributed gradient reduction.
            control = control_source.sum(dim=0, keepdim=True).expand_as(state) * 0
            if control_features is not None:
                control = control + control_features.sum(dim=0, keepdim=True) * 0
            data = data_source.sum(dim=0, keepdim=True).expand_as(state) * 0
        updated = self.norm(state + self.dropout(F.relu(control + data)))
        return updated


class BlockFlowLayer(nn.Module):
    """Update ready blocks over the LLVM basic-block CFG."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.message = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        state: torch.Tensor,
        cfg_edges: torch.Tensor,
        use_messages: bool = True,
    ) -> torch.Tensor:
        message_source = self.message(state)
        message = (
            _messages(message_source, cfg_edges, state.size(0))
            if use_messages
            else message_source.sum(dim=0, keepdim=True).expand_as(state) * 0
        )
        updated = self.norm(state + self.dropout(F.relu(message)))
        return updated


class CDFGHierarchical(nn.Module):
    """Encode a function only after all of its callees are complete.

    Functions sharing a leaf-first call depth are evaluated together. Within a
    function, instruction/SSA flow produces block states, the block CFG produces
    a function state, and that state is injected at every caller callsite.
    """

    def __init__(
        self,
        edge_pos_vocab_size: int,
        y_means: torch.Tensor,
        y_stds: torch.Tensor,
        instruction_vocab_size: int | None = None,
        hidden_dim: int = 128,
        num_layers: int = 3,
        instruction_num_layers: int | None = None,
        block_num_layers: int | None = None,
        dropout: float = 0.1,
        node_vocab_sizes: dict[str, int] | None = None,
        use_global_features: bool = False,
        use_context: bool = False,
        split_heads: bool = False,
        context_mode: str = "core",
        hurdle_heads: bool = False,
        hurdle_prediction_mode: str = "expected",
        build_head: bool = True,
        use_local_messages: bool = True,
        use_block_messages: bool | None = None,
        use_callee_messages: bool = True,
        hierarchy_mode: str = "structured",
    ):
        super().__init__()
        if hurdle_heads and not split_heads:
            raise ValueError("hurdle_heads requires split_heads=True")
        if hierarchy_mode not in {"structured", "orderless"}:
            raise ValueError("hierarchy_mode must be 'structured' or 'orderless'")
        if instruction_vocab_size is None:
            if not node_vocab_sizes or "instruction" not in node_vocab_sizes:
                raise ValueError("instruction_vocab_size is required")
            instruction_vocab_size = node_vocab_sizes["instruction"]
        self.register_buffer("y_means", y_means.clone())
        self.register_buffer("y_stds", y_stds.clone())
        self.hurdle_heads = hurdle_heads
        self.hurdle_prediction_mode = hurdle_prediction_mode
        self.use_global_features = use_global_features
        self.use_context = use_context
        self.use_local_messages = use_local_messages
        self.use_block_messages = (
            use_local_messages if use_block_messages is None else use_block_messages
        )
        self.use_callee_messages = use_callee_messages
        self.hierarchy_mode = hierarchy_mode
        instruction_num_layers = (
            num_layers if instruction_num_layers is None else instruction_num_layers
        )
        block_num_layers = (
            num_layers if block_num_layers is None else block_num_layers
        )

        self.input_proj = CDFGInputProjection(
            instruction_vocab_size, edge_pos_vocab_size, hidden_dim
        )
        self.callee_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.instruction_input = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.instruction_layers = nn.ModuleList(
            InstructionFlowLayer(hidden_dim, dropout)
            for _ in range(instruction_num_layers)
        )
        pooled_dim = 4 * hidden_dim + 1
        self.block_input = nn.Sequential(
            nn.Linear(pooled_dim + 2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.block_layers = nn.ModuleList(
            BlockFlowLayer(hidden_dim, dropout)
            for _ in range(block_num_layers)
        )
        self.function_input = nn.Sequential(
            nn.Linear(pooled_dim + 2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.root_readout = nn.Sequential(
            nn.Linear(2 * pooled_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        classifier_dim = hidden_dim
        if use_global_features:
            self.global_features = GlobalFeatureEncoder(
                instruction_vocab_size, hidden_dim
            )
            classifier_dim += hidden_dim
        if use_context:
            self.context_encoder = GraphContextEncoder(hidden_dim, context_mode)
            classifier_dim += hidden_dim
        self.output_dim = classifier_dim
        self.classifier = None
        if build_head:
            self.classifier = (
                SplitRegressionHead(
                    classifier_dim,
                    hidden_dim,
                    dropout,
                    hurdle_heads=hurdle_heads,
                )
                if split_heads
                else nn.Sequential(
                    nn.Linear(classifier_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, len(LABEL_KEYS)),
                )
            )

    @staticmethod
    def _owners(data):
        device = data["instruction"].x.device
        instruction_block = torch.full(
            (data["instruction"].num_nodes,), -1, dtype=torch.long, device=device
        )
        block_instruction = data[("block", "contains", "instruction")].edge_index
        instruction_block[block_instruction[1]] = block_instruction[0]
        block_function = torch.full(
            (data["block"].num_nodes,), -1, dtype=torch.long, device=device
        )
        function_block = data[("function", "contains", "block")].edge_index
        block_function[function_block[1]] = function_block[0]
        if (instruction_block < 0).any() or (block_function < 0).any():
            raise ValueError("Incomplete instruction/block/function containment")
        return instruction_block, block_function

    @staticmethod
    def _batch(data, node_type: str) -> torch.Tensor:
        batch = getattr(data[node_type], "batch", None)
        if batch is not None:
            return batch
        return torch.zeros(
            data[node_type].num_nodes,
            dtype=torch.long,
            device=data[node_type].x.device,
        )

    def _refine_instruction_state(
        self,
        state: torch.Tensor,
        instruction_block: torch.Tensor,
        block_count: int,
    ) -> torch.Tensor:
        """Optional within-block composition hook; canonical is identity."""
        return state

    def _exchange_variable_state(
        self,
        state: torch.Tensor,
        variable_state: torch.Tensor,
        data,
        instruction_ids: torch.Tensor,
        instruction_local: torch.Tensor,
        operand_edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """Optional first-class variable route; canonical is identity."""
        return state

    def _refine_block_state(
        self,
        state: torch.Tensor,
        block_function: torch.Tensor,
        function_count: int,
    ) -> torch.Tensor:
        """Optional within-function composition hook; canonical is identity."""
        return state

    def _encode_orderless(
        self,
        data,
        base: dict[str, torch.Tensor],
        edge_features: dict,
        static_instruction: torch.Tensor,
        instruction_block: torch.Tensor,
        block_function: torch.Tensor,
        block_pragma: torch.Tensor,
        function_pragma: torch.Tensor,
    ) -> torch.Tensor:
        """Capacity-matched control without structured upper-level composition.

        Instruction control/def-use propagation is intact. Reachable instruction,
        block, and function states are then treated as orderless bags: every block
        sees the same graph-wide instruction pool, block layers use self messages
        only, and every function sees the same graph-wide block pool. Containment,
        block-CFG, call edges, callee injection, and entry-function readout do not
        determine the representation. All modules in the structured encoder remain
        active so the control has the same parameter count and depth.
        """
        graph_count = int(getattr(data, "num_graphs", 1))
        reachable_functions = data["function"].is_reachable.bool()
        reachable_blocks = reachable_functions[block_function]
        reachable_instructions = reachable_blocks[instruction_block]

        instruction_ids = reachable_instructions.nonzero(as_tuple=False).flatten()
        instruction_local = torch.full(
            (data["instruction"].num_nodes,),
            -1,
            dtype=torch.long,
            device=instruction_ids.device,
        )
        instruction_local[instruction_ids] = torch.arange(
            instruction_ids.numel(), device=instruction_ids.device
        )
        control = ("instruction", "control", "instruction")
        control_edges = data[control].edge_index
        def_use_edges = data[DERIVED_DEF_USE_EDGE].edge_index
        control_mask = (
            reachable_instructions[control_edges[0]]
            & reachable_instructions[control_edges[1]]
        )
        def_use_mask = (
            reachable_instructions[def_use_edges[0]]
            & reachable_instructions[def_use_edges[1]]
        )
        instruction_state = self.instruction_input(
            static_instruction[instruction_ids]
        )
        for layer in self.instruction_layers:
            instruction_state = layer(
                instruction_state,
                instruction_local[control_edges[:, control_mask]],
                instruction_local[def_use_edges[:, def_use_mask]],
                edge_features[control][control_mask],
                use_messages=self.use_local_messages,
            )
        instruction_batch = self._batch(data, "instruction")[instruction_ids]
        instruction_pool = multi_pool(
            instruction_state, instruction_batch, graph_count
        )

        block_ids = reachable_blocks.nonzero(as_tuple=False).flatten()
        block_batch = self._batch(data, "block")[block_ids]
        block_state = self.block_input(
            torch.cat(
                [
                    base["block"][block_ids],
                    block_pragma[block_ids],
                    instruction_pool[block_batch],
                ],
                dim=-1,
            )
        )
        self_edges = torch.arange(block_ids.numel(), device=block_ids.device)
        self_edges = torch.stack([self_edges, self_edges])
        for layer in self.block_layers:
            block_state = layer(block_state, self_edges, use_messages=True)
        block_pool = multi_pool(block_state, block_batch, graph_count)

        function_ids = reachable_functions.nonzero(as_tuple=False).flatten()
        function_batch = self._batch(data, "function")[function_ids]
        function_state = self.function_input(
            torch.cat(
                [
                    base["function"][function_ids],
                    function_pragma[function_ids],
                    block_pool[function_batch],
                ],
                dim=-1,
            )
        )
        # Reuse the callee projection as a pointwise residual. This preserves
        # active capacity without consuming call edges or callee states.
        function_state = function_state + F.relu(self.callee_proj(function_state))
        function_pool = multi_pool(function_state, function_batch, graph_count)
        graph_state = self.root_readout(
            torch.cat([instruction_pool, function_pool], dim=-1)
        )
        features = [graph_state]
        if self.use_global_features:
            features.append(self.global_features(data))
        if self.use_context:
            features.append(self.context_encoder(data))
        return torch.cat(features, dim=-1)

    def encode(self, data):
        versions = torch.as_tensor(
            data.hierarchy_schema_version,
            device=data["instruction"].x.device,
        )
        if not bool(torch.all(versions == 2)):
            raise ValueError("Hierarchical tensor schema version 2 is required")
        x_dict = {node_type: data[node_type].x for node_type in NODE_TYPES}
        edge_attr_dict = {
            edge_type: data[edge_type].edge_attr.long()
            for edge_type in (
                ("instruction", "control", "instruction"),
                ("variable", "operand", "instruction"),
                ("constant", "operand", "instruction"),
            )
        }
        base, edge_features = self.input_proj(x_dict, edge_attr_dict)
        instruction_block, block_function = self._owners(data)
        call_depth = data["function"].call_depth.long()
        instruction_depth = data["instruction"].call_depth.long()
        block_depth = data["block"].call_depth.long()
        reachable_functions = data["function"].is_reachable.bool()

        pragma = base["pragma"]
        variable = base["variable"] + _messages(
            pragma,
            data[("pragma", "applies_to", "variable")].edge_index,
            data["variable"].num_nodes,
        )
        constant = base["constant"] + _messages(
            pragma,
            data[("pragma", "applies_to", "constant")].edge_index,
            data["constant"].num_nodes,
        )
        direct_pragma = _messages(
            pragma,
            data[("pragma", "applies_to", "instruction")].edge_index,
            data["instruction"].num_nodes,
        )
        variable_operand = ("variable", "operand", "instruction")
        constant_operand = ("constant", "operand", "instruction")
        static_instruction = (
            base["instruction"]
            + direct_pragma
            + _messages(
                variable,
                data[variable_operand].edge_index,
                data["instruction"].num_nodes,
                edge_features[variable_operand],
            )
            + _messages(
                constant,
                data[constant_operand].edge_index,
                data["instruction"].num_nodes,
                edge_features[constant_operand],
            )
            + _reverse_messages(
                variable,
                data[("instruction", "defines", "variable")].edge_index,
                data["instruction"].num_nodes,
            )
        )

        # Keep the small cross-depth state in FP32. LayerNorm is FP32 under
        # autocast, and callers need a stable summary of already-complete callees.
        function_state = torch.zeros_like(base["function"], dtype=torch.float32)
        calls = data[("instruction", "calls", "function")].edge_index
        control = ("instruction", "control", "instruction")
        block_cfg = data[("block", "control", "block")].edge_index
        control_edges = data[control].edge_index
        def_use_edges = data[DERIVED_DEF_USE_EDGE].edge_index
        block_pragma = _messages(
            pragma,
            data[("pragma", "applies_to", "block")].edge_index,
            data["block"].num_nodes,
        )
        function_pragma = _messages(
            pragma,
            data[("pragma", "applies_to", "function")].edge_index,
            data["function"].num_nodes,
        )

        if self.hierarchy_mode == "orderless":
            return self._encode_orderless(
                data,
                base,
                edge_features,
                static_instruction,
                instruction_block,
                block_function,
                block_pragma,
                function_pragma,
            )

        max_depth = int(call_depth[reachable_functions].max().item())
        for depth in range(max_depth + 1):
            function_ids = (
                (call_depth == depth) & reachable_functions
            ).nonzero(as_tuple=False).flatten()
            instruction_ids = (
                instruction_depth == depth
            ).nonzero(as_tuple=False).flatten()
            block_ids = (block_depth == depth).nonzero(as_tuple=False).flatten()

            instruction_local = torch.full(
                (data["instruction"].num_nodes,),
                -1,
                dtype=torch.long,
                device=instruction_ids.device,
            )
            instruction_local[instruction_ids] = torch.arange(
                instruction_ids.numel(), device=instruction_ids.device
            )
            block_local = torch.full(
                (data["block"].num_nodes,),
                -1,
                dtype=torch.long,
                device=block_ids.device,
            )
            block_local[block_ids] = torch.arange(
                block_ids.numel(), device=block_ids.device
            )
            function_local = torch.full(
                (data["function"].num_nodes,),
                -1,
                dtype=torch.long,
                device=function_ids.device,
            )
            function_local[function_ids] = torch.arange(
                function_ids.numel(), device=function_ids.device
            )

            call_mask = instruction_depth[calls[0]] == depth
            depth_calls = calls[:, call_mask]
            callee_source = self.callee_proj(function_state[depth_calls[1]])
            if self.use_callee_messages:
                callee_message = _messages(
                    callee_source,
                    torch.stack(
                        [
                            torch.arange(
                                depth_calls.size(1), device=depth_calls.device
                            ),
                            instruction_local[depth_calls[0]],
                        ]
                    ),
                    instruction_ids.numel(),
                )
            else:
                callee_message = (
                    callee_source.sum(dim=0, keepdim=True).expand(
                        instruction_ids.numel(), -1
                    )
                    * 0
                )
            instruction_state = self.instruction_input(
                static_instruction[instruction_ids] + callee_message
            )
            control_mask = instruction_depth[control_edges[0]] == depth
            local_control = instruction_local[control_edges[:, control_mask]]
            def_use_mask = instruction_depth[def_use_edges[0]] == depth
            local_def_use = instruction_local[def_use_edges[:, def_use_mask]]
            for layer in self.instruction_layers:
                instruction_state = layer(
                    instruction_state,
                    local_control,
                    local_def_use,
                    edge_features[control][control_mask],
                    use_messages=self.use_local_messages,
                )

            instruction_state = self._exchange_variable_state(
                instruction_state,
                variable,
                data,
                instruction_ids,
                instruction_local,
                edge_features[variable_operand],
            )
            instruction_state = self._refine_instruction_state(
                instruction_state,
                block_local[instruction_block[instruction_ids]],
                block_ids.numel(),
            )

            instruction_pool = multi_pool(
                instruction_state,
                block_local[instruction_block[instruction_ids]],
                block_ids.numel(),
            )
            current_block = self.block_input(
                torch.cat(
                    [
                        base["block"][block_ids],
                        block_pragma[block_ids],
                        instruction_pool,
                    ],
                    dim=-1,
                )
            )
            cfg_mask = block_depth[block_cfg[0]] == depth
            local_cfg = block_local[block_cfg[:, cfg_mask]]
            block_state = current_block
            for layer in self.block_layers:
                block_state = layer(
                    block_state,
                    local_cfg,
                    use_messages=self.use_block_messages,
                )

            block_state = self._refine_block_state(
                block_state,
                function_local[block_function[block_ids]],
                function_ids.numel(),
            )

            block_pool = multi_pool(
                block_state,
                function_local[block_function[block_ids]],
                function_ids.numel(),
            )
            current_function = self.function_input(
                torch.cat(
                    [
                        base["function"][function_ids],
                        function_pragma[function_ids],
                        block_pool,
                    ],
                    dim=-1,
                )
            )
            function_state = function_state.index_copy(
                0, function_ids, current_function
            )

        entries = data["function"].is_entry.bool()
        function_batch = getattr(data["function"], "batch", None)
        if function_batch is None:
            function_batch = torch.zeros(
                data["function"].num_nodes,
                dtype=torch.long,
                device=function_state.device,
            )
        entry_pool = multi_pool(
            function_state[entries], function_batch[entries], data.num_graphs
        )
        reachable_pool = multi_pool(
            function_state[reachable_functions],
            function_batch[reachable_functions],
            data.num_graphs,
        )
        graph_state = self.root_readout(
            torch.cat([entry_pool, reachable_pool], dim=-1)
        )
        features = [graph_state]
        if self.use_global_features:
            features.append(self.global_features(data))
        if self.use_context:
            features.append(self.context_encoder(data))
        return torch.cat(features, dim=-1)

    def forward(self, data):
        if self.classifier is None:
            raise RuntimeError("This hierarchical encoder has no prediction head")
        return self.classifier(self.encode(data))
