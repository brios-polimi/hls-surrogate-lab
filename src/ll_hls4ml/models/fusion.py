"""Late fusion of hierarchical LLVM and wa-hls4ml layer-graph encoders."""

from __future__ import annotations

import torch
import torch.nn as nn

from ll_hls4ml.models.high_level import HighLevelLayerEncoder
from ll_hls4ml.models.hierarchical import CDFGHierarchical
from ll_hls4ml.models.readout import SplitRegressionHead


class HierarchicalHighLevelFusion(nn.Module):
    """Encode both representations independently, then share prediction heads."""

    def __init__(
        self,
        instruction_vocab_size: int,
        edge_pos_vocab_size: int,
        high_level_input_dim: int,
        y_means: torch.Tensor,
        y_stds: torch.Tensor,
        hidden_dim: int = 64,
        num_layers: int = 3,
        heads: int = 1,
        dropout: float = 0.15,
        high_level_encoder: str = "gatv2",
        use_global_features: bool = True,
        use_context: bool = True,
        context_mode: str = "core",
        split_heads: bool = True,
        hurdle_heads: bool = True,
        hurdle_prediction_mode: str = "threshold",
    ):
        super().__init__()
        if not split_heads:
            raise ValueError("late fusion requires split_heads=True")
        self.register_buffer("y_means", y_means.clone())
        self.register_buffer("y_stds", y_stds.clone())
        self.hurdle_prediction_mode = hurdle_prediction_mode
        self.cdfg_encoder = CDFGHierarchical(
            instruction_vocab_size=instruction_vocab_size,
            edge_pos_vocab_size=edge_pos_vocab_size,
            y_means=y_means,
            y_stds=y_stds,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_global_features=use_global_features,
            use_context=use_context,
            context_mode=context_mode,
            build_head=False,
        )
        self.high_level_encoder = HighLevelLayerEncoder(
            input_dim=high_level_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            encoder=high_level_encoder,
        )
        self.classifier = SplitRegressionHead(
            self.cdfg_encoder.output_dim + self.high_level_encoder.output_dim,
            hidden_dim,
            dropout,
            hurdle_heads=hurdle_heads,
        )

    def encode(self, data):
        """Return the fused representation before the prediction heads."""
        cdfg = self.cdfg_encoder.encode(data)
        layer_store = data["high_level_layer"]
        high_level = self.high_level_encoder(
            layer_store.x,
            data[
                ("high_level_layer", "next", "high_level_layer")
            ].edge_index,
            layer_store.batch,
            data.num_graphs,
            data.high_level_strategy,
            data.high_level_io_type,
        )
        return torch.cat([cdfg, high_level], dim=-1)

    def forward(self, data):
        return self.classifier(self.encode(data))
