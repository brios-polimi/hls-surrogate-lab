"""Stable, label-free architecture identities for wa-hls4ml samples."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


TOPOLOGY_SIGNATURE_VERSION = "wa_hls4ml_family_topology_v1"
ARCHITECTURE_SIGNATURE_VERSION = "wa_hls4ml_model_name_v1"
COARSE_ARCHITECTURE_SIGNATURE_VERSION = "wa_hls4ml_family_layer_count_v1"
HIGH_LEVEL_TOPOLOGY_SIGNATURE_VERSION = "wa_hls4ml_ordered_layer_dag_v1"
HIGH_LEVEL_ARCHITECTURE_SIGNATURE_VERSION = "wa_hls4ml_ordered_layer_structure_v1"

# The cache stores layers in directed execution order and the fusion graph uses
# i -> i+1 connectivity. These fields describe computation, not synthesis.
_TOPOLOGY_COLUMNS = (
    "layer_type", "activation_type", "padding", "pooling", "batchnorm",
)
_ARCHITECTURE_COLUMNS = (
    "d_in1", "d_in2", "d_in3", "d_out1", "d_out2", "d_out3",
    *_TOPOLOGY_COLUMNS, "filters", "kernel_size", "stride",
)
_EXCLUDED_SYNTHESIS_COLUMNS = {"prec", "rf", "strategy", "io_type"}


def _digest(version: str, value: str) -> str:
    return hashlib.sha256(f"{version}\0{value}".encode()).hexdigest()[:24]


def canonical_family(value: str) -> str:
    """Normalize source-file and tensor family spellings."""
    family = Path(str(value)).stem.lower()
    for prefix in ("train_", "val_", "test_"):
        family = family.removeprefix(prefix)
    family = family.removesuffix("_merged")
    return {
        "2_20": "rule4ml",
        "latency": "dense_latency",
        "resource": "dense_resource",
    }.get(family, family)


def canonical_model_name(model_name: str, family: str) -> str | None:
    """Remove synthesis knobs from a wa-hls4ml model name.

    Synthetic model names encode the structural dimensions we want to retain.
    Precision, reuse factor, and strategy are deliberately excluded.
    """
    name = Path(str(model_name).rstrip("/")).name
    if not name:
        return None
    name = re.sub(r"_ap_(?:u?fixed|u?int)<[^>]+>", "", name)
    name = re.sub(r"_\d+rf(?:_[A-Za-z]+)?$", "", name)
    name = re.sub(r"_\d+b$", "", name)
    name = re.sub(r"_(?:latency|resource)$", "", name, flags=re.IGNORECASE)
    return f"{canonical_family(family)}:{name}"


def signature_fields(family: str, model_name: str | None = None) -> dict[str, str]:
    """Return auditable topology and architecture signature fields."""
    family = canonical_family(family)
    architecture = canonical_model_name(model_name or "", family)
    source = "model_name" if architecture else "family_fallback"
    architecture = architecture or family
    return {
        "topology_id": _digest(TOPOLOGY_SIGNATURE_VERSION, family),
        "topology_summary": family,
        "topology_signature_version": TOPOLOGY_SIGNATURE_VERSION,
        "architecture_id": _digest(ARCHITECTURE_SIGNATURE_VERSION, architecture),
        "architecture_summary": architecture,
        "architecture_signature_version": ARCHITECTURE_SIGNATURE_VERSION,
        "architecture_source": source,
    }


def coarse_architecture_fields(family: str, layer_count: int | None) -> dict[str, str]:
    """Use family topology, subdividing genuinely variable families by depth."""
    family = canonical_family(family)
    fixed_topology = family in {"2layer", "3layer"}
    if fixed_topology:
        summary = family
        source = "fixed_family_topology"
    elif layer_count is not None:
        summary = f"{family}:layers={int(layer_count)}"
        source = "high_level_layer_count"
    else:
        summary = family
        source = "family_fallback"
    return {
        "architecture_id": _digest(COARSE_ARCHITECTURE_SIGNATURE_VERSION, summary),
        "architecture_summary": summary,
        "architecture_signature_version": COARSE_ARCHITECTURE_SIGNATURE_VERSION,
        "architecture_source": source,
    }


def _canonical_number(value: float) -> int | float:
    value = float(value)
    integer = int(value)
    return integer if value == integer else value


def _ordered_layer_payload(features, feature_columns, selected_columns, family):
    columns = tuple(feature_columns)
    if len(columns) != len(set(columns)):
        raise ValueError("High-level feature columns are not unique")
    missing = sorted(set(selected_columns) - set(columns))
    if missing:
        raise ValueError(f"High-level cache misses signature columns: {missing}")
    if set(selected_columns) & _EXCLUDED_SYNTHESIS_COLUMNS:
        raise ValueError("A synthesis-only field entered the architecture signature")
    positions = [columns.index(name) for name in selected_columns]
    rows = features.tolist() if hasattr(features, "tolist") else features
    layers = [
        {
            name: _canonical_number(row[position])
            for name, position in zip(selected_columns, positions, strict=True)
        }
        for row in rows
    ]
    # The cached high-level graph is an ordered chain. Serialize connectivity
    # explicitly so a later cache with non-chain connectivity needs a new version.
    edges = [[index, index + 1] for index in range(max(0, len(layers) - 1))]
    return {
        "family": canonical_family(family),
        "directed_connectivity": edges,
        "layers": layers,
    }


def high_level_signature_fields(
    family: str, features, feature_columns,
) -> dict[str, str]:
    """Canonical topology/architecture IDs from ordered high-level layers.

    Architecture retains operator order, directed chain connectivity, tensor
    dimensions, filters, kernels, strides, padding, pooling, and batchnorm.
    Precision, reuse factor, strategy, I/O mode, tools, clocks, and labels are
    deliberately absent.
    """
    family = canonical_family(family)
    topology = _ordered_layer_payload(
        features, feature_columns, _TOPOLOGY_COLUMNS, family
    )
    architecture = _ordered_layer_payload(
        features, feature_columns, _ARCHITECTURE_COLUMNS, family
    )
    topology_json = json.dumps(topology, sort_keys=True, separators=(",", ":"))
    architecture_json = json.dumps(
        architecture, sort_keys=True, separators=(",", ":")
    )
    operator_sequence = ",".join(
        str(layer["layer_type"]) for layer in topology["layers"]
    )
    shape_sequence = ";".join(
        "x".join(str(layer[name]) for name in (
            "d_in1", "d_in2", "d_in3", "d_out1", "d_out2", "d_out3"
        ))
        for layer in architecture["layers"]
    )
    return {
        "topology_id": _digest(
            HIGH_LEVEL_TOPOLOGY_SIGNATURE_VERSION, topology_json
        ),
        "topology_summary": (
            f"{family}:layers={len(topology['layers'])}:ops={operator_sequence}"
        ),
        "topology_signature_version": HIGH_LEVEL_TOPOLOGY_SIGNATURE_VERSION,
        "architecture_id": _digest(
            HIGH_LEVEL_ARCHITECTURE_SIGNATURE_VERSION, architecture_json
        ),
        "architecture_summary": (
            f"{family}:layers={len(architecture['layers'])}:ops={operator_sequence}:"
            f"shapes={shape_sequence}"
        ),
        "architecture_signature_version": (
            HIGH_LEVEL_ARCHITECTURE_SIGNATURE_VERSION
        ),
        "architecture_source": "ordered_high_level_layer_structure",
    }
