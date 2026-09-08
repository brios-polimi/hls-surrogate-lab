#!/usr/bin/env python3
"""Audit E2 topology destruction on real frozen tensor graphs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from ll_hls4ml.data.augmentations.topology import (
    _CORRUPTED_EDGES,
    _owners,
    PermuteDestinationsWithinFunction,
    TOPOLOGY_DESTINATION_PERMUTE_VERSION,
)


def _tensor_attributes(store) -> dict[str, torch.Tensor]:
    return {
        key: value for key, value in store.items()
        if isinstance(value, torch.Tensor)
    }


def _same_multiset(left: torch.Tensor, right: torch.Tensor) -> bool:
    return torch.equal(torch.sort(left.cpu()).values, torch.sort(right.cpu()).values)


def audit_graph(graph, transformed) -> dict[str, int]:
    if graph.node_types != transformed.node_types or graph.edge_types != transformed.edge_types:
        raise AssertionError("Topology transform changed the heterogeneous schema")
    for node_type in graph.node_types:
        before = _tensor_attributes(graph[node_type])
        after = _tensor_attributes(transformed[node_type])
        if before.keys() != after.keys():
            raise AssertionError(f"Node attributes changed for {node_type}")
        for key in before:
            if not torch.equal(before[key], after[key]):
                raise AssertionError(f"Node tensor changed: {node_type}.{key}")

    owners = _owners(graph)
    eligible = changed = corrupted_edges = 0
    for edge_type in graph.edge_types:
        before = graph[edge_type].edge_index.cpu()
        after = transformed[edge_type].edge_index.cpu()
        if edge_type not in _CORRUPTED_EDGES:
            if not torch.equal(before, after):
                raise AssertionError(f"Protected relation changed: {edge_type}")
            continue
        corrupted_edges += before.shape[1]
        if not torch.equal(before[0], after[0]):
            raise AssertionError(f"Source list changed: {edge_type}")
        if not _same_multiset(before[1], after[1]):
            raise AssertionError(f"Destination multiset changed: {edge_type}")
        source_owner = owners[edge_type[0]][before[0]]
        old_destination_owner = owners[edge_type[2]][before[1]]
        new_destination_owner = owners[edge_type[2]][after[1]]
        intra = source_owner == old_destination_owner
        cross = ~intra
        if not torch.equal(before[1, cross], after[1, cross]):
            raise AssertionError(f"Cross-function edge changed: {edge_type}")
        if not torch.equal(old_destination_owner[intra], new_destination_owner[intra]):
            raise AssertionError(f"Destination escaped its function: {edge_type}")
        eligible += int(intra.sum())
        changed += int((before[1, intra] != after[1, intra]).sum())
    return {
        "corrupted_relation_edges": corrupted_edges,
        "eligible_intra_function_edges": eligible,
        "changed_destinations": changed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tensor-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-family", type=int, default=8)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    counts = Counter()
    selected = []
    for row in manifest["train"]:
        family = row["kernel_family"]
        if counts[family] < args.per_family:
            selected.append(row)
            counts[family] += 1
    transform = PermuteDestinationsWithinFunction(seed=args.seed)
    totals = Counter()
    graph_rows = []
    for row in selected:
        path = args.tensor_dir / row["tensor_path"]
        graph = torch.load(path, map_location="cpu", weights_only=False)
        graph.graph_id = path.stem
        first = transform(graph)
        second = transform(graph)
        for edge_type in graph.edge_types:
            if not torch.equal(first[edge_type].edge_index, second[edge_type].edge_index):
                raise AssertionError(f"Transform is not deterministic: {path}")
        result = audit_graph(graph, first)
        totals.update(result)
        graph_rows.append({
            "tensor_path": row["tensor_path"],
            "kernel_family": row["kernel_family"],
            **result,
        })
    if set(counts) != {row["kernel_family"] for row in manifest["train"]}:
        raise AssertionError("Audit failed to cover every training family")
    if totals["eligible_intra_function_edges"] <= 0:
        raise AssertionError("No eligible topology was found")
    if totals["changed_destinations"] <= 0:
        raise AssertionError("Topology transform did not alter any destination")
    report = {
        "transform_version": TOPOLOGY_DESTINATION_PERMUTE_VERSION,
        "seed": args.seed,
        "graphs_audited": len(selected),
        "graphs_by_family": dict(sorted(counts.items())),
        **dict(totals),
        "changed_fraction_of_eligible": (
            totals["changed_destinations"]
            / totals["eligible_intra_function_edges"]
        ),
        "invariants": {
            "node_features_unchanged": True,
            "membership_and_call_relations_unchanged": True,
            "source_lists_unchanged": True,
            "destination_multisets_unchanged": True,
            "cross_function_edges_unchanged": True,
            "deterministic_per_graph_and_seed": True,
        },
        "graphs": graph_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "graphs"}, indent=2))


if __name__ == "__main__":
    main()
