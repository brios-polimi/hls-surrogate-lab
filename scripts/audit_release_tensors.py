#!/usr/bin/env python3
"""Load and validate every tensor in a frozen release."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {args.output}")
    release = json.loads(args.release.read_text())
    schemas = Counter()
    feature_dims: dict[str, set[int]] = defaultdict(set)
    family_counts = Counter()
    missing_graphs = []
    for index, row in enumerate(release["samples"], start=1):
        path = args.tensor_root / row["tensor_path"]
        graph = torch.load(path, map_location="cpu", weights_only=False)
        if graph.y.view(-1).tolist() != row["labels"]:
            raise ValueError(f"Tensor/release labels disagree for {row['tensor_path']}")
        schemas[str(getattr(graph, "hierarchy_schema_version", None))] += 1
        for node_type in graph.node_types:
            if hasattr(graph[node_type], "x") and graph[node_type].x.ndim == 2:
                feature_dims[node_type].add(int(graph[node_type].x.shape[1]))
        if args.graph_root is not None and not (
            args.graph_root / row["graph_path"]
        ).is_file():
            missing_graphs.append(row["graph_path"])
        family_counts[row["kernel_family"]] += 1
        if index % 500 == 0 or index == len(release["samples"]):
            print(f"Loaded {index}/{len(release['samples'])} tensors", flush=True)
    if len(schemas) != 1:
        raise ValueError(f"Mixed hierarchy schemas: {schemas}")
    heterogeneous_dims = {
        node_type: sorted(values) for node_type, values in feature_dims.items()
        if len(values) != 1
    }
    if heterogeneous_dims:
        raise ValueError(f"Mixed feature dimensions: {heterogeneous_dims}")
    if missing_graphs:
        raise FileNotFoundError(
            f"Missing {len(missing_graphs)} graphs; first: {missing_graphs[:5]}"
        )
    report = {
        "release_id": release["release_id"],
        "loaded_tensors": len(release["samples"]),
        "hierarchy_schemas": dict(schemas),
        "feature_dimensions": {
            node_type: next(iter(values))
            for node_type, values in sorted(feature_dims.items())
        },
        "family_counts": dict(sorted(family_counts.items())),
        "missing_graphs": 0,
        "status": "passed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
