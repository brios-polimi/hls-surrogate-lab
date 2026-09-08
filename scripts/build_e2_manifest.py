#!/usr/bin/env python3
"""Build the thesis E2 structural-signature manifest from frozen E0 inputs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from ll_hls4ml.data.protocols import architecture_grouped_protocol, audit_protocol
from ll_hls4ml.data.signatures import (
    HIGH_LEVEL_ARCHITECTURE_SIGNATURE_VERSION,
    HIGH_LEVEL_TOPOLOGY_SIGNATURE_VERSION,
    high_level_signature_fields,
)
from ll_hls4ml.io.schema import LABEL_KEYS
from ll_hls4ml.reporting.accounting import split_sha256


FAMILIES = (
    "2layer", "3layer", "conv1d", "conv2d", "dense_latency",
    "dense_resource", "rule4ml",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_number(path: str) -> int:
    return int(Path(path).parts[1].removeprefix("archive_"))


def _write_stable(path: Path, value) -> None:
    content = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != content:
            raise FileExistsError(f"Refusing to replace different artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _signature_audit(rows: list[dict], key: str) -> dict:
    output = {}
    for family in FAMILIES:
        counts = Counter(row[key] for row in rows if row["kernel_family"] == family)
        output[family] = {
            "samples": sum(counts.values()),
            "groups": len(counts),
            "singleton_groups": sum(value == 1 for value in counts.values()),
            "singleton_group_fraction": (
                sum(value == 1 for value in counts.values()) / len(counts)
            ),
            "samples_in_singleton_groups": sum(
                value for value in counts.values() if value == 1
            ),
            "maximum_group_size": max(counts.values()),
            "median_group_size": statistics.median(counts.values()),
        }
    return output


def build_rows(index: dict, cache: dict, archives: int) -> tuple[list[dict], list[dict]]:
    if index.get("label_keys") != LABEL_KEYS:
        raise ValueError("Tensor index does not use the canonical six targets")
    feature_columns = tuple(cache.get("feature_columns", ()))
    samples = cache.get("samples")
    if not isinstance(samples, dict):
        raise ValueError("High-level cache has no samples dictionary")
    labels = index["labels"]
    metadata = index.get("metadata", {})
    selected = sorted(
        path for path in labels
        if (
            Path(path).parts[0] == "exemplar"
            or (
                Path(path).parts[0] in FAMILIES
                and _archive_number(path) <= archives
            )
        )
    )
    kept_by_id = {}
    duplicates = []
    unique_selected = []
    for path in selected:
        project_uuid = Path(path).stem
        if project_uuid in kept_by_id:
            duplicates.append({
                "project_uuid": project_uuid,
                "kept": kept_by_id[project_uuid],
                "dropped": path,
            })
            continue
        kept_by_id[project_uuid] = path
        unique_selected.append(path)
    missing = sorted(set(unique_selected) - set(samples))
    if missing:
        raise KeyError(f"High-level cache misses selected paths: {missing[:5]}")

    rows = []
    for path in unique_selected:
        project_uuid = Path(path).stem
        family = Path(path).parts[0]
        cached = samples[path]
        target = torch.as_tensor(cached["target"]).float()
        indexed_target = torch.as_tensor(labels[path]).float()
        if not torch.equal(target, indexed_target):
            raise ValueError(f"Cache/index target mismatch for {path}")
        signature = high_level_signature_fields(
            family, cached["features"], feature_columns
        )
        source = metadata.get(path, {})
        original_split = str(
            source.get("dataset_split", "exemplar" if family == "exemplar" else "")
        ).lower()
        rows.append({
            "project_uuid": project_uuid,
            "kernel_family": family,
            "archive": Path(path).parts[1],
            "original_dataset_split": original_split,
            "tensor_path": path,
            "labels": [float(value) for value in indexed_target.tolist()],
            "label_validity_mask": [True] * len(LABEL_KEYS),
            **signature,
        })
    return rows, duplicates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-index", type=Path, required=True)
    parser.add_argument("--high-level-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archives", type=int, default=31)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    index = json.loads(args.tensor_index.read_text())
    cache = torch.load(args.high_level_cache, map_location="cpu", weights_only=False)
    rows, duplicates = build_rows(index, cache, args.archives)
    manifest = architecture_grouped_protocol(rows, seed=args.seed)
    protocol_audit = audit_protocol(manifest, group_key="architecture_id")
    main_rows = [row for row in rows if row["kernel_family"] != "exemplar"]
    audit = {
        "study_id": "e2_architecture_grouped_structural_v1",
        "protocol_id": "architecture_grouped_structural_v1",
        "seed": args.seed,
        "archives": args.archives,
        "signature_definition": {
            "architecture_version": HIGH_LEVEL_ARCHITECTURE_SIGNATURE_VERSION,
            "topology_version": HIGH_LEVEL_TOPOLOGY_SIGNATURE_VERSION,
            "architecture_includes": [
                "ordered layer/operator classes", "directed chain connectivity",
                "activations", "input/output tensor dimensions", "filters",
                "kernel sizes", "strides", "padding", "pooling", "batchnorm",
            ],
            "explicitly_excludes": [
                "precision", "reuse factor", "strategy", "I/O mode", "pragmas",
                "clock", "part", "backend", "tool versions", "labels",
            ],
        },
        "source": {
            "tensor_index_sha256": _sha256(args.tensor_index),
            "high_level_cache_sha256": _sha256(args.high_level_cache),
        },
        "selected_samples": len(rows),
        "duplicates_removed": duplicates,
        "architecture_signatures": _signature_audit(main_rows, "architecture_id"),
        "topology_signatures": _signature_audit(main_rows, "topology_id"),
        "protocol": protocol_audit,
        "split_sha256": split_sha256(manifest),
    }
    output = args.output_dir.resolve()
    _write_stable(output / "architecture_grouped_structural_v1.json", manifest)
    _write_stable(output / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
