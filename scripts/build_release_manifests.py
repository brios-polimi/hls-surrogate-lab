#!/usr/bin/env python3
"""Build immutable E0 release and protocol manifests without training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from ll_hls4ml.data.protocols import (
    architecture_grouped_protocol,
    audit_protocol,
    leave_one_family_out_protocols,
    official_protocol,
)
from ll_hls4ml.data.signatures import coarse_architecture_fields, signature_fields
from ll_hls4ml.io.schema import LABEL_KEYS


DEFAULT_FAMILIES = (
    "2layer", "3layer", "conv1d", "conv2d", "dense_latency",
    "dense_resource", "rule4ml",
)


def _archive_number(path: str) -> int:
    return int(Path(path).parts[1].removeprefix("archive_"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def build_release(
    tensor_root: Path, archives: int, families: tuple[str, ...], release_id: str,
    *, require_tensors: bool = True, metadata_by_graph_id: dict | None = None,
    high_level_samples: dict | None = None,
) -> dict:
    index_path = tensor_root / "labels.json"
    index = json.loads(index_path.read_text())
    if index.get("label_keys") != LABEL_KEYS:
        raise ValueError("Tensor label order does not match the canonical six targets")
    labels = index["labels"]
    metadata = index.get("metadata", {})
    selected = [
        path for path in labels
        if (
            (Path(path).parts[0] in families and _archive_number(path) <= archives)
            or Path(path).parts[0] == "exemplar"
        )
    ]
    by_graph_id = {}
    duplicates = []
    samples = []
    for path in sorted(selected):
        graph_id = Path(path).stem
        if graph_id in by_graph_id:
            duplicates.append({"project_uuid": graph_id, "kept": by_graph_id[graph_id], "dropped": path})
            continue
        by_graph_id[graph_id] = path
        tensor_path = tensor_root / path
        if require_tensors and not tensor_path.is_file():
            raise FileNotFoundError(tensor_path)
        family = Path(path).parts[0]
        sample_metadata = dict(metadata.get(path, {}))
        if metadata_by_graph_id:
            sample_metadata.update(metadata_by_graph_id.get(graph_id, {}))
        signatures = signature_fields(family, sample_metadata.get("model_name"))
        # Prefer signatures persisted at tensorization time, but make any fallback explicit.
        signatures.update({
            key: sample_metadata[key] for key in signatures if key in sample_metadata
        })
        signatures["model_variant_id"] = signatures["architecture_id"]
        signatures["model_variant_summary"] = signatures["architecture_summary"]
        layer_count = None
        if high_level_samples and path in high_level_samples:
            layer_count = int(high_level_samples[path]["features"].shape[0])
        signatures.update(coarse_architecture_fields(family, layer_count))
        samples.append({
            "project_uuid": graph_id,
            "kernel_family": family,
            "archive": Path(path).parts[1],
            "original_dataset_split": str(sample_metadata.get("dataset_split", "exemplar" if family == "exemplar" else "")).lower(),
            "tensor_path": path,
            "graph_path": str(Path(path).with_suffix(".json")),
            "labels": labels[path],
            "label_validity_mask": [value is not None for value in labels[path]],
            "backend": sample_metadata.get("backend", ""),
            "toolchain_version": sample_metadata.get("vivado_version", ""),
            "target_part": sample_metadata.get("target_part", ""),
            "target_clock": sample_metadata.get("target_clock"),
            "hls4ml_version": sample_metadata.get("hls4ml_version", ""),
            "high_level_cache_membership": bool(
                high_level_samples and path in high_level_samples
            ),
            **signatures,
        })
    return {
        "release_id": release_id,
        "archive_policy": {"synthetic_archives": list(range(1, archives + 1)), "exemplar": "all indexed"},
        "families": list(families),
        "label_keys": LABEL_KEYS,
        "tensor_index_sha256": _sha256(index_path),
        "vocabulary_sha256": _sha256(tensor_root / "vocab.json"),
        "producer_commit": _git_commit(),
        "graph_schema": "canonical_cdfg",
        "tensor_feature_schema": "hierarchical_schema_v2",
        "duplicates_removed": duplicates,
        "samples": samples,
    }


def _write_new(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-root", type=Path, default=Path("../data/tensors"))
    parser.add_argument("--archives", type=int, default=31)
    parser.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES))
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--metadata-index", type=Path,
        help="Optional graph-ID metadata overlay containing canonical signatures",
    )
    parser.add_argument(
        "--high-level-cache", type=Path,
        help="Cache used for coarse layer-count architecture groups and exact release subsetting",
    )
    args = parser.parse_args()
    metadata_overlay = (
        json.loads(args.metadata_index.read_text()) if args.metadata_index else None
    )
    source_high_level_cache = (
        __import__("torch").load(args.high_level_cache, weights_only=False)
        if args.high_level_cache else None
    )
    release = build_release(
        args.tensor_root.resolve(), args.archives, tuple(args.families), args.release_id,
        metadata_by_graph_id=metadata_overlay,
        high_level_samples=(source_high_level_cache or {}).get("samples"),
    )
    output = args.output_dir.resolve()
    _write_new(output / "release.json", release)
    samples = release["samples"]
    official = official_protocol(samples)
    _write_new(output / "official.json", official)
    if source_high_level_cache is not None:
        import torch

        selected_paths = {
            row["tensor_path"] for rows in official.values() for row in rows
        }
        missing_cache = sorted(selected_paths - set(source_high_level_cache["samples"]))
        if missing_cache:
            raise KeyError(f"High-level cache misses release paths: {missing_cache[:5]}")
        exact_cache = dict(source_high_level_cache)
        exact_cache["manifest_path"] = str(output / "official.json")
        exact_cache["release_id"] = args.release_id
        exact_cache["samples"] = {
            path: source_high_level_cache["samples"][path] for path in sorted(selected_paths)
        }
        torch.save(exact_cache, output / "high_level_cache.pt")
    audits = {"official": audit_protocol(official)}
    try:
        grouped = architecture_grouped_protocol(samples, seed=args.seed)
    except ValueError as error:
        audits["architecture_grouped"] = {"status": "blocked", "reason": str(error)}
    else:
        _write_new(output / "architecture_grouped.json", grouped)
        audits["architecture_grouped"] = audit_protocol(grouped, group_key="architecture_id")
    folds = leave_one_family_out_protocols(samples, seed=args.seed)
    for family, manifest in folds.items():
        _write_new(output / f"leave_{family}_out.json", manifest)
        audits[f"leave_{family}_out"] = audit_protocol(manifest)
    fallback_count = sum(
        row["architecture_source"] == "family_fallback" for row in samples
        if row["kernel_family"] != "exemplar"
    )
    audits["release"] = {
        "samples": len(samples),
        "architecture_family_fallback_samples": fallback_count,
        "duplicates_removed": len(release["duplicates_removed"]),
    }
    _write_new(output / "audit.json", audits)
    print(json.dumps(audits, indent=2))


if __name__ == "__main__":
    main()
