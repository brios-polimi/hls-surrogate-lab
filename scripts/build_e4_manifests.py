#!/usr/bin/env python3
"""Build E4 leave-one-family-out manifests from the frozen exact E2 manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from ll_hls4ml.data.protocols import (
    SPLIT_NAMES,
    audit_protocol,
    leave_one_family_out_protocols,
)
from ll_hls4ml.data.signatures import (
    HIGH_LEVEL_ARCHITECTURE_SIGNATURE_VERSION,
    HIGH_LEVEL_TOPOLOGY_SIGNATURE_VERSION,
)
from ll_hls4ml.reporting.accounting import split_sha256


STUDY_ID = "e4_leave_one_family_out_exact_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_stable(path: Path, value: object) -> None:
    content = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != content:
            raise FileExistsError(f"Refusing to replace different artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _source_rows(source: dict[str, list[dict]]) -> list[dict]:
    if set(source) != set(SPLIT_NAMES):
        raise ValueError(f"Expected exactly these source splits: {SPLIT_NAMES}")
    rows = [row for split in SPLIT_NAMES for row in source[split]]
    paths = [row["tensor_path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("Source manifest contains duplicate tensor paths")
    architecture_versions = {
        row.get("architecture_signature_version") for row in rows
    }
    if architecture_versions != {HIGH_LEVEL_ARCHITECTURE_SIGNATURE_VERSION}:
        raise ValueError(
            "E4 requires the exact ordered-layer architecture signature; got "
            f"{sorted(map(str, architecture_versions))}"
        )
    topology_versions = {row.get("topology_signature_version") for row in rows}
    if topology_versions != {HIGH_LEVEL_TOPOLOGY_SIGNATURE_VERSION}:
        raise ValueError(
            "E4 requires the exact ordered-layer topology signature; got "
            f"{sorted(map(str, topology_versions))}"
        )
    return rows


def build_folds(
    source: dict[str, list[dict]], *, seed: int = 42,
    val_fraction: float = 0.15,
) -> tuple[dict[str, dict[str, list[dict]]], dict]:
    """Construct and fully audit the E4 folds without touching tensor files."""
    rows = _source_rows(source)
    source_paths = {row["tensor_path"] for row in rows}
    folds = leave_one_family_out_protocols(
        rows, seed=seed, val_fraction=val_fraction
    )
    audits = {}
    for held_out, manifest in folds.items():
        actual_paths = {
            row["tensor_path"] for split in SPLIT_NAMES for row in manifest[split]
        }
        if actual_paths != source_paths:
            raise ValueError(f"{held_out}: fold does not preserve the source universe")
        if {row["kernel_family"] for row in manifest["test"]} != {held_out}:
            raise ValueError(f"{held_out}: test split is not exactly the held-out family")
        for split in ("train", "validation"):
            if any(row["kernel_family"] == held_out for row in manifest[split]):
                raise ValueError(f"{held_out}: held-out family leaked into {split}")
        audit = audit_protocol(manifest, group_key="architecture_id")
        audit.update({
            "held_out_family": held_out,
            "protocol_id": f"leave_{held_out}_out_exact_v1",
            "split_sha256": split_sha256(manifest),
        })
        audits[held_out] = audit

    main = [row for row in rows if row["kernel_family"] != "exemplar"]
    source_summary = {
        "samples": len(rows),
        "families": dict(Counter(row["kernel_family"] for row in rows)),
        "architecture_groups_by_family": {
            family: len({
                row["architecture_id"]
                for row in main if row["kernel_family"] == family
            })
            for family in sorted(folds)
        },
        "architecture_signature_version": (
            HIGH_LEVEL_ARCHITECTURE_SIGNATURE_VERSION
        ),
        "topology_signature_version": HIGH_LEVEL_TOPOLOGY_SIGNATURE_VERSION,
        "source_split_sha256": split_sha256(source),
    }
    return folds, {"source": source_summary, "folds": audits}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--families", nargs="+",
        help="Optional subset to materialize; the audit still validates all folds",
    )
    args = parser.parse_args()

    source_path = args.source_manifest.resolve()
    source = json.loads(source_path.read_text())
    folds, audit = build_folds(
        source, seed=args.seed, val_fraction=args.val_fraction
    )
    selected = sorted(folds) if args.families is None else args.families
    unknown = sorted(set(selected) - set(folds))
    if unknown:
        raise ValueError(f"Unknown families: {unknown}")

    output = args.output_dir.resolve()
    for family in selected:
        _write_stable(output / f"leave_{family}_out.json", folds[family])
    audit = {
        "study_id": STUDY_ID,
        "seed": args.seed,
        "validation_fraction": args.val_fraction,
        "source_manifest": str(source_path),
        "source_manifest_sha256": _sha256(source_path),
        "materialized_families": selected,
        **audit,
    }
    _write_stable(output / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
