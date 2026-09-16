#!/usr/bin/env python3
"""Run the mandatory E2 replications as independent local processes."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from ll_hls4ml.data.augmentations.topology import (
    TOPOLOGY_DESTINATION_PERMUTE_VERSION,
)
from ll_hls4ml.data.protocols import SPLIT_NAMES
from ll_hls4ml.data.signatures import (
    HIGH_LEVEL_ARCHITECTURE_SIGNATURE_VERSION,
    HIGH_LEVEL_TOPOLOGY_SIGNATURE_VERSION,
)
from ll_hls4ml.reporting.accounting import split_sha256


STUDY_ID = "e2_architecture_grouped_structural_v1"
PROTOCOL_ID = "architecture_grouped_structural_v1"
OFFICIAL_SPLIT_SHA256 = (
    "7d3fb85fd8664e5b2ea01b627e0f8401426098255a6c8c693f505e163160f2ca"
)
OFFICIAL_VOCAB_SHA256 = (
    "2a40117368c2536372ffdd0109198bb787227cb0ef9e7da9ac9c1b41d4f1b2bb"
)
OFFICIAL_HIGH_LEVEL_SHA256 = (
    "1fc6a8392ee583b25e8f8a5e16ffe81ed1e753a5391c3e89a843aa60e733b078"
)
OFFICIAL_TENSOR_REVISION = "1999fbea0187d4cac859f37dd20488fd63eb8860"
FAMILIES = {
    "2layer", "3layer", "conv1d", "conv2d", "dense_latency",
    "dense_resource", "rule4ml",
}
DEFAULT_MODELS = (
    "fusion", "topology_destroyed", "h0", "no_local_message", "extra_trees",
)
HIERARCHY_ABLATIONS = (
    "hierarchy_orderless", "no_block_cfg", "no_callee",
)
MODEL_ORDER = (*DEFAULT_MODELS, *HIERARCHY_ABLATIONS)
NEURAL_MODELS = {
    "h0": ("hierarchical", "e2_h0_structural_seed{seed}"),
    "topology_destroyed": (
        "hierarchical_topology_destroyed",
        "e2_topology_destroyed_structural_seed{seed}",
    ),
    "fusion": (
        "hierarchical_high_level_fusion",
        "e2_fusion_structural_seed{seed}",
    ),
    "no_local_message": (
        "hierarchical_no_local_message",
        "e2_no_local_message_structural_seed{seed}",
    ),
    "hierarchy_orderless": (
        "hierarchical_orderless",
        "e2_hierarchy_orderless_seed{seed}",
    ),
    "no_block_cfg": (
        "hierarchical_no_block_cfg",
        "e2_no_block_cfg_seed{seed}",
    ),
    "no_callee": (
        "hierarchical_no_callee",
        "e2_no_callee_seed{seed}",
    ),
}


@dataclass
class Job:
    control: str
    seed: int
    experiment: str
    run_dir: str
    log_path: str
    config_path: str | None = None
    device: str | None = None
    status: str = "PENDING"
    returncode: int | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_stable_json(path: Path, value: object) -> None:
    content = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != content:
            raise FileExistsError(f"Refusing to replace different metadata: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _write_index(path: Path, jobs: list[Job]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps([asdict(job) for job in jobs], indent=2) + "\n")
    temporary.replace(path)


def _validate_manifest(path: Path, expected_hash: str | None) -> tuple[dict, list[str], str]:
    manifest = json.loads(path.read_text())
    if set(manifest) != set(SPLIT_NAMES):
        raise ValueError(f"{path}: expected exactly the four standard splits")
    if any(not manifest[split] for split in SPLIT_NAMES):
        raise ValueError(f"{path}: every split must be non-empty")
    rows = [row for split in SPLIT_NAMES for row in manifest[split]]
    paths = [row["tensor_path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError(f"{path}: duplicate tensor paths")
    architecture_versions = {
        row.get("architecture_signature_version") for row in rows
    }
    if architecture_versions != {HIGH_LEVEL_ARCHITECTURE_SIGNATURE_VERSION}:
        raise ValueError(
            f"{path}: stale architecture signatures "
            f"{sorted(map(str, architecture_versions))}"
        )
    topology_versions = {row.get("topology_signature_version") for row in rows}
    if topology_versions != {HIGH_LEVEL_TOPOLOGY_SIGNATURE_VERSION}:
        raise ValueError(
            f"{path}: stale topology signatures {sorted(map(str, topology_versions))}"
        )
    architecture_ids = {
        split: {row["architecture_id"] for row in manifest[split]}
        for split in ("train", "validation", "test")
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = architecture_ids[left] & architecture_ids[right]
        if overlap:
            raise ValueError(
                f"{path}: {len(overlap)} architecture IDs overlap {left}/{right}"
            )
    families = {
        row["kernel_family"]
        for split in ("train", "validation", "test")
        for row in manifest[split]
    }
    if families != FAMILIES:
        raise ValueError(f"{path}: expected E2 families {sorted(FAMILIES)}, got {sorted(families)}")
    actual_hash = split_sha256(manifest)
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(
            f"{path}: split SHA-256 is {actual_hash}, expected {expected_hash}"
        )
    return manifest, sorted(families), actual_hash


def _validate_complete(job: Job, split_hash: str) -> bool:
    run_dir = Path(job.run_dir)
    marker = run_dir / ("summary.csv" if job.control == "extra_trees" else "summary.json")
    if not marker.is_file() or not (run_dir / "predictions.csv").is_file():
        return False
    resolved_path = run_dir / "resolved_config.json"
    if not resolved_path.is_file():
        raise ValueError(f"Complete-looking run lacks resolved config: {run_dir}")
    resolved = json.loads(resolved_path.read_text())
    expected_model = (
        "extra_trees" if job.control == "extra_trees" else NEURAL_MODELS[job.control][0]
    )
    expected = {
        "experiment_name": job.experiment,
        "model": expected_model,
        "seed": job.seed,
        "split_sha256": split_hash,
    }
    mismatches = {
        key: (resolved.get(key), value)
        for key, value in expected.items()
        if resolved.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Existing completed run has incompatible identity: {run_dir}: {mismatches}")
    return True


def _neural_config(
    args, base: dict, control: str, seed: int, experiment: str,
    kernel_types: list[str], split_hash: str,
) -> dict:
    model_name = NEURAL_MODELS[control][0]
    seed_results = (args.output_dir / f"seed{seed}").resolve()
    run_dir = seed_results / experiment
    config = {
        **base,
        "experiment_name": experiment,
        "model": model_name,
        "study_id": STUDY_ID,
        "protocol_id": PROTOCOL_ID,
        "architecture_signature_version": HIGH_LEVEL_ARCHITECTURE_SIGNATURE_VERSION,
        "topology_signature_version": HIGH_LEVEL_TOPOLOGY_SIGNATURE_VERSION,
        "tensor_dir": str(args.tensor_dir),
        "tensor_source_revision": args.tensor_source_revision,
        "vocab_path": str(args.vocab),
        "split_manifest_path": str(args.manifest),
        "split_sha256": split_hash,
        "require_complete_split_manifest": True,
        "results_dir": str(seed_results),
        "checkpoint_dir": str(run_dir / "checkpoints"),
        "kernel_types": kernel_types,
        "archive_count": 31,
        "seed": seed,
        "distributed_world_size": 1,
        "effective_global_batch": int(base.get("batch_size", 16)),
    }
    if control == "topology_destroyed":
        config.update(
            graph_transform="permute_destinations_within_function_v2",
            corruption_seed=args.topology_corruption_seed,
            topology_transform_version=TOPOLOGY_DESTINATION_PERMUTE_VERSION,
        )
    elif control == "fusion":
        config.update(
            high_level_cache=str(args.high_level_cache),
            high_level_cache_sha256=args.high_level_cache_sha256,
            high_level_encoder="gatv2",
        )
    elif control == "no_local_message":
        config["causal_control"] = {
            "version": "no_local_message_v1",
            "bypassed_relations": [
                "instruction/control/instruction",
                "instruction/def_use/instruction",
                "block/control/block",
            ],
            "preserved": [
                "input_projections", "hierarchy", "pooling", "call_composition",
                "global_features", "context", "heads", "optimizer", "parameters",
            ],
        }
    elif control == "hierarchy_orderless":
        config["causal_control"] = {
            "version": "hierarchy_orderless_v1",
            "role": "primary_hierarchy_ablation",
            "removed": [
                "instruction_to_block_containment_pooling",
                "block_cfg_propagation",
                "block_to_function_containment_pooling",
                "leaf_first_call_composition",
                "callee_state_injection",
                "entry_reachable_function_dual_readout",
            ],
            "preserved": [
                "instruction_control_def_use_messages",
                "node_and_pragma_inputs",
                "orderless_block_and_function_bags",
                "global_features", "context", "heads", "optimizer",
                "active_parameter_count",
            ],
        }
    elif control == "no_block_cfg":
        config["causal_control"] = {
            "version": "no_block_cfg_v1",
            "role": "secondary_mechanism_ablation",
            "bypassed_relations": ["block/control/block"],
            "preserved": [
                "instruction_messages", "containment_pooling", "call_composition",
                "global_features", "context", "heads", "optimizer", "parameters",
            ],
        }
    elif control == "no_callee":
        config["causal_control"] = {
            "version": "no_callee_injection_v1",
            "role": "secondary_mechanism_ablation",
            "bypassed_relations": ["instruction/calls/function"],
            "preserved": [
                "instruction_messages", "block_cfg_messages", "containment_pooling",
                "function_pooling", "global_features", "context", "heads",
                "optimizer", "parameters",
            ],
        }
    return config


def _seeds_for_control(control: str, requested: list[int]) -> list[int]:
    return [42] if control == "no_local_message" else list(requested)


def _materialize_jobs(args, base: dict, kernel_types: list[str], split_hash: str) -> list[Job]:
    jobs: list[Job] = []
    controls = [control for control in MODEL_ORDER if control in args.models]
    slots = [device for device in args.devices for _ in range(args.jobs_per_device)]
    neural_position = 0
    for control in controls:
        seeds = _seeds_for_control(control, args.seeds)
        for seed in seeds:
            experiment = (
                f"e2_extra_trees_structural_seed{seed}"
                if control == "extra_trees"
                else NEURAL_MODELS[control][1].format(seed=seed)
            )
            run_dir = (args.output_dir / f"seed{seed}" / experiment).resolve()
            log_path = (
                args.output_dir / "logs" / f"seed{seed}" / f"{experiment}.log"
            ).resolve()
            device = None
            config_path = None
            if control != "extra_trees":
                device = slots[neural_position % len(slots)]
                neural_position += 1
                config = _neural_config(
                    args, base, control, seed, experiment, kernel_types, split_hash
                )
                partial = run_dir.exists() and any(run_dir.iterdir())
                suffix = ""
                if partial and not _validate_complete(
                    Job(control, seed, experiment, str(run_dir), str(log_path)),
                    split_hash,
                ):
                    checkpoint = run_dir / "checkpoints" / f"{experiment}_backup.pt"
                    if not args.resume:
                        status = "BLOCKED_PARTIAL_USE_RESUME"
                    elif not checkpoint.is_file():
                        status = "BLOCKED_NO_BACKUP_CHECKPOINT"
                    else:
                        status = "PENDING"
                        suffix = ".resume"
                        config["resume_checkpoint_path"] = str(checkpoint)
                else:
                    status = "PENDING"
                config_path = (
                    args.output_dir / "configs" / f"seed{seed}"
                    / f"{experiment}{suffix}.json"
                ).resolve()
                _write_stable_json(config_path, config)
            else:
                status = "PENDING"
                partial = run_dir.exists() and any(run_dir.iterdir())

            job = Job(
                control=control,
                seed=seed,
                experiment=experiment,
                run_dir=str(run_dir),
                log_path=str(log_path),
                config_path=str(config_path) if config_path else None,
                device=device,
                status=status,
            )
            if _validate_complete(job, split_hash):
                job.status = "SKIPPED_COMPLETE"
            elif control == "extra_trees" and partial:
                job.status = "BLOCKED_PARTIAL_EXTRA_TREES"
            jobs.append(job)
    return jobs


def _command(job: Job, args) -> list[str]:
    if job.control != "extra_trees":
        return [
            sys.executable, str(_REPO_ROOT / "scripts" / "train.py"),
            "--config", str(job.config_path),
        ]
    return [
        sys.executable, str(_REPO_ROOT / "scripts" / "run_e2_extratrees.py"),
        "--manifest", str(args.manifest),
        "--tensor-dir", str(args.tensor_dir),
        "--vocab", str(args.vocab),
        "--output-dir", job.run_dir,
        "--feature-cache", str(args.feature_cache),
        "--seed", str(job.seed),
        "--experiment-name", job.experiment,
    ]


def _gpu_preflight(device: str) -> None:
    environment = os.environ.copy()
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = device
    check = subprocess.run(
        [
            sys.executable, "-c",
            "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))",
        ],
        env=environment,
        capture_output=True,
        text=True,
    )
    if check.returncode:
        raise RuntimeError(
            f"CUDA preflight failed for device {device}: "
            f"{check.stderr.strip() or check.stdout.strip()}"
        )
    print(f"GPU {device}: {check.stdout.strip()}", flush=True)


def _launch(job: Job, args, environment: dict[str, str] | None = None):
    log_path = Path(job.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a" if args.resume else "w")
    process = subprocess.Popen(
        _command(job, args),
        cwd=_REPO_ROOT,
        env=environment,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    return process, log_handle


def _finish_job(job: Job, returncode: int, split_hash: str) -> bool:
    job.returncode = returncode
    if returncode == 0 and not _validate_complete(job, split_hash):
        print(
            f"FAILED: {job.experiment} exited successfully without a complete bundle",
            file=sys.stderr,
        )
        job.returncode = 2
    job.status = "COMPLETE" if job.returncode == 0 else "FAILED"
    return job.returncode != 0


def _run_neural(jobs: list[Job], args, index_path: Path) -> bool:
    pending = list(jobs)
    available = [device for device in args.devices for _ in range(args.jobs_per_device)]
    active: list[tuple[subprocess.Popen, Job, object]] = []
    failed = False
    try:
        while pending or active:
            while pending and available and not (failed and args.fail_fast):
                job = pending.pop(0)
                device = available.pop(0)
                job.device = device
                environment = os.environ.copy()
                environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                environment["CUDA_VISIBLE_DEVICES"] = device
                environment.setdefault("LL_HLS4ML_TQDM", "0")
                process, log_handle = _launch(job, args, environment)
                job.status = "RUNNING"
                active.append((process, job, log_handle))
                print(f"Started {job.experiment} on GPU {device}; log={job.log_path}", flush=True)
                _write_index(index_path, args.all_jobs)

            if not active:
                break
            time.sleep(2)
            remaining = []
            for process, job, log_handle in active:
                returncode = process.poll()
                if returncode is None:
                    remaining.append((process, job, log_handle))
                    continue
                log_handle.close()
                available.append(str(job.device))
                failed = _finish_job(job, returncode, args.split_hash) or failed
                print(f"{job.status}: {job.experiment}", flush=True)
                _write_index(index_path, args.all_jobs)
            active = remaining
        if failed and args.fail_fast:
            for job in pending:
                job.status = "SKIPPED_FAIL_FAST"
            _write_index(index_path, args.all_jobs)
    except KeyboardInterrupt:
        for process, job, log_handle in active:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log_handle.close()
            job.status = "INTERRUPTED"
            job.returncode = process.returncode
        _write_index(index_path, args.all_jobs)
        raise
    return failed


def _run_extra_trees(jobs: list[Job], args, index_path: Path) -> bool:
    failed = False
    for job in jobs:
        if failed and args.fail_fast:
            job.status = "SKIPPED_FAIL_FAST"
            continue
        job.status = "RUNNING"
        process, log_handle = _launch(job, args)
        print(f"Started {job.experiment} on CPU; log={job.log_path}", flush=True)
        _write_index(index_path, args.all_jobs)
        try:
            returncode = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            job.status = "INTERRUPTED"
            job.returncode = process.returncode
            log_handle.close()
            _write_index(index_path, args.all_jobs)
            raise
        log_handle.close()
        failed = _finish_job(job, returncode, args.split_hash) or failed
        print(f"{job.status}: {job.experiment}", flush=True)
        _write_index(index_path, args.all_jobs)
    _write_index(index_path, args.all_jobs)
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-dir", type=Path, required=True)
    parser.add_argument("--vocab", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--high-level-cache", type=Path)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-config", type=Path,
        default=_REPO_ROOT / "configs/studies/e2_replication.yaml",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 137])
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_ORDER, default=list(DEFAULT_MODELS),
    )
    parser.add_argument("--devices", nargs="+", default=["0"])
    parser.add_argument("--jobs-per-device", type=int, default=1)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--topology-corruption-seed", type=int, default=42)
    parser.add_argument("--expected-split-sha256", default=OFFICIAL_SPLIT_SHA256)
    parser.add_argument("--expected-vocab-sha256", default=OFFICIAL_VOCAB_SHA256)
    parser.add_argument(
        "--expected-high-level-cache-sha256", default=OFFICIAL_HIGH_LEVEL_SHA256,
    )
    parser.add_argument("--tensor-source-revision", default=OFFICIAL_TENSOR_REVISION)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if args.jobs_per_device < 1:
        parser.error("--jobs-per-device must be positive")
    if len(args.devices) != len(set(args.devices)):
        parser.error("--devices must not contain duplicates")
    if len(args.models) != len(set(args.models)):
        parser.error("--models must not contain duplicates")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must not contain duplicates")
    args.tensor_dir = args.tensor_dir.resolve()
    args.manifest = args.manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    args.base_config = args.base_config.resolve()
    args.vocab = (args.vocab or args.tensor_dir / "vocab.json").resolve()
    args.feature_cache = (
        args.feature_cache or args.output_dir / "feature_cache/e2_core_context.pkl"
    ).resolve()
    args.high_level_cache = (
        args.high_level_cache.resolve() if args.high_level_cache else None
    )
    if not args.tensor_dir.is_dir():
        raise FileNotFoundError(args.tensor_dir)
    for path in (args.vocab, args.manifest, args.base_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    if "fusion" in args.models and (
        args.high_level_cache is None or not args.high_level_cache.is_file()
    ):
        parser.error("--high-level-cache must name an existing file when fusion is selected")

    vocab_hash = _sha256(args.vocab)
    if args.expected_vocab_sha256 and vocab_hash != args.expected_vocab_sha256:
        raise ValueError(
            f"{args.vocab}: SHA-256 is {vocab_hash}, "
            f"expected {args.expected_vocab_sha256}"
        )
    args.high_level_cache_sha256 = (
        _sha256(args.high_level_cache) if args.high_level_cache else None
    )
    if (
        "fusion" in args.models
        and args.expected_high_level_cache_sha256
        and args.high_level_cache_sha256 != args.expected_high_level_cache_sha256
    ):
        raise ValueError(
            f"{args.high_level_cache}: SHA-256 is {args.high_level_cache_sha256}, "
            f"expected {args.expected_high_level_cache_sha256}"
        )

    _, kernel_types, split_hash = _validate_manifest(
        args.manifest, args.expected_split_sha256
    )
    args.split_hash = split_hash
    base = yaml.safe_load(args.base_config.read_text())
    if not isinstance(base, dict):
        raise ValueError(f"Base config must be a mapping: {args.base_config}")
    if args.batch_size is not None:
        base["batch_size"] = args.batch_size
    if args.num_workers is not None:
        base["num_workers"] = args.num_workers

    metadata = {
        "study_id": STUDY_ID,
        "protocol_id": PROTOCOL_ID,
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "split_sha256": split_hash,
        "vocab": str(args.vocab),
        "vocab_sha256": vocab_hash,
        "high_level_cache": (
            str(args.high_level_cache) if args.high_level_cache else None
        ),
        "high_level_cache_sha256": args.high_level_cache_sha256,
        "tensor_source_revision": args.tensor_source_revision,
        "topology_corruption_seed": args.topology_corruption_seed,
        "replication_seeds": args.seeds,
        "no_local_message_seed": 42,
    }
    if any(model in HIERARCHY_ABLATIONS for model in args.models):
        metadata["hierarchy_ablation_suite"] = {
            "version": "e2_hierarchy_ablation_v1",
            "primary": "hierarchy_orderless",
            "secondary": ["no_block_cfg", "no_callee"],
            "requested_models": args.models,
        }
    _write_stable_json(args.output_dir / "study_metadata.json", metadata)
    jobs = _materialize_jobs(args, base, kernel_types, split_hash)
    args.all_jobs = jobs
    index_path = args.output_dir / "study_index.json"
    _write_index(index_path, jobs)
    runnable = [job for job in jobs if job.status == "PENDING"]
    blocked = [job for job in jobs if job.status.startswith("BLOCKED")]
    if args.dry_run:
        for job in runnable:
            prefix = (
                f"CUDA_VISIBLE_DEVICES={shlex.quote(str(job.device))} "
                if job.control != "extra_trees" else ""
            )
            print(prefix + shlex.join(_command(job, args)))
        print(f"Runnable={len(runnable)} blocked={len(blocked)} index={index_path}")
        return
    if blocked:
        for job in blocked:
            print(f"{job.status}: {job.run_dir}", file=sys.stderr)
        raise RuntimeError("Resolve partial runs; neural runs may be continued with --resume")
    if not runnable:
        print("All requested E2 runs are already complete.")
        return

    neural = [job for job in runnable if job.control != "extra_trees"]
    extra_trees = [job for job in runnable if job.control == "extra_trees"]
    for device in dict.fromkeys(job.device for job in neural):
        _gpu_preflight(str(device))
    failed = _run_neural(neural, args, index_path) if neural else False
    if failed and args.fail_fast:
        for job in extra_trees:
            job.status = "SKIPPED_FAIL_FAST"
        _write_index(index_path, jobs)
    elif extra_trees:
        failed = _run_extra_trees(extra_trees, args, index_path) or failed
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
