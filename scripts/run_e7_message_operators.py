#!/usr/bin/env python3
"""Run the staged E7 hierarchical architecture study."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from ll_hls4ml.models.hierarchical_operators import OPERATOR_PROFILES
from ll_hls4ml.models.hierarchical_attention import ATTENTION_PROFILES
from ll_hls4ml.models.hierarchical_edge_attention import EDGE_ATTENTION_PROFILES
from ll_hls4ml.models.hierarchical_variables import VARIABLE_ROUTE_PROFILES
from ll_hls4ml.io.schema import DERIVED_DEF_USE_EDGE
from scripts.run_e2 import (
    OFFICIAL_SPLIT_SHA256,
    OFFICIAL_TENSOR_REVISION,
    OFFICIAL_VOCAB_SHA256,
    _gpu_preflight,
    _sha256,
    _validate_manifest,
    _write_stable_json,
)


STUDY_ID = "e7_message_operator_funnel_v1"
PROTOCOL_ID = "e7_e2_frozen_operator_funnel_2026_09_17"
SCREEN_SPLIT_SHA256 = {
    7: "68eaf13add5f7547ede743bba87b5bf49705c3146b047c84d44db6717159f052",
    42: "fe0879fcbebfc81958c4c967cdd963e9f29a54d97c3d684adcb9b80487653030",
    137: "b777d0cf0b33ddf6d532c38a3a8df74957d4493c2b76d20e8dbad0c9dc6d0e27",
}
DEFAULT_SCREEN_ROOT = (
    _REPO_ROOT / "artifacts/results/e6_architecture_scaling_v1"
)
MECHANISM_CANDIDATES = tuple(
    name
    for name, spec in OPERATOR_PROFILES.items()
    if name != "forward_mean" and spec.rank is not None
)
PERFORMANCE_CANDIDATES = tuple(
    name for name, spec in OPERATOR_PROFILES.items() if spec.rank is None
)
ATTENTION_CANDIDATES = tuple(ATTENTION_PROFILES)
EDGE_ATTENTION_CANDIDATES = tuple(EDGE_ATTENTION_PROFILES)
VARIABLE_CANDIDATES = tuple(VARIABLE_ROUTE_PROFILES)
SCREEN_CANDIDATES = (
    *MECHANISM_CANDIDATES,
    *PERFORMANCE_CANDIDATES,
    *ATTENTION_CANDIDATES,
    *EDGE_ATTENTION_CANDIDATES,
    *VARIABLE_CANDIDATES,
)
ALL_CANDIDATES = ("canonical", *SCREEN_CANDIDATES)

# A first wave chosen for diversity of inductive bias, not an assertion that
# the omitted ladder/capacity variants are uninteresting. The remaining models
# are conditional follow-ups once a parent mechanism is promising.
CORE_CANDIDATES = (
    "receiver_gate",
    "bidirectional",
    "gate_bidir",
    "pna",
    "pna_gate_bidir",
    "wide_full_operator",
    "edge_attention",
    "attn_instruction",
    "attn_block",
    "attn_dual",
    "variable_route_all",
)
CANDIDATE_SETS = {
    "core": CORE_CANDIDATES,
    "mechanism": MECHANISM_CANDIDATES,
    "performance": PERFORMANCE_CANDIDATES,
    "attention": (*ATTENTION_CANDIDATES, *EDGE_ATTENTION_CANDIDATES),
    "variable": VARIABLE_CANDIDATES,
    "all": SCREEN_CANDIDATES,
}


def candidate_track(candidate: str) -> str:
    if candidate == "canonical":
        return "reference"
    if candidate in MECHANISM_CANDIDATES:
        return "mechanism"
    if candidate in PERFORMANCE_CANDIDATES:
        return "performance"
    if candidate in ATTENTION_CANDIDATES:
        return "attention"
    if candidate in EDGE_ATTENTION_CANDIDATES:
        return "edge_attention"
    if candidate in VARIABLE_CANDIDATES:
        return "variable"
    raise ValueError(f"Unknown E7 candidate: {candidate}")


@dataclass
class Job:
    candidate: str
    seed: int
    experiment: str
    run_dir: str
    log_path: str
    config_path: str
    device: str | None = None
    status: str = "PENDING"
    returncode: int | None = None


def _manifest_path(args, seed: int) -> Path:
    if args.stage == "confirm":
        return args.manifest
    budget = 10 if args.stage == "smoke" else 50
    return args.screen_root / f"protocol/manifests/seed{seed}/train_{budget:03d}.json"


def _baseline_run(args, seed: int) -> Path:
    return (
        args.screen_root
        / f"runs/h0/seed{seed}/e6_h0_seed{seed}_p050"
    )


def _validate_screen_baseline(
    args, seed: int, split_hash: str
) -> Path:
    run_dir = _baseline_run(args, seed)
    resolved_path = run_dir / "resolved_config.json"
    predictions = run_dir / "predictions.csv"
    if not resolved_path.is_file() or not predictions.is_file():
        raise FileNotFoundError(
            f"Reusable E6 H0 bundle is incomplete for seed {seed}: {run_dir}"
        )
    resolved = json.loads(resolved_path.read_text())
    expected = {
        "model": "hierarchical",
        "seed": seed,
        "split_sha256": split_hash,
        "tensor_source_revision": OFFICIAL_TENSOR_REVISION,
        "hidden_dim": 64,
        "num_layers": 3,
        "epochs": 400,
        "patience": 20,
        "early_stopping_metric": "smape",
        "parameter_count": 239696,
    }
    mismatches = {
        key: (resolved.get(key), value)
        for key, value in expected.items()
        if resolved.get(key) != value
    }
    evaluated = set(resolved.get("evaluation_splits", []))
    if "validation" not in evaluated:
        mismatches["evaluation_splits"] = (sorted(evaluated), "contains validation")
    if mismatches:
        raise ValueError(f"Incompatible E6 H0 baseline for seed {seed}: {mismatches}")
    return predictions.resolve()


def candidate_config(candidate: str) -> dict:
    if candidate == "canonical":
        return {"model": "hierarchical"}
    if candidate in OPERATOR_PROFILES and candidate != "forward_mean":
        return {
            "model": "hierarchical_operator",
            "operator_profile": candidate,
        }
    if candidate in ATTENTION_PROFILES:
        return {
            "model": "hierarchical_attention",
            **ATTENTION_PROFILES[candidate],
        }
    if candidate in EDGE_ATTENTION_PROFILES:
        return {
            "model": "hierarchical_edge_attention",
            **EDGE_ATTENTION_PROFILES[candidate],
        }
    if candidate in VARIABLE_ROUTE_PROFILES:
        return {
            "model": "hierarchical_variable_route",
            **VARIABLE_ROUTE_PROFILES[candidate],
        }
    raise ValueError(f"Unknown E7 candidate: {candidate}")


def _requires_pna(candidate: str) -> bool:
    profile = candidate_config(candidate).get("operator_profile")
    return bool(
        profile
        and OPERATOR_PROFILES[profile].aggregation == "pna"
    )


def _pna_degree_stats(
    manifest: Path,
    tensor_dir: Path,
    split_hash: str,
    cache_path: Path,
) -> dict[str, float]:
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text())
        expected = {
            "split_sha256": split_hash,
            "tensor_dir": str(tensor_dir),
        }
        if any(cached.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Incompatible PNA degree-stat cache: {cache_path}")
        return cached["averages"]
    rows = json.loads(manifest.read_text())["train"]
    relations = {
        "instruction_control": (
            ("instruction", "control", "instruction"),
            "instruction",
        ),
        "instruction_def_use": (DERIVED_DEF_USE_EDGE, "instruction"),
        "block_cfg": (("block", "control", "block"), "block"),
    }
    totals = {
        f"{name}_{direction}": [0.0, 0]
        for name in relations
        for direction in ("forward", "reverse")
    }
    for index, row in enumerate(rows, start=1):
        path = tensor_dir / row["tensor_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        data = torch.load(path, map_location="cpu", weights_only=False)
        for name, (edge_type, node_type) in relations.items():
            edge_index = data[edge_type].edge_index
            node_count = data[node_type].num_nodes
            for direction, target_row in (("forward", 1), ("reverse", 0)):
                degree = torch.bincount(
                    edge_index[target_row], minlength=node_count
                ).float()
                accumulator = totals[f"{name}_{direction}"]
                accumulator[0] += float(torch.log1p(degree).sum())
                accumulator[1] += int(node_count)
        if index % 1000 == 0:
            print(
                f"PNA degree statistics: {index}/{len(rows)} tensors",
                flush=True,
            )
    averages = {
        key: total / max(count, 1) for key, (total, count) in totals.items()
    }
    payload = {
        "split_sha256": split_hash,
        "tensor_dir": str(tensor_dir),
        "train_designs": len(rows),
        "averages": averages,
    }
    _write_stable_json(cache_path, payload)
    return averages


def _config(
    args,
    base: dict,
    candidate: str,
    seed: int,
    manifest: Path,
    split_hash: str,
    experiment: str,
    run_dir: Path,
) -> dict:
    candidate_fields = candidate_config(candidate)
    profile = candidate_fields.get("operator_profile")
    config = {
        **base,
        "experiment_name": experiment,
        **candidate_fields,
        "study_id": STUDY_ID,
        "protocol_id": PROTOCOL_ID,
        "e7_stage": args.stage,
        "operator_profile": profile,
        "operator_spec": (
            asdict(OPERATOR_PROFILES[profile]) if profile is not None else None
        ),
        "tensor_dir": str(args.tensor_dir),
        "tensor_source_revision": OFFICIAL_TENSOR_REVISION,
        "vocab_path": str(args.vocab),
        "split_manifest_path": str(manifest),
        "split_sha256": split_hash,
        "require_complete_split_manifest": True,
        "results_dir": str(run_dir.parent),
        "checkpoint_dir": str(run_dir / "checkpoints"),
        "seed": seed,
        "distributed_world_size": 1,
        "effective_global_batch": int(base.get("batch_size", 16)),
        "matching_status": {
            "training_contract_matched": args.stage != "smoke",
            "same_hidden_width": True,
            "parameter_matched": candidate == "canonical",
            "wall_clock_matched": False,
        },
    }
    if _requires_pna(candidate):
        config["pna_avg_log_degrees"] = args.pna_stats[seed]
    if args.stage == "screen":
        config.update(
            evaluation_splits=["validation"],
            baseline_predictions_path=str(
                _validate_screen_baseline(args, seed, split_hash)
            ),
            selection_use="screening_only_no_test_evaluation",
        )
    elif args.stage == "smoke":
        config.update(
            epochs=2,
            patience=2,
            evaluation_splits=["validation"],
            selection_use="engineering_only_do_not_rank",
        )
    else:
        config.update(
            evaluation_splits=["validation", "test", "exemplar"],
            selection_use="frozen_shortlist_confirmation",
        )
    return config


def _validate_complete(job: Job, split_hash: str) -> bool:
    run_dir = Path(job.run_dir)
    summary = run_dir / "summary.json"
    predictions = run_dir / "predictions.csv"
    resolved_path = run_dir / "resolved_config.json"
    if not summary.is_file() or not predictions.is_file():
        return False
    if not resolved_path.is_file():
        raise ValueError(f"Complete-looking run lacks resolved config: {run_dir}")
    resolved = json.loads(resolved_path.read_text())
    candidate_fields = candidate_config(job.candidate)
    expected = {
        "experiment_name": job.experiment,
        **candidate_fields,
        "seed": job.seed,
        "split_sha256": split_hash,
    }
    mismatches = {
        key: (resolved.get(key), value)
        for key, value in expected.items()
        if resolved.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Existing run has incompatible identity: {mismatches}")
    return True


def _write_index(path: Path, jobs: list[Job]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps([asdict(job) for job in jobs], indent=2) + "\n")
    temporary.replace(path)


def _write_campaign_metadata(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _merge_jobs(path: Path, jobs: list[Job]) -> list[Job]:
    """Accumulate sequential subset launches in one campaign directory."""
    merged: dict[tuple[str, int], Job] = {}
    if path.is_file():
        for raw in json.loads(path.read_text()):
            previous = Job(**raw)
            merged[(previous.candidate, previous.seed)] = previous
    for job in jobs:
        merged[(job.candidate, job.seed)] = job
    candidate_order = {name: index for index, name in enumerate(ALL_CANDIDATES)}
    return sorted(
        merged.values(),
        key=lambda job: (candidate_order[job.candidate], job.seed),
    )


def _merge_metadata(path: Path, current: dict) -> dict:
    if not path.is_file():
        return current
    previous = json.loads(path.read_text())
    for key in (
        "study_id",
        "protocol_id",
        "stage",
        "tensor_dir",
        "vocab_path",
        "base_config_path",
        "reused_screen_baseline_root",
    ):
        if key in previous and previous.get(key) != current.get(key):
            raise ValueError(
                f"Cannot mix incompatible E7 campaigns in {path}: {key}"
            )
    candidate_order = {name: index for index, name in enumerate(ALL_CANDIDATES)}
    candidates = sorted(
        set(previous.get("candidates", ())) | set(current["candidates"]),
        key=candidate_order.__getitem__,
    )
    seeds = sorted(set(previous.get("seeds", ())) | set(current["seeds"]))
    merged = {**previous, **current, "candidates": candidates, "seeds": seeds}
    for key in (
        "operator_specs",
        "candidate_configs",
        "candidate_tracks",
        "pna_avg_log_degrees",
        "split_hashes",
    ):
        old = {str(name): value for name, value in previous.get(key, {}).items()}
        new = {str(name): value for name, value in current.get(key, {}).items()}
        merged[key] = {**old, **new}
    merged["last_invocation_candidates"] = current["candidates"]
    merged["last_invocation_seeds"] = current["seeds"]
    return merged


def _materialize(args, base: dict) -> list[Job]:
    jobs = []
    split_hashes = {}
    pna_by_split: dict[str, dict[str, float]] = {}
    for seed in args.seeds:
        manifest = _manifest_path(args, seed).resolve()
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        expected = (
            OFFICIAL_SPLIT_SHA256
            if args.stage == "confirm"
            else SCREEN_SPLIT_SHA256.get(seed) if args.stage == "screen" else None
        )
        _, _, split_hash = _validate_manifest(manifest, expected)
        split_hashes[seed] = split_hash
        if any(_requires_pna(candidate) for candidate in args.candidates):
            cache_path = (
                args.output_dir
                / "protocol"
                / "pna_degree_stats"
                / f"seed{seed}.json"
            )
            if split_hash not in pna_by_split:
                pna_by_split[split_hash] = _pna_degree_stats(
                    manifest,
                    args.tensor_dir,
                    split_hash,
                    cache_path,
                )
            args.pna_stats[seed] = pna_by_split[split_hash]
        for candidate in args.candidates:
            experiment = f"e7_{args.stage}_{candidate}_seed{seed}"
            run_dir = (
                args.output_dir / "runs" / candidate / f"seed{seed}" / experiment
            ).resolve()
            log_path = (
                args.output_dir / "logs" / candidate / f"seed{seed}.log"
            ).resolve()
            config_path = (
                args.output_dir / "configs" / candidate / f"seed{seed}.json"
            ).resolve()
            config = _config(
                args,
                base,
                candidate,
                seed,
                manifest,
                split_hash,
                experiment,
                run_dir,
            )
            partial = run_dir.exists() and any(run_dir.iterdir())
            status = "PENDING"
            suffix = ""
            provisional = Job(
                candidate,
                seed,
                experiment,
                str(run_dir),
                str(log_path),
                str(config_path),
            )
            if _validate_complete(provisional, split_hash):
                status = "SKIPPED_COMPLETE"
            elif partial:
                checkpoint = run_dir / "checkpoints" / f"{experiment}_backup.pt"
                if not args.resume:
                    status = "BLOCKED_PARTIAL_USE_RESUME"
                elif not checkpoint.is_file():
                    status = "BLOCKED_NO_BACKUP_CHECKPOINT"
                else:
                    suffix = ".resume"
                    config["resume_checkpoint_path"] = str(checkpoint)
            if suffix:
                config_path = config_path.with_suffix(suffix + ".json")
            _write_stable_json(config_path, config)
            jobs.append(
                Job(
                    candidate,
                    seed,
                    experiment,
                    str(run_dir),
                    str(log_path),
                    str(config_path),
                    status=status,
                )
            )
    args.split_hashes = split_hashes
    return jobs


def _command(job: Job) -> list[str]:
    return [
        sys.executable,
        str(_REPO_ROOT / "scripts/train.py"),
        "--config",
        job.config_path,
    ]


def _run(jobs: list[Job], args, index_path: Path) -> bool:
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
                Path(job.log_path).parent.mkdir(parents=True, exist_ok=True)
                log_handle = Path(job.log_path).open("a" if args.resume else "w")
                process = subprocess.Popen(
                    _command(job),
                    cwd=_REPO_ROOT,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
                job.status = "RUNNING"
                active.append((process, job, log_handle))
                print(f"Started {job.experiment} on GPU {device}", flush=True)
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
                job.returncode = returncode
                split_hash = args.split_hashes[job.seed]
                complete = returncode == 0 and _validate_complete(job, split_hash)
                job.status = "COMPLETE" if complete else "FAILED"
                failed = failed or not complete
                print(f"{job.status}: {job.experiment}", flush=True)
                _write_index(index_path, args.all_jobs)
            active = remaining
        if failed and args.fail_fast:
            for job in pending:
                job.status = "SKIPPED_FAIL_FAST"
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
    _write_index(index_path, args.all_jobs)
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "screen", "confirm"), required=True)
    parser.add_argument("--tensor-dir", type=Path, required=True)
    parser.add_argument("--vocab", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--screen-root", type=Path, default=DEFAULT_SCREEN_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=_REPO_ROOT / "configs/studies/e2_replication.yaml",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--candidates", nargs="+", choices=ALL_CANDIDATES)
    selection.add_argument(
        "--candidate-set",
        choices=tuple(CANDIDATE_SETS),
        help="Named subset; 'core' is the recommended diverse first wave.",
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--devices", nargs="+", default=["0"])
    parser.add_argument("--jobs-per-device", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if args.jobs_per_device < 1:
        parser.error("--jobs-per-device must be positive")
    if len(args.devices) != len(set(args.devices)):
        parser.error("--devices must not contain duplicates")
    args.seeds = args.seeds or (
        [42]
        if args.stage == "smoke"
        else [7, 42, 137] if args.stage == "confirm" else [7, 42]
    )
    if args.candidate_set:
        args.candidates = list(CANDIDATE_SETS[args.candidate_set])
    if args.candidates is None:
        parser.error(
            "choose an explicit --candidates subset or --candidate-set; "
            "use --candidate-set core for the recommended first wave"
        )
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must not contain duplicates")
    if len(args.candidates) != len(set(args.candidates)):
        parser.error("--candidates must not contain duplicates")
    if args.stage == "screen" and "canonical" in args.candidates:
        parser.error("screen reuses the E6 H0 baseline; do not request canonical")
    if args.stage == "confirm" and args.manifest is None:
        parser.error("--manifest is required for confirmation")

    args.tensor_dir = args.tensor_dir.resolve()
    args.vocab = (args.vocab or args.tensor_dir / "vocab.json").resolve()
    args.manifest = args.manifest.resolve() if args.manifest else None
    args.screen_root = args.screen_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.base_config = args.base_config.resolve()
    if not args.tensor_dir.is_dir():
        raise FileNotFoundError(args.tensor_dir)
    for path in (args.vocab, args.base_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    vocab_hash = _sha256(args.vocab)
    if vocab_hash != OFFICIAL_VOCAB_SHA256:
        raise ValueError(
            f"{args.vocab}: SHA-256 is {vocab_hash}, expected {OFFICIAL_VOCAB_SHA256}"
        )
    base = yaml.safe_load(args.base_config.read_text())
    if not isinstance(base, dict):
        raise ValueError(f"Base config must be a mapping: {args.base_config}")

    args.pna_stats = {}
    jobs = _materialize(args, base)
    slots = [device for device in args.devices for _ in range(args.jobs_per_device)]
    for index, job in enumerate(jobs):
        job.device = slots[index % len(slots)]
    metadata = {
        "study_id": STUDY_ID,
        "protocol_id": PROTOCOL_ID,
        "stage": args.stage,
        "tensor_dir": str(args.tensor_dir),
        "vocab_path": str(args.vocab),
        "base_config_path": str(args.base_config),
        "candidates": args.candidates,
        "seeds": args.seeds,
        "operator_specs": {
            name: asdict(OPERATOR_PROFILES[candidate_config(name)["operator_profile"]])
            for name in args.candidates
            if candidate_config(name).get("operator_profile") in OPERATOR_PROFILES
        },
        "candidate_configs": {
            name: candidate_config(name) for name in args.candidates
        },
        "candidate_tracks": {
            name: candidate_track(name) for name in args.candidates
        },
        "pna_avg_log_degrees": args.pna_stats,
        "split_hashes": args.split_hashes,
        "reused_screen_baseline_root": (
            str(args.screen_root) if args.stage == "screen" else None
        ),
        "screen_does_not_evaluate_test_or_exemplar": args.stage == "screen",
        "matching": {
            "training_contract": "matched except engineering smoke",
            "parameters": "not matched; recorded per run",
            "wall_clock": "not matched; recorded per run",
        },
    }
    metadata_path = args.output_dir / "study_metadata.json"
    metadata = _merge_metadata(metadata_path, metadata)
    _write_campaign_metadata(metadata_path, metadata)
    index_path = args.output_dir / "study_index.json"
    args.all_jobs = _merge_jobs(index_path, jobs)
    _write_index(index_path, args.all_jobs)
    runnable = [job for job in jobs if job.status == "PENDING"]
    blocked = [job for job in jobs if job.status.startswith("BLOCKED")]
    if args.dry_run:
        for job in runnable:
            print(
                f"CUDA_VISIBLE_DEVICES={shlex.quote(str(job.device))} "
                + shlex.join(_command(job))
            )
        print(f"Runnable={len(runnable)} blocked={len(blocked)} index={index_path}")
        return
    if blocked:
        for job in blocked:
            print(f"{job.status}: {job.run_dir}", file=sys.stderr)
        raise RuntimeError("Resolve partial runs or continue with --resume")
    if not runnable:
        print("All requested E7 runs are already complete.")
        return
    for device in args.devices:
        _gpu_preflight(device)
    if _run(runnable, args, index_path):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
