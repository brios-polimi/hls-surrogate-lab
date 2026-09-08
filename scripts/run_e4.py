#!/usr/bin/env python3
"""Run E4 H0 folds as independent processes on one or more local GPUs."""

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

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from ll_hls4ml.data.protocols import SPLIT_NAMES
from ll_hls4ml.data.signatures import HIGH_LEVEL_ARCHITECTURE_SIGNATURE_VERSION


FAMILIES = (
    "2layer", "3layer", "conv1d", "conv2d", "dense_latency",
    "dense_resource", "rule4ml",
)
DEFAULT_FAMILY_ORDER = (
    "conv2d", "rule4ml", "dense_resource", "2layer", "3layer",
    "conv1d", "dense_latency",
)
STUDY_ID = "e4_leave_one_family_out_exact_v1"


@dataclass
class Job:
    family: str
    seed: int
    device: str
    experiment: str
    config_path: str
    log_path: str
    run_dir: str
    status: str = "PENDING"
    returncode: int | None = None


def _write_stable_json(path: Path, value: object) -> None:
    content = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != content:
            raise FileExistsError(f"Refusing to replace different config: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _write_index(path: Path, jobs: list[Job]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps([asdict(job) for job in jobs], indent=2) + "\n")
    temporary.replace(path)


def _validate_manifest(path: Path, held_out: str) -> list[str]:
    manifest = json.loads(path.read_text())
    if set(manifest) != set(SPLIT_NAMES):
        raise ValueError(f"{path}: expected exactly the four standard splits")
    rows = [row for split in SPLIT_NAMES for row in manifest[split]]
    paths = [row["tensor_path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError(f"{path}: duplicate tensor paths")
    versions = {row.get("architecture_signature_version") for row in rows}
    if versions != {HIGH_LEVEL_ARCHITECTURE_SIGNATURE_VERSION}:
        raise ValueError(
            f"{path}: stale architecture grouping {sorted(map(str, versions))}"
        )
    if {row["kernel_family"] for row in manifest["test"]} != {held_out}:
        raise ValueError(f"{path}: test split is not exactly {held_out}")
    if any(
        row["kernel_family"] == held_out
        for split in ("train", "validation") for row in manifest[split]
    ):
        raise ValueError(f"{path}: held-out family leaks into training/validation")
    return sorted({row["kernel_family"] for row in rows if row["kernel_family"] != "exemplar"})


def _gpu_preflight(device: str) -> None:
    environment = os.environ.copy()
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = device
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; assert torch.cuda.is_available(); "
            "print(torch.cuda.get_device_name(0))",
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


def _materialize_jobs(args, base: dict) -> list[Job]:
    jobs = []
    slots = [device for device in args.devices for _ in range(args.jobs_per_device)]
    for position, (seed, family) in enumerate(
        (seed, family) for seed in args.seeds for family in args.families
    ):
        manifest_path = (args.manifest_dir / f"leave_{family}_out.json").resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        kernel_types = _validate_manifest(manifest_path, family)
        experiment = f"e4_h0_leave_{family}_out_seed{seed}"
        seed_results = (args.output_dir / f"seed{seed}").resolve()
        run_dir = seed_results / experiment
        checkpoint_dir = run_dir / "checkpoints"
        resume_path = checkpoint_dir / f"{experiment}_backup.pt"
        complete = (run_dir / "REPORT.md").is_file() and (
            run_dir / "predictions.csv"
        ).is_file()
        partial = run_dir.exists() and any(run_dir.iterdir())

        config = {
            **base,
            "experiment_name": experiment,
            "study_id": STUDY_ID,
            "protocol_id": f"leave_{family}_out_exact_v1",
            "held_out_family": family,
            "architecture_signature_version": (
                HIGH_LEVEL_ARCHITECTURE_SIGNATURE_VERSION
            ),
            "tensor_dir": str(args.tensor_dir.resolve()),
            "vocab_path": str(args.vocab.resolve()),
            "split_manifest_path": str(manifest_path),
            "require_complete_split_manifest": True,
            "results_dir": str(seed_results),
            "checkpoint_dir": str(checkpoint_dir),
            "kernel_types": kernel_types,
            "seed": seed,
            "distributed_world_size": 1,
            "effective_global_batch": int(base.get("batch_size", 16)),
        }
        suffix = ""
        status = "PENDING"
        if complete:
            status = "SKIPPED_COMPLETE"
        elif partial:
            if not args.resume:
                status = "BLOCKED_PARTIAL_USE_RESUME"
            elif not resume_path.is_file():
                status = "BLOCKED_NO_BACKUP_CHECKPOINT"
            else:
                config["resume_checkpoint_path"] = str(resume_path)
                suffix = ".resume"

        config_path = (
            args.output_dir / "configs" / f"seed{seed}"
            / f"{experiment}{suffix}.json"
        ).resolve()
        _write_stable_json(config_path, config)
        jobs.append(Job(
            family=family,
            seed=seed,
            device=slots[position % len(slots)],
            experiment=experiment,
            config_path=str(config_path),
            log_path=str((
                args.output_dir / "logs" / f"seed{seed}" / f"{experiment}.log"
            ).resolve()),
            run_dir=str(run_dir),
            status=status,
        ))
    return jobs


def _command(job: Job) -> list[str]:
    return [
        sys.executable, str(_REPO_ROOT / "scripts" / "train.py"),
        "--config", job.config_path,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-dir", type=Path, required=True)
    parser.add_argument("--vocab", type=Path)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-config", type=Path,
        default=_REPO_ROOT / "configs/studies/e4_leave_family_out.yaml",
    )
    parser.add_argument(
        "--families", nargs="+", choices=FAMILIES,
        default=list(DEFAULT_FAMILY_ORDER),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--devices", nargs="+", default=["0"],
        help="CUDA device indices or UUIDs; each child sees its assigned GPU as cuda:0",
    )
    parser.add_argument("--jobs-per-device", type=int, default=1)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if args.jobs_per_device < 1:
        parser.error("--jobs-per-device must be positive")
    args.tensor_dir = args.tensor_dir.resolve()
    args.manifest_dir = args.manifest_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.base_config = args.base_config.resolve()
    args.vocab = (args.vocab or args.tensor_dir / "vocab.json").resolve()
    if not args.tensor_dir.is_dir():
        raise FileNotFoundError(args.tensor_dir)
    if not args.vocab.is_file():
        raise FileNotFoundError(args.vocab)
    base = yaml.safe_load(args.base_config.read_text())
    if args.batch_size is not None:
        base["batch_size"] = args.batch_size
    if args.num_workers is not None:
        base["num_workers"] = args.num_workers

    jobs = _materialize_jobs(args, base)
    index_path = args.output_dir / "study_index.json"
    _write_index(index_path, jobs)
    runnable = [job for job in jobs if job.status == "PENDING"]
    blocked = [job for job in jobs if job.status.startswith("BLOCKED")]
    if args.dry_run:
        for job in runnable:
            environment = f"CUDA_VISIBLE_DEVICES={shlex.quote(job.device)}"
            print(environment, shlex.join(_command(job)))
        print(f"Runnable={len(runnable)} blocked={len(blocked)} index={index_path}")
        return
    if blocked:
        for job in blocked:
            print(f"{job.status}: {job.run_dir}", file=sys.stderr)
        raise RuntimeError("Resolve partial runs or pass --resume before launching")
    if not runnable:
        print("All requested E4 folds are already complete.")
        return

    for device in dict.fromkeys(job.device for job in runnable):
        _gpu_preflight(device)

    pending = list(runnable)
    available = [
        device for device in args.devices for _ in range(args.jobs_per_device)
    ]
    active: list[tuple[subprocess.Popen, Job, object]] = []
    failed = False
    try:
        while pending or active:
            while pending and available and not (failed and args.fail_fast):
                job = pending.pop(0)
                device = available.pop(0)
                job.device = device
                log_path = Path(job.log_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_handle = log_path.open("a" if args.resume else "w")
                environment = os.environ.copy()
                environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                environment["CUDA_VISIBLE_DEVICES"] = device
                job.status = "RUNNING"
                process = subprocess.Popen(
                    _command(job), cwd=_REPO_ROOT, env=environment,
                    stdout=log_handle, stderr=subprocess.STDOUT,
                )
                active.append((process, job, log_handle))
                print(
                    f"Started {job.experiment} on GPU {device}; log={log_path}",
                    flush=True,
                )
                _write_index(index_path, jobs)

            time.sleep(2)
            remaining = []
            for process, job, log_handle in active:
                returncode = process.poll()
                if returncode is None:
                    remaining.append((process, job, log_handle))
                    continue
                log_handle.close()
                available.append(job.device)
                job.returncode = returncode
                job.status = "COMPLETE" if returncode == 0 else "FAILED"
                failed = failed or returncode != 0
                print(f"{job.status}: {job.experiment}", flush=True)
                _write_index(index_path, jobs)
            active = remaining
            if failed and args.fail_fast and not active:
                for job in pending:
                    job.status = "SKIPPED_FAIL_FAST"
                pending.clear()
                _write_index(index_path, jobs)
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
        _write_index(index_path, jobs)
        raise

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
