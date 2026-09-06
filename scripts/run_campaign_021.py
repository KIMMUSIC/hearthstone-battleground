"""Execute one frozen 021 campaign phase with cumulative budgets and incremental returns."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import zipfile
from typing import Any

from run_diversity_009 import require_runtime, size, verify_files


SUPERVISOR_TIMEOUT_SECONDS = 180
RSS_LIMIT_BYTES = 2 * 1024**3


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def phase_id_train(seed: int, segment: int) -> str:
    return f"train-s{seed:03d}-seg{segment:02d}"


def phase_id_evaluate(seed: int, milestone: int) -> str:
    return f"eval-s{seed:03d}-step{milestone:06d}"


def phase_output(root: Path, phase_id: str) -> Path:
    if phase_id.startswith("train-"):
        parts = phase_id.split("-")
        return root / "outputs" / f"seed-{int(parts[1][1:]):03d}" / f"segment-{int(parts[2][3:]):02d}"
    parts = phase_id.split("-")
    return root / "outputs" / f"seed-{int(parts[1][1:]):03d}" / f"eval-{int(parts[2][4:]):06d}"


def require_fresh_phase(root: Path, phase_id: str) -> None:
    paths = [
        root / "outputs" / f"control-{phase_id}",
        phase_output(root, phase_id),
        root / "returns" / f"return-{phase_id}.zip",
    ]
    if any(path.exists() for path in paths):
        raise RuntimeError("Phase already attempted; preserve it, do not restart")


def require_no_incomplete_controls(root: Path) -> None:
    for control in sorted((root / "outputs").glob("control-*")):
        result_path = control / "result.json"
        if not result_path.exists():
            raise RuntimeError(f"Incomplete prior control directory: {control.name}")
        result = read_json(result_path)
        if result.get("status") != "completed":
            raise RuntimeError(f"Incomplete prior control directory: {control.name}")


def completed_control(root: Path, phase_id: str) -> None:
    result = read_json(root / "outputs" / f"control-{phase_id}" / "result.json")
    if result.get("status") != "completed":
        raise RuntimeError(f"Incomplete predecessor: {phase_id}")


def require_dependencies(root: Path, *, mode: str, seed: int, segment: int | None, milestone: int | None) -> None:
    if mode == "train" and segment and segment > 1:
        completed_control(root, phase_id_train(seed, segment - 1))
    if mode == "evaluate":
        if milestone == 0:
            completed_control(root, phase_id_train(seed, 1))
        elif milestone == 49152:
            completed_control(root, phase_id_train(seed, 6))
        elif milestone == 100000:
            completed_control(root, phase_id_train(seed, 13))
        else:
            raise ValueError("Unsupported evaluation milestone")


def prior_elapsed(root: Path) -> dict[str, float]:
    totals = {"train": 0.0, "evaluate": 0.0, "all": 0.0}
    for path in sorted((root / "outputs").glob("control-*/result.json")):
        result = read_json(path)
        elapsed = float(result.get("end_to_end_elapsed_seconds", result.get("elapsed_seconds", 0.0)))
        phase_id = path.parent.name.removeprefix("control-")
        kind = "train" if phase_id.startswith("train-") else "evaluate" if phase_id.startswith("eval-") else ""
        if kind:
            totals[kind] += elapsed
            totals["all"] += elapsed
    return totals


def cumulative_limit(plan: dict, mode: str) -> float:
    limits = plan["campaign_spec"]["limits"]
    if mode == "train":
        return float(limits["campaign_train_wall_seconds"])
    return float(limits["campaign_dev_eval_wall_seconds"])


def total_limit(plan: dict) -> float:
    return float(plan["campaign_spec"]["limits"]["campaign_total_wall_seconds"])


def driver_wall_elapsed(root: Path) -> float | None:
    path = root / "outputs" / "driver-status.json"
    if not path.exists():
        return None
    started = read_json(path).get("started_unix")
    if type(started) not in (int, float):
        raise ValueError("driver-status.json must contain numeric started_unix")
    return max(0.0, time.time() - float(started))


def budget_status(
    plan: dict,
    mode: str,
    prior: dict[str, float],
    elapsed: float,
    campaign_wall_elapsed: float | None = None,
) -> str | None:
    if prior[mode] + elapsed > cumulative_limit(plan, mode):
        return "cumulative_wall_exceeded"
    total_elapsed = campaign_wall_elapsed if campaign_wall_elapsed is not None else prior["all"] + elapsed
    if total_elapsed > total_limit(plan):
        return "campaign_total_wall_exceeded"
    return None


def validate_invocation(plan: dict[str, Any], args: argparse.Namespace) -> None:
    spec = plan["campaign_spec"]
    if args.mode == "train":
        if args.seed not in spec["training_seeds"]:
            raise ValueError("Seed must be one of the campaign training seeds")
        segments = spec["segment_additional_steps"]
        if not 1 <= args.segment <= len(segments):
            raise ValueError("Segment index out of range")
    else:
        if args.seed not in spec["training_seeds"]:
            raise ValueError("Seed must be one of the campaign training seeds")
        if args.milestone not in spec["evaluation_milestones"]:
            raise ValueError("Unsupported evaluation milestone")


def acquire_campaign_lock(root: Path, phase_id: str) -> int:
    output = root / "outputs"
    output.mkdir(exist_ok=True)
    lock = output / ".campaign.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError("Campaign phase already running; preserve CPU1 serial execution") from error
    os.write(
        descriptor,
        json.dumps({"phase_id": phase_id, "pid": os.getpid(), "started_unix": time.time()}).encode(),
    )
    return descriptor


def release_campaign_lock(root: Path, descriptor: int | None) -> None:
    if descriptor is None:
        return
    os.close(descriptor)
    (root / "outputs" / ".campaign.lock").unlink(missing_ok=True)


def archive_phase(root: Path, phase_id: str, limit: int) -> dict[str, int | str]:
    returns = root / "returns"
    returns.mkdir(exist_ok=True)
    archive = returns / f"return-{phase_id}.zip"
    phase_dir = phase_output(root, phase_id)
    control_dir = root / "outputs" / f"control-{phase_id}"
    output_bytes = size(phase_dir) + size(control_dir)
    prior_archives = sum(path.stat().st_size for path in returns.glob("return-*.zip"))
    if size(root / "outputs") + prior_archives > limit:
        raise RuntimeError("Output plus existing return ZIPs exceeds disk budget before ZIP creation")
    manifest = {
        "campaign_id": "campaign-021",
        "phase_id": phase_id,
        "phase_output": str(phase_dir.relative_to(root)),
        "control_output": str(control_dir.relative_to(root)),
        "incremental_only": True,
        "input_files_not_rebundled": True,
    }
    manifest_path = control_dir / "return-manifest.json"
    write_json(manifest_path, manifest)
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted((phase_dir, control_dir)):
            for child in path.rglob("*"):
                if child.is_file():
                    zf.write(child, child.relative_to(root).as_posix())
        for name in ("execution.json", "FILES.json"):
            zf.write(root / name, name)
        progress = root / "outputs" / "progress.json"
        if progress.exists():
            zf.write(progress, progress.relative_to(root).as_posix())
    archive_bytes = archive.stat().st_size
    all_archives = sum(path.stat().st_size for path in returns.glob("return-*.zip"))
    if size(root / "outputs") + all_archives > limit:
        raise RuntimeError("Output plus all return ZIPs exceeds disk budget")
    return {
        "archive": str(archive),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "phase_output_bytes": output_bytes,
        "archive_bytes": archive_bytes,
        "all_return_zip_bytes": all_archives,
    }


def command_for(root: Path, python: Path, source_sha: str, args: argparse.Namespace) -> tuple[str, list[str]]:
    if args.mode == "train":
        phase_id = phase_id_train(args.seed, args.segment)
        command = [
            str(python),
            "-B",
            str(root / "scripts/campaign_021.py"),
            "--checkout",
            str(root),
            "--output",
            str(root / "outputs"),
            "--expected-source-sha256",
            source_sha,
            "train",
            "--seed",
            str(args.seed),
            "--segment",
            str(args.segment),
        ]
    else:
        phase_id = phase_id_evaluate(args.seed, args.milestone)
        command = [
            str(python),
            "-B",
            str(root / "scripts/campaign_021.py"),
            "--checkout",
            str(root),
            "--output",
            str(root / "outputs"),
            "--expected-source-sha256",
            source_sha,
            "evaluate",
            "--seed",
            str(args.seed),
            "--milestone",
            str(args.milestone),
        ]
    return phase_id, command


def run_phase(root: Path, args: argparse.Namespace) -> dict[str, object]:
    verify_files(root)
    plan = read_json(root / "execution.json")
    python = require_runtime(plan)
    validate_invocation(plan, args)
    phase_id, command = command_for(root, python, plan["source_sha256"], args)
    require_fresh_phase(root, phase_id)
    require_no_incomplete_controls(root)
    require_dependencies(root, mode=args.mode, seed=args.seed, segment=getattr(args, "segment", None), milestone=getattr(args, "milestone", None))
    if shutil.disk_usage(root).free < plan["minimum_free_bytes"]:
        raise RuntimeError("Insufficient free disk; never clean old results")
    totals = prior_elapsed(root)
    preflight_budget = budget_status(plan, args.mode, totals, 0.0, driver_wall_elapsed(root))
    if preflight_budget:
        raise RuntimeError(f"Campaign budget already exhausted: {preflight_budget}")
    limit = int(plan["output_disk_limit_bytes"])
    archive_bytes = sum(path.stat().st_size for path in (root / "returns").glob("return-*.zip"))
    if size(root / "outputs") + archive_bytes > limit:
        raise RuntimeError("Output plus return ZIPs already exceeds disk budget")
    lock_descriptor = acquire_campaign_lock(root, phase_id)
    try:
        output = root / "outputs"
        control = output / f"control-{phase_id}"
        control.mkdir(parents=True, exist_ok=False)
        spec = {
            "timeout_seconds": SUPERVISOR_TIMEOUT_SECONDS,
            "rss_limit_bytes": RSS_LIMIT_BYTES,
            "commands": [command],
        }
        spec_path = control / "supervisor-spec.json"
        write_json(spec_path, spec)
        argv = [
            str(python),
            "-B",
            str(root / "scripts/supervise.py"),
            "--spec",
            str(spec_path),
            "--output",
            str(control / "supervisor"),
        ]
        env = os.environ.copy()
        env.update(
            {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(root / "src"),
            }
        )
        started = time.monotonic()
        result: dict[str, object] = {
            "phase_id": phase_id,
            "mode": args.mode,
            "started_unix": time.time(),
            "status": "running",
            "argv": argv,
            "cpu_threads": 1,
            "supervisor_timeout_seconds": SUPERVISOR_TIMEOUT_SECONDS,
            "rss_limit_bytes": RSS_LIMIT_BYTES,
            "prior_elapsed_seconds": totals,
        }
        child = None
        try:
            with (control / "entry.log").open("wb") as log:
                child = subprocess.Popen(
                    argv,
                    cwd=root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                result["supervisor_pid"] = child.pid
                write_json(control / "live.json", result)
                peak = 0
                while child.poll() is None:
                    elapsed = time.monotonic() - started
                    peak = max(peak, size(output))
                    archive_bytes = sum(
                        path.stat().st_size for path in (root / "returns").glob("return-*.zip")
                    )
                    if peak + archive_bytes > limit:
                        result["status"] = "disk_exceeded"
                        child.send_signal(signal.SIGTERM)
                        break
                    budget = budget_status(plan, args.mode, totals, elapsed, driver_wall_elapsed(root))
                    if budget:
                        result["status"] = budget
                        child.send_signal(signal.SIGTERM)
                        break
                    time.sleep(0.5)
                result["returncode"] = child.wait(timeout=10)
                result["peak_sampled_output_bytes"] = max(peak, size(output))
                postpoll_budget = budget_status(
                    plan, args.mode, totals, time.monotonic() - started, driver_wall_elapsed(root)
                )
                if result["status"] == "running" and postpoll_budget:
                    result["status"] = postpoll_budget
            supervision = read_json(control / "supervisor/result.json")
            result["supervision"] = supervision
            if result["status"] == "running":
                result["status"] = supervision["status"]
            verify_files(root)
        except BaseException as error:
            result.update(status="failed", error=f"{type(error).__name__}: {error}")
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGTERM)
                child.wait(timeout=10)
            raise
        finally:
            result["elapsed_seconds"] = time.monotonic() - started
            result["ended_unix"] = time.time()
            write_json(control / "result.json", result)
            archive = root / "returns" / f"return-{phase_id}.zip"
            if not archive.exists():
                result["return_package"] = archive_phase(root, phase_id, limit)
            result["end_to_end_elapsed_seconds"] = time.monotonic() - started
            postarchive_budget = budget_status(
                plan,
                args.mode,
                totals,
                float(result["end_to_end_elapsed_seconds"]),
                driver_wall_elapsed(root),
            )
            if result["status"] == "completed" and postarchive_budget:
                result["status"] = postarchive_budget
            write_json(control / "result.json", result)
        return result
    finally:
        release_campaign_lock(root, lock_descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--segment", type=int, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--seed", type=int, required=True)
    evaluate.add_argument("--milestone", type=int, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = run_phase(root, args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
