"""Execute one frozen 019 phase in its declared Linux venv; preserve outputs."""

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

from run_diversity_009 import require_runtime, size, verify_files


PHASE_OUTPUTS = {"train": "train", "evaluate": "evaluate", "baseline": "baseline"}
SUPERVISOR_TIMEOUT_SECONDS = 180
RSS_LIMIT_BYTES = 2 * 1024**3


def require_fresh_phase(root: Path, phase: str) -> None:
    phase_output = PHASE_OUTPUTS[phase]
    paths = [
        root / "outputs" / ("control-" + phase),
        root / "outputs" / phase_output,
        root / f"return-{phase}.zip",
    ]
    if any(path.exists() for path in paths):
        raise RuntimeError("Phase already attempted; preserve it, do not restart")


def completed_dependency(root: Path, dependency: str) -> None:
    path = root / "outputs" / f"control-{dependency}" / "result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["status"] != "completed":
        raise RuntimeError(f"Incomplete predecessor: {dependency}")


def bounded_zip(root: Path, output: Path, archive: Path, limit: int) -> dict[str, int | str]:
    output_bytes = size(output)
    previous_archive_bytes = sum(
        path.stat().st_size for path in root.glob("return-*.zip") if path != archive
    )
    if output_bytes + previous_archive_bytes > limit:
        raise RuntimeError("Output plus existing ZIPs exceeds disk budget before ZIP creation")
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as zf:
        for name in json.loads((root / "FILES.json").read_text(encoding="utf-8")):
            zf.write(root / name, name)
        zf.write(root / "FILES.json", "FILES.json")
        for path in sorted(output.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(root).as_posix())
    archive_bytes = archive.stat().st_size
    all_archive_bytes = sum(path.stat().st_size for path in root.glob("return-*.zip"))
    if output_bytes + all_archive_bytes > limit:
        raise RuntimeError("Output plus all return ZIPs exceeds disk budget")
    return {
        "archive": str(archive),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "output_bytes": output_bytes,
        "archive_bytes": archive_bytes,
        "all_return_zip_bytes": all_archive_bytes,
    }


def run_phase(root: Path, phase_name: str) -> dict[str, object]:
    verify_files(root)
    plan = json.loads((root / "execution.json").read_text(encoding="utf-8"))
    python = require_runtime(plan)
    phase = plan["phases"][phase_name]
    output = root / "outputs"
    report_dir = output / ("control-" + phase_name)
    archive = root / f"return-{phase_name}.zip"
    require_fresh_phase(root, phase_name)
    for dependency in phase["requires"]:
        completed_dependency(root, dependency)
    if shutil.disk_usage(root).free < plan["minimum_free_bytes"]:
        raise RuntimeError("Insufficient free disk; never clean old results")
    report_dir.mkdir(parents=True, exist_ok=False)
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
    commands = [
        [arg.replace("{root}", str(root)).replace("{python}", str(python)) for arg in argv]
        for argv in phase["commands"]
    ]
    spec = {
        "timeout_seconds": SUPERVISOR_TIMEOUT_SECONDS,
        "rss_limit_bytes": RSS_LIMIT_BYTES,
        "commands": commands,
    }
    spec_path = report_dir / "supervisor-spec.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    argv = [
        str(python),
        "-B",
        str(root / "scripts/supervise.py"),
        "--spec",
        str(spec_path),
        "--output",
        str(report_dir / "supervisor"),
    ]
    result: dict[str, object] = {
        "phase": phase_name,
        "started_unix": time.time(),
        "status": "running",
        "argv": argv,
        "disk_limit_bytes": plan["output_disk_limit_bytes"],
        "supervisor_timeout_seconds": SUPERVISOR_TIMEOUT_SECONDS,
        "rss_limit_bytes": RSS_LIMIT_BYTES,
        "cpu_threads": 1,
    }
    child = None
    try:
        with (report_dir / "entry.log").open("wb") as log:
            child = subprocess.Popen(
                argv,
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            result["supervisor_pid"] = child.pid
            (report_dir / "live.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            peak = 0
            while child.poll() is None:
                peak = max(peak, size(output))
                previous_archives = sum(
                    path.stat().st_size for path in root.glob("return-*.zip")
                )
                if peak + previous_archives > plan["output_disk_limit_bytes"]:
                    result["status"] = "disk_exceeded"
                    child.send_signal(signal.SIGTERM)
                    break
                time.sleep(0.5)
            result["returncode"] = child.wait(timeout=10)
            result["peak_sampled_output_bytes"] = max(peak, size(output))
        supervision = json.loads((report_dir / "supervisor/result.json").read_text(encoding="utf-8"))
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
        result["ended_unix"] = time.time()
        (report_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        if not archive.exists():
            try:
                result["return_package"] = bounded_zip(
                    root, output, archive, plan["output_disk_limit_bytes"]
                )
            finally:
                (report_dir / "result.json").write_text(
                    json.dumps(result, indent=2), encoding="utf-8"
                )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=tuple(PHASE_OUTPUTS))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = run_phase(root, args.phase)
    archive = result.get("return_package", {})
    print(json.dumps({"result": result, **archive}, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
