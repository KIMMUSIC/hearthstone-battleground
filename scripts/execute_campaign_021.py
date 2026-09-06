"""Drive the frozen campaign once, sequentially, without retries."""

import json
from pathlib import Path
import subprocess
import time


def main():
    root = Path(__file__).resolve().parents[1]
    execution = json.loads((root / "execution.json").read_text(encoding="utf-8"))
    spec = execution["campaign_spec"]
    output = root / "outputs"
    if output.exists():
        raise FileExistsError("Campaign already started; preserve existing evidence")
    output.mkdir()
    started = time.monotonic()
    status = {"status": "running", "completed_phases": 0, "started_unix": time.time()}

    def persist():
        status["elapsed_seconds"] = time.monotonic() - started
        (output / "driver-status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    jobs = []
    for seed in spec["training_seeds"]:
        jobs.extend(["train", "--seed", str(seed), "--segment", str(i)]
                    for i in range(1, len(spec["segment_additional_steps"]) + 1))
    for seed in spec["training_seeds"]:
        jobs.extend(["evaluate", "--seed", str(seed), "--milestone", str(step)]
                    for step in spec["evaluation_milestones"])
    persist()
    try:
        for args in jobs:
            if time.monotonic() - started >= spec["limits"]["campaign_total_wall_seconds"]:
                raise TimeoutError("Whole campaign wall budget reached")
            status["active_args"] = args
            persist()
            result = subprocess.run(
                [execution["python"], "-B", str(root / "scripts/run_campaign_021.py"), *args],
                cwd=root, capture_output=True, text=True,
            )
            log = output / f"driver-phase-{status['completed_phases']:02d}.log"
            log.write_text(result.stdout + result.stderr, encoding="utf-8")
            if result.returncode:
                raise RuntimeError(f"Phase failed, preserved at {log}: exit={result.returncode}")
            returned = json.loads(result.stdout.strip().splitlines()[-1])
            if returned["status"] != "completed":
                raise RuntimeError(f"Phase incomplete: {returned['status']}")
            status["completed_phases"] += 1
            persist()
            print(json.dumps({"completed_phases": status["completed_phases"],
                              "last_phase": args, "elapsed_seconds": status["elapsed_seconds"]}), flush=True)
        status["status"] = "completed"
    except BaseException as error:
        status["status"] = "failed"
        status["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        persist()


if __name__ == "__main__":
    main()
