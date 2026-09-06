"""Six-model fixed-budget comparison of learner opponent difficulty."""

import argparse
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile

from lobby_032 import digest, dump, evaluate, package, read
from run_diversity_009 import require_runtime, size, verify_files


def jobs(spec):
    pairs = [(s, arm) for s in spec["seeds"] for arm in ("control", "candidate")]
    return [(mode, s, arm) for mode in ("train", "evaluate") for s, arm in pairs] + [("baseline", 7, "control")]


def identity(job):
    mode, seed, arm = job
    return f"{mode}-s{seed:03d}-{arm}"


def training_config(spec, seed, arm):
    setting = spec["arms"][arm]
    overrides = setting if isinstance(setting, dict) else {"heuristic_opponents": setting}
    return spec["train"] | overrides | {"seed": seed}


def phase(root, index, spec_name="lobby035_comparison.json"):
    spec = read(root / "configs" / spec_name)
    require_runtime(spec)
    verify_files(root)
    sys.path.insert(0, str(root / "src"))
    import torch
    from hearthstone_ai.lobby_training import train
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print(json.dumps({"torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads()}), flush=True)
    job = jobs(spec)[index]
    mode, seed, arm = job
    cfg = training_config(spec, seed, arm)
    output = root / "outputs" / identity(job)
    if mode == "train":
        result = train(cfg, output)
        with zipfile.ZipFile(output / result["checkpoint"] / "model.zip") as z:
            data = json.loads(z.read("data"))
        expected_updates = cfg["max_steps"] // cfg["n_steps"] * cfg["n_epochs"]
        if result["num_timesteps"] != cfg["max_steps"] or data["_n_updates"] != expected_updates:
            raise RuntimeError("Incomplete fixed-budget training")
    else:
        training = root / "outputs" / identity(("train", seed, arm))
        labels = ("initial", "final") if mode == "evaluate" else ("heuristic", "random")
        result = evaluate(root, spec, training=training, output=output, labels=labels, allow_opponent_shift=True,
                          observation_scale=cfg.get("observation_scale", "raw"))
        dump(output / "comparison-context.json", {"training_heuristic_opponents": cfg["heuristic_opponents"],
                                                   "evaluation_heuristic_opponents": 7,
                                                   "observation_scale": cfg.get("observation_scale", "raw"),
                                                   "explicit_opponent_shift_allowed": True})
    verify_files(root)
    return result


def run(root, spec_name="lobby035_comparison.json", *, job_list=None, phase_script="lobby_035.py"):
    spec = read(root / "configs" / spec_name)
    python = str(require_runtime(spec))
    verify_files(root)
    output, returns = root / "outputs", root / "returns"
    if output.exists() or returns.exists():
        raise FileExistsError("Preserve previous attempt")
    if shutil.disk_usage(root).free < 2 * spec["limits"]["disk_bytes"]:
        raise RuntimeError("Insufficient free disk")
    output.mkdir()
    state = {"status": "running", "completed_phases": []}
    active = None
    started = time.monotonic()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for index, job in enumerate(jobs(spec) if job_list is None else job_list):
            verify_files(root)
            name = identity(job)
            phase_spec = output / f"spec-{name}.json"
            control = output / f"control-{name}"
            dump(phase_spec, {"commands": [[python, "-B", str(root / "scripts" / phase_script), "phase", "--index", str(index), "--spec", spec_name]],
                              "timeout_seconds": spec["limits"]["phase_seconds"], "rss_limit_bytes": spec["limits"]["rss_bytes"]})
            phase_started = time.monotonic()
            with (output / f"driver-{name}.log").open("wb") as log:
                active = subprocess.Popen([python, "-B", str(root / "scripts/supervise.py"), "--spec", str(phase_spec),
                                           "--output", str(control)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while active.poll() is None:
                    if time.monotonic() - phase_started > spec["limits"]["phase_seconds"] + 2 or size(output) + size(returns) >= spec["limits"]["disk_bytes"]:
                        raise RuntimeError("Outer comparison limit")
                    time.sleep(.2)
            if active.returncode != 0 or read(control / "result.json")["status"] != "completed":
                raise RuntimeError(f"Failed phase {name}")
            active = None
            verify_files(root)
            paths = [p for folder in (control, output / name) for p in folder.rglob("*") if p.is_file()]
            returns.mkdir(exist_ok=True)
            archive = returns / f"{name}.zip"
            with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as z:
                for p in paths:
                    z.write(p, p.relative_to(root).as_posix())
                z.writestr("FILES.json", json.dumps({p.relative_to(root).as_posix(): digest(p) for p in paths}))
            dump(archive.with_suffix(".json"), {"sha256": digest(archive), "bytes": archive.stat().st_size})
            if size(output) + size(returns) >= spec["limits"]["disk_bytes"]:
                raise RuntimeError("Post-return disk limit")
            state["completed_phases"].append(name)
            state["elapsed_seconds"] = time.monotonic() - started
            dump(output / "driver-status.json", state)
            print(json.dumps(state), flush=True)
        state["status"] = "completed"
    except BaseException:
        state["status"] = "failed"
        raise
    finally:
        try:
            if active is not None and active.poll() is None:
                active.send_signal(signal.SIGTERM)
                active.wait(timeout=5)
        except subprocess.TimeoutExpired:
            state["status"] = "failed"
            state["cleanup_error"] = "Supervisor did not exit after TERM; inspect live process before any further run"
            raise
        finally:
            state["elapsed_seconds"] = time.monotonic() - started
            dump(output / "driver-status.json", state)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    return state


def decide(rows):
    if sorted(rows) != [7, 17, 27]:
        raise ValueError("All three training seeds required")
    paired = [rows[s]["control"]["final"]["mean_rank"] - rows[s]["candidate"]["final"]["mean_rank"] for s in sorted(rows)]
    progress = [rows[s]["candidate"]["initial"]["mean_rank"] - rows[s]["candidate"]["final"]["mean_rank"] for s in sorted(rows)]
    top4_delta = sum(rows[s]["candidate"]["final"]["top4_fraction_ranked"] - rows[s]["control"]["final"]["top4_fraction_ranked"] for s in rows) / 3
    no_truncation = all(r["truncated"] == 0 for row in rows.values() for arm in row.values() for r in arm.values())
    keep = sum(paired)/3 >= .5 and sum(v > 0 for v in paired) >= 2 and sum(progress)/3 >= .5 and top4_delta >= 0 and no_truncation
    return {"paired_rank_improvements": paired, "mean_paired_rank_improvement": sum(paired)/3,
            "candidate_initial_final_improvements": progress, "mean_candidate_progress": sum(progress)/3,
            "top4_delta": top4_delta, "positive_seeds": sum(v > 0 for v in paired),
            "no_truncation": no_truncation, "candidate_keep": keep}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["package", "run", "phase"])
    parser.add_argument("--name", default="lobby-035-local-r1")
    parser.add_argument("--index", type=int)
    parser.add_argument("--spec", choices=["lobby035_comparison.json", "lobby037_comparison.json", "lobby039_comparison.json"], default="lobby035_comparison.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = (package(root, args.name, [root / "experiments/LOBBY_035_COMPARISON.md"]) if args.mode == "package"
              else run(root, args.spec) if args.mode == "run" else phase(root, args.index, args.spec))
    print(json.dumps(result, allow_nan=False))
