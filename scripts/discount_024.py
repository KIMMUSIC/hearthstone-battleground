"""Freeze, execute and audit the single-use local discount comparison."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile

from discount_decision import decide
from evaluation_evidence import audit_report
from package_campaign_021 import collect_paths
from run_diversity_009 import verify_files


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_digest(path):
    return hashlib.sha256(json.dumps(read(path), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def jobs(spec):
    if spec["seeds"] != [7, 17, 27] or spec["arms"] != {"control": 0.99, "candidate": 1.0}:
        raise ValueError("Predeclared seeds and arms required")
    pairs = [(seed, arm) for seed in spec["seeds"] for arm in ("control", "candidate")]
    return [("train", s, a, "final") for s, a in pairs] + [
        ("evaluate", s, a, point) for s, a in pairs for point in ("initial", "final")]


def identity(job):
    mode, seed, arm, point = job
    return f"{mode}-s{seed:03d}-{arm}-{point}"


def footprint(root):
    return sum(p.stat().st_size for folder in ("outputs", "returns")
               for p in (root / folder).rglob("*") if p.is_file())


def checkpoint(path, steps, gamma):
    import torch
    with zipfile.ZipFile(path / "model.zip") as z:
        if z.testzip() is not None:
            raise ValueError("Corrupt model ZIP")
        data = json.loads(z.read("data"))
        if (data["num_timesteps"] != steps or data["_n_updates"] != steps // 32 * 4
                or data["gamma"] != gamma):
            raise ValueError("Actual model counters or gamma mismatch")
        policy = torch.load(io.BytesIO(z.read("policy.pth")), weights_only=True, map_location="cpu")
        if not all(torch.isfinite(t).all() for t in policy.values()):
            raise ValueError("Non-finite policy")
    return policy


def package(root, name):
    destination = root / "handoff" / name
    if destination.parent.resolve() != (root / "handoff").resolve():
        raise ValueError("Package name must be a direct child")
    if destination.exists() or destination.with_suffix(".zip").exists():
        raise FileExistsError("Preserve prior package")
    paths = collect_paths(root) + [root / "experiments/DISCOUNT_024.md"]
    destination.mkdir()
    for path in paths:
        target = destination / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    for source, name in (("runs/expansion-019-local-r1/outputs/baseline/heuristic.json", "heuristic020.json"),
                         ("runs/expansion-019-local-r1/FILES.json", "baseline019-FILES.json")):
        target = destination / "references" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(root / source, target)
    files = {p.relative_to(destination).as_posix(): digest(p)
             for p in destination.rglob("*") if p.is_file()}
    dump(destination / "FILES.json", files)
    archive = destination.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as bundle:
        for p in destination.rglob("*"):
            if p.is_file():
                bundle.write(p, p.relative_to(destination).as_posix())
    result = {"sha256": digest(archive), "bytes": archive.stat().st_size, "archive": str(archive)}
    dump(destination.with_suffix(".manifest.json"), result)
    return result


def phase(root, index):
    spec = read(root / "configs/discount024_campaign.json")
    mode, seed, arm, point = job = jobs(spec)[index]
    sys.path.insert(0, str(root / "src"))
    import torch
    from hearthstone_ai.artifacts import compatibility_signature
    from hearthstone_ai.training import train
    from hearthstone_ai.evaluation import evaluate as evaluate_policy

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    verify_files(root)
    output = root / "outputs" / identity(job)
    if output.exists():
        raise FileExistsError(output)
    signature = compatibility_signature()
    if mode == "train":
        result = train(spec["train"] | {"seed": seed, "gamma": spec["arms"][arm]}, output)
        if result["additional_steps"] != spec["train"]["max_steps"]:
            raise RuntimeError("Incomplete training")
        checkpoint(output / result["checkpoint"], spec["train"]["max_steps"], spec["arms"][arm])
        dump(output / "phase-result.json", {"summary": result, "compatibility": signature})
    else:
        training = root / "outputs" / identity(("train", seed, arm, "final"))
        summary = read(training / "phase-result.json")["summary"]
        model_dir = training / summary["initial_checkpoint" if point == "initial" else "checkpoint"]
        before = digest(model_dir / "model.zip")
        cfg = read(root / "configs/expansion020_eval.json")
        report = evaluate_policy(policy="model", seeds=cfg["seeds"],
                                 opponent_path=root / cfg["opponent_path"], checkpoint=model_dir,
                                 max_turns=cfg["max_turns"], max_actions=cfg["max_actions"], trace=True)
        if before != digest(model_dir / "model.zip"):
            raise RuntimeError("Evaluation mutated model")
        trace_audit = audit_report(report, cfg["seeds"])
        dump(output / "report.json", report)
        dump(output / "phase-result.json", {"model_sha256": before, "compatibility": signature,
                                            "trace_audit": trace_audit})


def run(root):
    spec = read(root / "configs/discount024_campaign.json")
    if not sys.platform.startswith("linux") or Path(sys.executable).resolve() != Path(spec["python"]).resolve():
        raise RuntimeError("Dedicated Linux runtime required")
    verify_files(root)
    if (root / "outputs").exists() or (root / "returns").exists():
        raise FileExistsError("Campaign already attempted; no automatic retry")
    if shutil.disk_usage(root).free < 2 * spec["limits"]["disk_bytes"]:
        raise RuntimeError("Insufficient free disk")
    (root / "outputs").mkdir()
    started = time.monotonic()
    elapsed = {"train": 0.0, "evaluate": 0.0}
    state = {"status": "running", "completed_phases": 0, "started_unix": time.time()}
    active = None

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for index, job in enumerate(jobs(spec)):
            mode = job[0]
            phase_started = time.monotonic()
            timeout = min(spec["limits"]["phase_seconds"],
                          spec["limits"][f"{mode}_seconds"] - elapsed[mode],
                          spec["limits"]["total_seconds"] - (phase_started - started))
            if timeout <= 0 or footprint(root) >= spec["limits"]["disk_bytes"]:
                raise RuntimeError("Campaign budget exhausted")
            verify_files(root)
            name = identity(job)
            state["active_phase"] = name
            dump(root / "outputs/driver-status.json", state)
            control = root / "outputs" / f"control-{name}"
            command = [spec["python"], "-B", str(root / "scripts/discount_024.py"), "phase", "--index", str(index)]
            supervisor_spec = {"commands": [command], "timeout_seconds": timeout,
                               "rss_limit_bytes": spec["limits"]["rss_bytes"]}
            spec_path = root / "outputs" / f"spec-{name}.json"
            dump(spec_path, supervisor_spec)
            with (root / "outputs" / f"driver-{name}.log").open("wb") as log:
                active = subprocess.Popen([spec["python"], "-B", str(root / "scripts/supervise.py"),
                                           "--spec", str(spec_path), "--output", str(control)],
                                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while active.poll() is None:
                    if (time.monotonic() - phase_started > timeout + 2
                            or footprint(root) >= spec["limits"]["disk_bytes"]):
                        active.send_signal(signal.SIGTERM)
                        active.wait(timeout=5)
                        raise RuntimeError("Outer campaign limit")
                    time.sleep(0.2)
            result = read(control / "result.json")
            if active.returncode != 0 or result["status"] != "completed":
                raise RuntimeError(f"Phase failed: {name}")
            active = None
            verify_files(root)
            archive = root / "returns" / f"{name}.zip"
            archive.parent.mkdir(exist_ok=True)
            paths = [p for folder in (control, root / "outputs" / name) for p in folder.rglob("*") if p.is_file()]
            with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as z:
                for path in paths:
                    z.write(path, path.relative_to(root).as_posix())
                z.writestr("FILES.json", json.dumps({p.relative_to(root).as_posix(): digest(p) for p in paths}))
            elapsed[mode] += time.monotonic() - phase_started
            if (elapsed[mode] > spec["limits"][f"{mode}_seconds"]
                    or time.monotonic() - started > spec["limits"]["total_seconds"]
                    or footprint(root) >= spec["limits"]["disk_bytes"]):
                raise RuntimeError("Post-archive budget violation")
            dump(archive.with_suffix(".json"), {"sha256": digest(archive), "bytes": archive.stat().st_size})
            state.update(completed_phases=index + 1, elapsed_seconds=time.monotonic() - started,
                         phase_totals=elapsed.copy())
            dump(root / "outputs/driver-status.json", state)
            print(json.dumps(state), flush=True)
        state["status"] = "completed"
    except BaseException:
        state["status"] = "failed"
        if active is not None and active.poll() is None:
            active.send_signal(signal.SIGTERM)
            active.wait(timeout=5)
        raise
    finally:
        state["elapsed_seconds"] = time.monotonic() - started
        dump(root / "outputs/driver-status.json", state)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def audit(root):
    import torch
    verify_files(root)
    spec = read(root / "configs/discount024_campaign.json")
    state = read(root / "outputs/driver-status.json")
    if state["status"] != "completed" or state["completed_phases"] != 18:
        raise ValueError("Campaign incomplete")
    for mode in ("train", "evaluate"):
        if state["phase_totals"][mode] > spec["limits"][f"{mode}_seconds"]:
            raise ValueError("Cumulative budget violation")
    if state["elapsed_seconds"] > spec["limits"]["total_seconds"] or footprint(root) >= spec["limits"]["disk_bytes"]:
        raise ValueError("Total budget violation")
    if (root / "experiments/reserved").exists():
        raise ValueError("Reserved final inputs must be absent")
    baseline = read(root / "references/heuristic020.json")
    cfg = read(root / "configs/expansion020_eval.json")
    audit_report(baseline, cfg["seeds"])
    if (baseline["mean_reward"] != 3.4 or baseline["opponent_set_sha256"] != content_digest(root / cfg["opponent_path"])
            or baseline["max_actions"] != cfg["max_actions"] or baseline["max_turns"] != cfg["max_turns"]):
        raise ValueError("Baseline evaluation contract differs")
    for path, sha in read(root / "references/baseline019-FILES.json").items():
        if (path.startswith("src/") and not path.endswith("/training.py")) or path.startswith("data/") or path == "docs/RULES.md":
            if digest(root / path) != sha:
                raise ValueError("Baseline engine changed beyond gamma training setting")
    reports, initial_policies, model_hashes = {}, {}, {}
    signature = None
    peak_rss = 0
    for job in jobs(spec):
        mode, seed, arm, point = job
        name = identity(job)
        output = root / "outputs" / name
        control = read(root / "outputs" / f"control-{name}" / "result.json")
        if control["status"] != "completed" or len(control["commands"]) != 1:
            raise ValueError("Supervisor failed")
        command = control["commands"][0]
        peak_rss = max(peak_rss, command["peak_sampled_rss"])
        if (command["returncode"] != 0 or command["remaining_pids"]
                or command["peak_sampled_rss"] >= spec["limits"]["rss_bytes"]
                or control["elapsed_seconds"] > spec["limits"]["phase_seconds"]):
            raise ValueError("Supervisor invariant failed")
        archive = root / "returns" / f"{name}.zip"
        receipt = read(archive.with_suffix(".json"))
        if digest(archive) != receipt["sha256"] or archive.stat().st_size != receipt["bytes"]:
            raise ValueError("Archive receipt mismatch")
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:
                raise ValueError("Bad ZIP CRC")
            entries = json.loads(z.read("FILES.json"))
            expected = {p.relative_to(root).as_posix() for folder in
                        (output, root / "outputs" / f"control-{name}") for p in folder.rglob("*") if p.is_file()}
            if set(entries) != expected or set(z.namelist()) != expected | {"FILES.json"}:
                raise ValueError("Archive file set mismatch")
            for path, sha in entries.items():
                if digest(root / path) != sha or z.read(path) != (root / path).read_bytes():
                    raise ValueError("Archived output changed")
        result = read(output / "phase-result.json")
        signature = signature or result["compatibility"]
        if result["compatibility"] != signature:
            raise ValueError("Mixed source contracts")
        if mode == "train":
            summary = result["summary"]
            if summary["status"] != "completed" or summary["additional_steps"] != 8192:
                raise ValueError("Actual training amount mismatch")
            for label, steps in (("initial_checkpoint", 0), ("checkpoint", 8192)):
                path = output / summary[label]
                metadata = read(path / "metadata.json")
                expected_config = spec["train"] | {"seed": seed, "gamma": spec["arms"][arm]}
                if metadata["config"] != expected_config or metadata["compatibility"] != signature:
                    raise ValueError("Model metadata contract mismatch")
                if metadata["num_timesteps"] != steps or metadata["additional_steps"] != steps:
                    raise ValueError("Model metadata counters mismatch")
                policy = checkpoint(path, steps, spec["arms"][arm])
                if steps == 0:
                    initial_policies[seed, arm] = policy
                model_hashes[seed, arm, "initial" if steps == 0 else "final"] = digest(path / "model.zip")
        else:
            report = read(output / "report.json")
            cfg = read(root / "configs/expansion020_eval.json")
            audit_report(report, cfg["seeds"])
            if report["opponent_set_sha256"] != content_digest(root / cfg["opponent_path"]):
                raise ValueError("Development opponent mismatch")
            if result["model_sha256"] != model_hashes[seed, arm, point]:
                raise ValueError("Evaluation model mismatch")
            reports.setdefault(str(seed), {}).setdefault(arm, {})[point] = {
                "mean_reward": report["mean_reward"], "survival_rate": report["survival_rate"]}
    for seed in spec["seeds"]:
        left, right = initial_policies[seed, "control"], initial_policies[seed, "candidate"]
        if set(left) != set(right) or not all(torch.equal(left[k], right[k]) for k in left):
            raise ValueError("Paired initial policies differ")
    result = {"status": "passed", "training_steps": 49152, "evaluation_games": 1200,
              "peak_sampled_rss": peak_rss, "output_and_return_bytes": footprint(root),
              "compatibility": signature, "reports": reports, "decision": decide(reports, baseline["mean_reward"]),
              "driver": state, "paired_initial_policies_equal": True}
    dump(root / "audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("package", "run", "phase", "audit"))
    parser.add_argument("--name", default="discount-024-local-r1")
    parser.add_argument("--index", type=int)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve() if args.root else Path(__file__).resolve().parents[1]
    if args.mode == "package":
        print(json.dumps(package(root, args.name)))
    elif args.mode == "run":
        run(root)
    elif args.mode == "audit":
        print(json.dumps(audit(root)["decision"]))
    else:
        phase(root, args.index)


if __name__ == "__main__":
    main()
