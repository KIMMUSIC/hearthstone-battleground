"""Freeze and supervise the bounded single-learner lobby smoke experiment."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import zipfile

from package_campaign_021 import collect_paths
from run_diversity_009 import require_runtime, size, verify_files


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(root, name, extra_paths=()):
    destination = (root / "handoff" / name).resolve()
    if destination.parent != (root / "handoff").resolve():
        raise ValueError("Package must be directly inside handoff")
    archive = destination.with_suffix(".zip")
    if destination.exists() or archive.exists():
        raise FileExistsError("Preserve previous package")
    paths = collect_paths(root) + [root / "docs/LOBBY_CONTRACT.md", root / "experiments/LOBBY_032.md"] + list(extra_paths)
    destination.mkdir()
    for path in paths:
        target = destination / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    dump(destination / "FILES.json", {p.relative_to(destination).as_posix(): digest(p)
                                     for p in destination.rglob("*") if p.is_file()})
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as z:
        for p in destination.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(destination).as_posix())
    result = {"sha256": digest(archive), "bytes": archive.stat().st_size}
    dump(destination.with_suffix(".manifest.json"), result)
    return result


def evaluate(root, spec, *, training=None, output=None, labels=("initial", "final", "heuristic"), allow_opponent_shift=False,
             observation_scale="raw"):
    import numpy as np
    from hearthstone_ai.lobby_env import LobbyEnv
    from hearthstone_ai.lobby_policies import heuristic_policy, random_policy
    from hearthstone_ai.lobby_training import load_model

    training = training or root / "outputs/train"
    summary = read(training / "status.json")
    env = LobbyEnv(learner_seat=spec["train"]["learner_seat"], max_rounds=spec["train"]["max_rounds"],
                   observation_scale=observation_scale)
    output = output or root / "outputs/evaluate"
    output.mkdir(exist_ok=False)
    reports = {}
    for label in labels:
        model = None
        model_path = None
        if label not in ("heuristic", "random"):
            checkpoint = training / summary["initial_checkpoint" if label == "initial" else "checkpoint"]
            model_path = checkpoint / "model.zip"
            before = digest(model_path)
            model = (load_model(checkpoint, env, allow_opponent_shift=True) if allow_opponent_shift
                     else load_model(checkpoint, env))

        def episode(seed, model=model, policy_name=label):
            obs, _ = env.reset(seed=seed)
            rng = random.Random(f"lobby-probe:{seed}:policy:0")
            decisions, rewards = [], []
            done = False
            while not done:
                mask = env.action_masks()
                if model is not None:
                    action = int(model.predict(obs, deterministic=True, action_masks=mask)[0])
                elif policy_name == "random":
                    action = random_policy(env.game.view(env.learner_seat), rng)
                else:
                    action = heuristic_policy(env.game.view(env.learner_seat))
                if not mask[action] or not env.observation_space.contains(obs):
                    raise AssertionError("Invalid model action or observation")
                obs, reward, terminated, truncated, info = env.step(action)
                env.game.assert_conservation()
                decisions.append(action)
                rewards.append(reward)
                done = terminated or truncated
            rank = env.game.players[env.learner_seat].rank
            if not env.observation_space.contains(obs) or not np.isfinite(rewards).all():
                raise AssertionError("Invalid terminal observation/rewards")
            expected = 0 if rank is None else (4.5 - rank) / 3.5
            if sum(rewards) != expected or (terminated and rank is None) or (truncated and rank is not None):
                raise AssertionError("Rank/reward/termination mismatch")
            return {"seed": seed, "rank": rank, "reward": sum(rewards), "terminated": terminated,
                    "truncated": truncated, "decisions": decisions, "round": env.game.round,
                    "info": info, "trace": env.game.trace}

        episodes = [episode(seed) for seed in spec["eval_seeds"]]
        replays = []
        if label == "final":
            for seed in spec["replay_seeds"]:
                repeat = episode(seed)
                if repeat != episodes[spec["eval_seeds"].index(seed)]:
                    raise AssertionError("Loaded policy replay mismatch")
                replays.append(seed)
        if model_path is not None and digest(model_path) != before:
            raise AssertionError("Evaluation mutated model")
        ranked = [e["rank"] for e in episodes if e["rank"] is not None]
        report = {"policy": label, "episodes": episodes, "replay_seeds": replays,
                  "model_sha256": before if model is not None else None,
                  "mean_rank": float(np.mean(ranked)) if ranked else None,
                  "top4_fraction_ranked": sum(r <= 4 for r in ranked) / len(ranked) if ranked else None,
                  "mean_reward": float(np.mean([e["reward"] for e in episodes])),
                  "truncated": sum(e["truncated"] for e in episodes)}
        dump(output / f"{label}.json", report)
        reports[label] = {k: v for k, v in report.items() if k != "episodes"}
    env.close()
    dump(output / "summary.json", reports)
    return reports


def phase(root, name):
    spec = read(root / "configs/lobby032_smoke.json")
    require_runtime(spec)
    verify_files(root)
    sys.path.insert(0, str(root / "src"))
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print(json.dumps({"pid": os.getpid(), "torch_threads": torch.get_num_threads(),
                      "torch_interop_threads": torch.get_num_interop_threads()}), flush=True)
    if name == "train":
        from hearthstone_ai.lobby_training import train
        result = train(spec["train"], root / "outputs/train")
        with zipfile.ZipFile(root / "outputs/train" / result["checkpoint"] / "model.zip") as z:
            counters = json.loads(z.read("data"))
        if counters["_n_updates"] <= 0 or counters["num_timesteps"] != result["num_timesteps"]:
            raise RuntimeError("Training produced no verified update")
    else:
        result = evaluate(root, spec)
    verify_files(root)
    return result


def run(root):
    spec = read(root / "configs/lobby032_smoke.json")
    python = str(require_runtime(spec))
    verify_files(root)
    if (root / "outputs").exists() or (root / "returns").exists():
        raise FileExistsError("Already attempted; no automatic retry")
    if shutil.disk_usage(root).free < 2 * spec["limits"]["disk_bytes"]:
        raise RuntimeError("Insufficient disk")
    output = root / "outputs"
    output.mkdir()
    active = None
    state = {"status": "running", "completed_phases": []}
    started = time.monotonic()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for name in ("train", "evaluate"):
            verify_files(root)
            command = [python, "-B", str(root / "scripts/lobby_032.py"), "phase", "--phase", name]
            control = output / f"control-{name}"
            spec_path = output / f"spec-{name}.json"
            dump(spec_path, {"commands": [command], "timeout_seconds": spec["limits"]["phase_seconds"],
                             "rss_limit_bytes": spec["limits"]["rss_bytes"]})
            phase_started = time.monotonic()
            with (output / f"driver-{name}.log").open("wb") as log:
                active = subprocess.Popen([python, "-B", str(root / "scripts/supervise.py"),
                                           "--spec", str(spec_path), "--output", str(control)],
                                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while active.poll() is None:
                    if (time.monotonic() - phase_started > spec["limits"]["phase_seconds"] + 2
                            or size(output) + size(root / "returns") >= spec["limits"]["disk_bytes"]):
                        active.send_signal(signal.SIGTERM)
                        active.wait(timeout=5)
                        raise RuntimeError("Outer execution budget exceeded")
                    time.sleep(0.2)
            result = read(control / "result.json")
            if active.returncode != 0 or result["status"] != "completed":
                raise RuntimeError(f"Failed phase: {name}")
            active = None
            verify_files(root)
            paths = [p for folder in (control, output / name) for p in folder.rglob("*") if p.is_file()]
            archive = root / "returns" / f"{name}.zip"
            archive.parent.mkdir(exist_ok=True)
            with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as z:
                for p in paths:
                    z.write(p, p.relative_to(root).as_posix())
                z.writestr("FILES.json", json.dumps({p.relative_to(root).as_posix(): digest(p) for p in paths}))
            dump(archive.with_suffix(".json"), {"sha256": digest(archive), "bytes": archive.stat().st_size})
            if size(output) + size(root / "returns") >= spec["limits"]["disk_bytes"]:
                raise RuntimeError("Output budget exceeded")
            state["completed_phases"].append(name)
            dump(output / "driver-status.json", state)
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
        dump(output / "driver-status.json", state)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["package", "run", "phase"])
    parser.add_argument("--name", default="lobby-032-local-r1")
    parser.add_argument("--phase", choices=["train", "evaluate"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = (package(root, args.name) if args.mode == "package" else
              run(root) if args.mode == "run" else phase(root, args.phase))
    print(json.dumps(result, allow_nan=False))
