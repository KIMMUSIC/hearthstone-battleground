"""Audit frozen lobby smoke counters, archives, and every evaluation action trace."""

import argparse
import io
import json
from pathlib import Path
import sys
import time
import zipfile

from lobby_032 import digest, dump, read
from run_diversity_009 import size, verify_files


def model_evidence(path):
    import torch
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:
            raise ValueError("Corrupt checkpoint")
        data = json.loads(z.read("data"))
        policy = torch.load(io.BytesIO(z.read("policy.pth")), weights_only=True, map_location="cpu")
        if not policy or not all(torch.isfinite(t).all() for t in policy.values()):
            raise ValueError("Nonfinite policy")
        if "policy.optimizer.pth" not in z.namelist():
            raise ValueError("Optimizer missing")
    return data, policy


def audit(root, output=None):
    output = output or root / "audit.json"
    if output.exists():
        raise FileExistsError("Preserve existing audit; choose a fresh output")
    started = time.monotonic()
    verify_files(root)
    spec = read(root / "configs/lobby032_smoke.json")
    driver = read(root / "outputs/driver-status.json")
    if driver["status"] != "completed" or driver["completed_phases"] != ["train", "evaluate"]:
        raise ValueError("Incomplete run")
    if (root / "experiments/reserved").exists():
        raise ValueError("Final holdout included")
    output_bytes = size(root / "outputs") + size(root / "returns")
    if output_bytes >= spec["limits"]["disk_bytes"]:
        raise ValueError("Output limit")
    peak = 0
    for phase in ("train", "evaluate"):
        control = read(root / f"outputs/control-{phase}/result.json")
        if control["status"] != "completed" or len(control["commands"]) != 1:
            raise ValueError("Supervisor failed")
        command = control["commands"][0]
        peak = max(peak, command["peak_sampled_rss"])
        if (command["returncode"] != 0 or command["remaining_pids"]
                or command["peak_sampled_rss"] >= spec["limits"]["rss_bytes"]
                or control["elapsed_seconds"] > spec["limits"]["phase_seconds"]):
            raise ValueError("Supervisor invariant failed")
        worker = json.loads((root / f"outputs/control-{phase}/000.log").read_text().splitlines()[0])
        if worker["torch_threads"] != 1 or worker["torch_interop_threads"] != 1:
            raise ValueError("Worker CPU thread setting mismatch")
        archive = root / f"returns/{phase}.zip"
        receipt = read(archive.with_suffix(".json"))
        if digest(archive) != receipt["sha256"] or archive.stat().st_size != receipt["bytes"]:
            raise ValueError("Archive hash mismatch")
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:
                raise ValueError("Archive CRC failure")
            import hashlib
            for name, sha in json.loads(z.read("FILES.json")).items():
                if hashlib.sha256(z.read(name)).hexdigest() != sha or digest(root / name) != sha:
                    raise ValueError("Archive/source member mismatch")
    sys.path.insert(0, str(root / "src"))
    import torch
    from hearthstone_ai import lobby_env
    from hearthstone_ai.lobby_policies import heuristic_policy
    from hearthstone_ai.lobby_training import load_model
    if not Path(lobby_env.__file__).resolve().is_relative_to(root / "src"):
        raise ValueError("Audit must use frozen engine")
    torch.set_num_threads(1)
    training = root / "outputs/train"
    summary = read(training / "status.json")
    manifest = read(training / "manifest.json")
    if any(manifest["config"].get(k) != v for k, v in spec["train"].items()):
        raise ValueError("Training config differs from spec")
    episodes = summary["episodes"]
    if any(type(e["seed"]) is not int or not 0 <= e["seed"] <= 9999 for e in episodes):
        raise ValueError("Training episode outside training seed pool")
    if summary["episode_count"] != sum(e["rank"] is not None for e in episodes):
        raise ValueError("Training episode count mismatch")
    if summary["episode_truncations"] != sum(e["truncated"] for e in episodes):
        raise ValueError("Training truncation count mismatch")
    initial_dir, final_dir = (training / summary[k] for k in ("initial_checkpoint", "checkpoint"))
    initial, initial_policy = model_evidence(initial_dir / "model.zip")
    final, final_policy = model_evidence(final_dir / "model.zip")
    steps = final["num_timesteps"]
    if (initial["num_timesteps"] != 0 or steps != summary["num_timesteps"]
            or not 0 < steps <= spec["train"]["max_steps"] or final["_n_updates"] <= 0
            or final["gamma"] != 1.0 or final["n_steps"] != 32 or final["batch_size"] != 32
            or final["n_epochs"] != 4 or summary["status"] != "completed"):
        raise ValueError("Actual training counters/config invalid")
    if all(torch.equal(initial_policy[k], final_policy[k]) for k in initial_policy):
        raise ValueError("No policy update")
    env = lobby_env.LobbyEnv(learner_seat=spec["train"]["learner_seat"], max_rounds=spec["train"]["max_rounds"])
    for checkpoint in (initial_dir, final_dir):
        loaded = load_model(checkpoint, env)
        if loaded.num_timesteps != (0 if checkpoint == initial_dir else steps):
            raise ValueError("Loaded model counter mismatch")
    aggregates = {}
    decision_count = 0
    for label in ("initial", "final", "heuristic"):
        report = read(root / f"outputs/evaluate/{label}.json")
        if [e["seed"] for e in report["episodes"]] != spec["eval_seeds"]:
            raise ValueError("Evaluation seed selection")
        if label != "heuristic":
            checkpoint = initial_dir if label == "initial" else final_dir
            if report["model_sha256"] != digest(checkpoint / "model.zip"):
                raise ValueError("Evaluated model mismatch")
            model = load_model(checkpoint, env)
        else:
            model = None
        ranks, rewards = [], []
        truncated_count = 0
        for recorded in report["episodes"]:
            obs, _ = env.reset(seed=recorded["seed"])
            total = 0.0
            done = False
            for action in recorded["decisions"]:
                if done or not env.action_masks()[action] or not env.observation_space.contains(obs):
                    raise ValueError("Illegal evaluation trace")
                predicted = (heuristic_policy(env.game.view(env.learner_seat)) if model is None else
                             int(model.predict(obs, deterministic=True, action_masks=env.action_masks())[0]))
                if predicted != action:
                    raise ValueError("Recorded action differs from saved policy")
                obs, reward, terminated, truncated, info = env.step(action)
                env.game.assert_conservation()
                done = terminated or truncated
                total += reward
                decision_count += 1
            rank = env.game.players[env.learner_seat].rank
            trace = json.loads(json.dumps(env.game.trace))
            if (not done or total != recorded["reward"] or rank != recorded["rank"]
                    or terminated != recorded["terminated"] or truncated != recorded["truncated"]
                    or info != recorded["info"] or trace != recorded["trace"]
                    or not env.observation_space.contains(obs)):
                raise ValueError("Evaluation replay mismatch")
            if rank is not None:
                ranks.append(rank)
            rewards.append(total)
            truncated_count += truncated
        expected = {"mean_rank": sum(ranks) / len(ranks) if ranks else None,
                    "mean_reward": sum(rewards) / len(rewards),
                    "top4_fraction_ranked": sum(r <= 4 for r in ranks) / len(ranks) if ranks else None,
                    "truncated": truncated_count}
        for key, value in expected.items():
            if value is None:
                if report[key] is not None:
                    raise ValueError("Aggregate mismatch")
            elif abs(report[key] - value) > 1e-12:
                raise ValueError("Aggregate mismatch")
        if report["replay_seeds"] != (spec["replay_seeds"] if label == "final" else []):
            raise ValueError("Missing policy replays")
        aggregates[label] = expected
    env.close()
    verify_files(root)
    result = {"status": "passed", "actual_steps": steps, "actual_updates": final["_n_updates"],
              "evaluation_episodes": 3 * len(spec["eval_seeds"]), "policy_replays": len(spec["replay_seeds"]),
              "independent_action_replays": 3 * len(spec["eval_seeds"]), "learner_decisions": decision_count,
              "metrics": aggregates, "peak_sampled_rss_bytes": peak, "outputs_and_returns_bytes": output_bytes,
              "training_ranked_episodes": summary["episode_count"],
              "training_truncations": summary["episode_truncations"],
              "training_seconds": summary["elapsed_seconds"], "steps_per_second": summary["steps_per_second"],
              "frozen_inputs_unchanged": True, "audit_seconds": time.monotonic() - started}
    dump(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run.resolve(), args.output), allow_nan=False))
