"""Full evidence audit of the fixed three-seed opponent comparison."""

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import time
import zipfile

from audit_lobby_032 import model_evidence
from lobby_032 import digest, dump, read
from lobby_035 import decide, identity, jobs, training_config
from run_diversity_009 import size, verify_files


def optimizer_steps(path, expected):
    import torch
    with zipfile.ZipFile(path) as z:
        optimizer = torch.load(io.BytesIO(z.read("policy.optimizer.pth")), weights_only=True, map_location="cpu")
    parameters = [p for group in optimizer["param_groups"] for p in group["params"]]
    states = optimizer["state"]
    if not parameters or set(states) != set(parameters):
        raise ValueError("Missing optimizer parameter state")
    steps = []
    for state in states.values():
        if not all(torch.isfinite(state[k]).all() for k in ("step", "exp_avg", "exp_avg_sq")):
            raise ValueError("Nonfinite Adam state")
        step = float(state["step"])
        if step != expected:
            raise ValueError("Actual Adam steps mismatch")
        steps.append(int(step))
    return {"parameter_states": len(states), "min_step": min(steps), "max_step": max(steps)}


def audit(root, experiment="035"):
    if experiment not in {"035", "037", "039"}:
        raise ValueError("Unknown comparison")
    if (root / "audit.json").exists():
        raise FileExistsError("Preserve audit")
    started = time.monotonic()
    verify_files(root)
    spec = read(root / f"configs/lobby{experiment}_comparison.json")
    expected_arms = ({"control": 7, "candidate": 3} if experiment == "035" else
                     {"control": {"heuristic_opponents": 3, "observation_scale": "raw"},
                      "candidate": {"heuristic_opponents": 3, "observation_scale": "fixed-v1"}})
    if experiment == "039":
        expected_arms = {"control": {"n_steps": 32}, "candidate": {"n_steps": 128}}
        if spec["train"].get("heuristic_opponents") != 3 or spec["train"].get("observation_scale") != "fixed-v1":
            raise ValueError("039 shared settings changed")
    if spec["seeds"] != [7, 17, 27] or spec["arms"] != expected_arms:
        raise ValueError("Precommitted selection changed")
    state = read(root / "outputs/driver-status.json")
    if state["status"] != "completed" or state["completed_phases"] != [identity(j) for j in jobs(spec)]:
        raise ValueError("Incomplete comparison")
    sys.path.insert(0, str(root / "src"))
    import torch
    from hearthstone_ai import lobby_env
    from hearthstone_ai.lobby_training import load_model
    from hearthstone_ai.lobby_policies import heuristic_policy, random_policy
    if not Path(lobby_env.__file__).resolve().is_relative_to(root / "src"):
        raise ValueError("Use frozen source")
    torch.set_num_threads(1)
    peak, totals = 0, {"train": 0.0, "evaluate": 0.0}
    output_bytes = size(root / "outputs") + size(root / "returns")
    if output_bytes >= spec["limits"]["disk_bytes"]:
        raise ValueError("Output budget")
    for job in jobs(spec):
        name = identity(job)
        control = read(root / f"outputs/control-{name}/result.json")
        c = control["commands"][0]
        if (control["status"] != "completed" or len(control["commands"]) != 1 or c["returncode"] != 0 or c["remaining_pids"]
                or c["peak_sampled_rss"] >= spec["limits"]["rss_bytes"] or control["elapsed_seconds"] > spec["limits"]["phase_seconds"]):
            raise ValueError("Supervisor failure")
        peak = max(peak, c["peak_sampled_rss"])
        totals["train" if job[0] == "train" else "evaluate"] += control["elapsed_seconds"]
        worker = json.loads((root / f"outputs/control-{name}/000.log").read_text().splitlines()[0])
        if worker != {"torch_threads": 1, "torch_interop_threads": 1}:
            raise ValueError("Worker CPU mismatch")
        archive = root / f"returns/{name}.zip"
        receipt = read(archive.with_suffix(".json"))
        if digest(archive) != receipt["sha256"] or archive.stat().st_size != receipt["bytes"]:
            raise ValueError("Return hash mismatch")
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:
                raise ValueError("Return CRC failed")
            for path, sha in json.loads(z.read("FILES.json")).items():
                if digest(root / path) != sha or hashlib.sha256(z.read(path)).hexdigest() != sha:
                    raise ValueError("Return file changed")
    rows, training_evidence = {}, {}
    total_actions = 0
    for seed in spec["seeds"]:
        rows[seed] = {}
        initial_weights = {}
        for arm in spec["arms"]:
            expected_cfg = training_config(spec, seed, arm)
            env = lobby_env.LobbyEnv(learner_seat=0, max_rounds=100, heuristic_opponents=7,
                                    **({"observation_scale": expected_cfg["observation_scale"]} if experiment != "035" else {}))
            train_dir = root / "outputs" / identity(("train", seed, arm))
            summary = read(train_dir / "status.json")
            cfg = read(train_dir / "manifest.json")["config"]
            if any(cfg.get(k) != v for k, v in expected_cfg.items()):
                raise ValueError("Training configuration mismatch")
            initial_dir = train_dir / summary["initial_checkpoint"]
            final_dir = train_dir / summary["checkpoint"]
            initial, weights = model_evidence(initial_dir / "model.zip")
            initial_weights[arm] = weights
            final, final_weights = model_evidence(final_dir / "model.zip")
            rollout = expected_cfg["n_steps"]
            if (initial["num_timesteps"] != 0 or final["num_timesteps"] != 8192 or final["_n_updates"] != 8192 // rollout * 4
                    or final["gamma"] != 1 or final["gae_lambda"] != .95 or final["n_steps"] != rollout or final["batch_size"] != 32 or final["n_epochs"] != 4
                    or summary["num_timesteps"] != 8192 or summary["stop_reason"] != "steps"
                    or all(torch.equal(weights[k], final_weights[k]) for k in weights)):
                raise ValueError("Actual training/update mismatch")
            episodes = summary["episodes"]
            if any(type(e["seed"]) is not int or not 0 <= e["seed"] <= 9999 for e in episodes):
                raise ValueError("Training seed leak")
            distribution = Counter(e["rank"] for e in episodes if e["rank"] is not None)
            if summary["episode_count"] != sum(distribution.values()) or summary["episode_truncations"] != sum(e["truncated"] for e in episodes):
                raise ValueError("Training metrics mismatch")
            training_evidence[f"s{seed}-{arm}"] = {"rank_distribution": dict(distribution), "truncations": summary["episode_truncations"],
                                                   "seconds": summary["elapsed_seconds"], "steps_per_second": summary["steps_per_second"]}
            if experiment == "039":
                training_evidence[f"s{seed}-{arm}"].update(
                    sb3_n_updates=final["_n_updates"], n_steps=rollout,
                    adam=optimizer_steps(final_dir / "model.zip", 1024))
            rows[seed][arm] = {}
            for point, checkpoint in (("initial", initial_dir), ("final", final_dir)):
                report = read(root / "outputs" / identity(("evaluate", seed, arm)) / f"{point}.json")
                if report["model_sha256"] != digest(checkpoint / "model.zip"):
                    raise ValueError("Wrong evaluated model")
                model = load_model(checkpoint, env, allow_opponent_shift=True)
                stats, actions = replay_report(report, env, model, None, spec, heuristic_policy, random_policy)
                total_actions += actions
                rows[seed][arm][point] = stats
                if report["replay_seeds"] != (spec["replay_seeds"] if point == "final" else []):
                    raise ValueError("Missing policy replays")
            env.close()
        if not all(torch.equal(initial_weights["control"][k], initial_weights["candidate"][k]) for k in initial_weights["control"]):
            raise ValueError("Paired initial policies differ")
    baselines = {}
    env = lobby_env.LobbyEnv(learner_seat=0, max_rounds=100, heuristic_opponents=7)
    for policy in ("heuristic", "random"):
        report = read(root / "outputs" / identity(("baseline", 7, "control")) / f"{policy}.json")
        stats, actions = replay_report(report, env, None, policy, spec, heuristic_policy, random_policy)
        baselines[policy] = stats
        total_actions += actions
    env.close()
    verify_files(root)
    result = {"status": "passed", "actual_steps": 49152, "models": 6, "updates_per_model": 1024,
              "evaluation_episodes": 280, "policy_replays": 12, "independent_episode_replays": 280,
              "learner_actions_replayed": total_actions, "paired_initial_policies_equal": True,
              "decision": decide(rows), "rows": rows, "baselines": baselines, "training": training_evidence,
              "peak_sampled_rss_bytes": peak, "outputs_and_returns_bytes": output_bytes, "phase_seconds": totals,
              "audit_seconds": time.monotonic() - started, "frozen_inputs_unchanged": True}
    if experiment == "039":
        result.pop("updates_per_model")
        result["adam_steps_per_parameter"] = 1024
        result["sb3_n_updates_by_arm"] = {"control": 1024, "candidate": 256}
    dump(root / "audit.json", result)
    return result


def replay_report(report, env, model, policy, spec, heuristic_policy, random_policy):
    if [e["seed"] for e in report["episodes"]] != spec["eval_seeds"]:
        raise ValueError("Evaluation seed selection")
    ranks, rewards, actions = [], [], Counter()
    truncated_count, forced = 0, 0
    for record in report["episodes"]:
        obs, _ = env.reset(seed=record["seed"])
        rng = random.Random(f"lobby-probe:{record['seed']}:policy:0")
        total, done = 0, False
        for action in record["decisions"]:
            if done or not env.action_masks()[action] or not env.observation_space.contains(obs):
                raise ValueError("Illegal evaluation action")
            predicted = (int(model.predict(obs, deterministic=True, action_masks=env.action_masks())[0]) if model is not None else
                         random_policy(env.game.view(0), rng) if policy == "random" else heuristic_policy(env.game.view(0)))
            if action != predicted:
                raise ValueError("Saved policy action mismatch")
            obs, reward, terminated, truncated, info = env.step(action)
            env.game.assert_conservation()
            total += reward
            actions[action] += 1
            done = terminated or truncated
        rank = env.game.players[0].rank
        if (not done or rank != record["rank"] or total != record["reward"] or terminated != record["terminated"]
                or truncated != record["truncated"] or info != record["info"] or not env.observation_space.contains(obs)
                or json.loads(json.dumps(env.game.trace)) != record["trace"]):
            raise ValueError("Evaluation trace mismatch")
        if rank is not None:
            ranks.append(rank)
        rewards.append(total)
        truncated_count += truncated
        forced += sum(e["event"] == "recruit_end" and e.get("seat") == 0 and e.get("forced", False) for e in env.game.trace)
    metrics = {"mean_rank": sum(ranks)/len(ranks) if ranks else None,
               "mean_reward": sum(rewards)/len(rewards), "truncated": truncated_count,
               "top4_fraction_ranked": sum(r <= 4 for r in ranks)/len(ranks) if ranks else None}
    for key, value in metrics.items():
        if value is None:
            if report[key] is not None:
                raise ValueError("Bad aggregate")
        elif abs(report[key] - value) > 1e-12:
            raise ValueError("Bad aggregate")
    metrics.update(actions=sum(actions.values()), freezes=actions[3], buys=sum(actions[a] for a in range(4, 11)),
                   plays=sum(actions[a] for a in range(18, 28)), forced_ends=forced)
    return metrics, sum(actions.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--experiment", choices=["035", "037", "039"], default="035")
    args = parser.parse_args()
    print(json.dumps(audit(args.run.resolve(), args.experiment), allow_nan=False))
