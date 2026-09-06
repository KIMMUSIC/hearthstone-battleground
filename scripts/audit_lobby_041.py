"""Independent dataset, classifier, optimizer, and game replay audit for 041."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import zipfile

from audit_lobby_032 import model_evidence
from audit_lobby_035 import replay_report
from lobby_032 import digest, dump, read
from lobby_035 import identity
from lobby_041 import decide, jobs
from run_diversity_009 import size, verify_files


def verify_dataset(path, seeds, env, heuristic):
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    if set(data) != {"actions", "seeds"} | {"obs__" + k for k in env.observation_space.spaces}:
        raise ValueError("Unexpected dataset fields")
    count = len(data["actions"])
    if any(len(v) != count for v in data.values()):
        raise ValueError("Dataset row count mismatch")
    index = 0
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            action = heuristic(env.game.view(0))
            if index >= count or data["seeds"][index] != seed or data["actions"][index] != action:
                raise ValueError("Dataset seed or teacher label differs")
            for key, value in obs.items():
                if not np.array_equal(data["obs__" + key][index], value):
                    raise ValueError("Dataset observation differs from frozen teacher trajectory")
            obs, _, terminated, truncated, _ = env.step(action)
            env.game.assert_conservation()
            if truncated:
                raise ValueError("Teacher trajectory truncated")
            done = terminated
            index += 1
    if index != count:
        raise ValueError("Extra dataset rows")
    return data


def classify(model, data):
    import numpy as np
    import torch
    correct, loss, counts, hits = 0, 0.0, {}, {}
    for start in range(0, len(data["actions"]), 256):
        obs = {k[5:]: v[start:start+256] for k, v in data.items() if k.startswith("obs__")}
        labels = data["actions"][start:start+256]
        masks = obs["action_mask"].astype(bool)
        predicted = model.predict(obs, deterministic=True, action_masks=masks)[0]
        tensors, _ = model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            _, log_prob, _ = model.policy.evaluate_actions(tensors, torch.as_tensor(labels, dtype=torch.long), action_masks=masks)
        loss -= float(log_prob.sum())
        correct += int(np.sum(predicted == labels))
        for label, prediction in zip(labels, predicted, strict=True):
            key = str(int(label))
            counts[key] = counts.get(key, 0) + 1
            hits[key] = hits.get(key, 0) + int(label == prediction)
    n = len(data["actions"])
    return {"accuracy": correct/n, "nll": loss/n, "rows": n,
            "per_action": {k: {"count": count, "correct": hits[k], "accuracy": hits[k]/count} for k, count in counts.items()}}


def actor_optimizer(model_path, model, expected_steps=1024):
    import torch
    with zipfile.ZipFile(model_path) as z:
        state = torch.load(io.BytesIO(z.read("policy.optimizer.pth")), weights_only=True, map_location="cpu")
    parameter_ids = [p for group in state["param_groups"] for p in group["params"]]
    if any(group["lr"] != .0003 for group in state["param_groups"]):
        raise ValueError("Unexpected imitation learning rate")
    named = list(model.policy.named_parameters())
    if len(parameter_ids) != len(named):
        raise ValueError("Optimizer topology changed")
    actor_ids = {p for p, (name, _) in zip(parameter_ids, named, strict=True)
                 if name.startswith(("mlp_extractor.policy_net.", "action_net."))}
    if not actor_ids or set(state["state"]) != actor_ids:
        raise ValueError("Unexpected actor/critic optimizer state")
    for value in state["state"].values():
        if float(value["step"]) != expected_steps or not all(torch.isfinite(value[k]).all() for k in ("step", "exp_avg", "exp_avg_sq")):
            raise ValueError("Bad supervised optimizer state")
    return {"actor_parameter_states": len(actor_ids), "steps": expected_steps, "critic_parameter_states": 0}


def audit(root):
    started = time.monotonic()
    if (root / "audit.json").exists():
        raise FileExistsError("Preserve previous audit")
    verify_files(root)
    spec = read(root / "configs/lobby041_imitation.json")
    if (spec["seeds"] != [7,17,27] or spec["dataset"]["train_seeds"] != list(range(100))
            or spec["dataset"]["dev_seeds"] != list(range(20001,20021)) or spec["eval_seeds"] != list(range(20001,20021))):
        raise ValueError("Precommitted seed sets changed")
    if spec["imitation"] != {"updates": 1024, "batch_size": 64, "learning_rate": .0003, "max_seconds": 60}:
        raise ValueError("Supervised settings changed")
    state = read(root / "outputs/driver-status.json")
    if state["status"] != "completed" or state["completed_phases"] != [identity(j) for j in jobs(spec)]:
        raise ValueError("Incomplete campaign")
    peak = 0
    output_bytes = size(root / "outputs") + size(root / "returns")
    if output_bytes >= spec["limits"]["disk_bytes"]:
        raise ValueError("Disk budget exceeded")
    for job in jobs(spec):
        name = identity(job)
        control = read(root / f"outputs/control-{name}/result.json")
        command = control["commands"][0]
        if (control["status"] != "completed" or len(control["commands"]) != 1 or command["returncode"] != 0 or command["remaining_pids"]
                or command["peak_sampled_rss"] >= spec["limits"]["rss_bytes"] or control["elapsed_seconds"] > 180):
            raise ValueError("Supervisor failure")
        peak = max(peak, command["peak_sampled_rss"])
        worker = json.loads((root / f"outputs/control-{name}/000.log").read_text().splitlines()[0])
        if worker != {"torch_threads": 1, "torch_interop_threads": 1}:
            raise ValueError("Worker CPU configuration")
        archive = root / f"returns/{name}.zip"
        receipt = read(archive.with_suffix(".json"))
        if digest(archive) != receipt["sha256"] or archive.stat().st_size != receipt["bytes"]:
            raise ValueError("Return hash differs")
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:
                raise ValueError("Return CRC")
            for path, sha in json.loads(z.read("FILES.json")).items():
                if digest(root / path) != sha or hashlib.sha256(z.read(path)).hexdigest() != sha:
                    raise ValueError("Return member differs")
    sys.path.insert(0, str(root / "src"))
    import torch
    from hearthstone_ai import lobby_env
    from hearthstone_ai.lobby_policies import heuristic_policy, random_policy
    from hearthstone_ai.lobby_training import load_model
    if not Path(lobby_env.__file__).resolve().is_relative_to(root / "src"):
        raise ValueError("Use frozen source")
    torch.set_num_threads(1)
    env = lobby_env.LobbyEnv(observation_scale="fixed-v1")
    dataset = root / "outputs/dataset-s007-student"
    train = verify_dataset(dataset / "train.npz", list(range(100)), env, heuristic_policy)
    dev = verify_dataset(dataset / "dev.npz", list(range(20001,20021)), env, heuristic_policy)
    classification, games, optimizers = {}, {}, {}
    actions = 0
    for seed in (7,17,27):
        folder = root / f"outputs/train-s{seed:03d}-student"
        status = read(folder / "status.json")
        if status["optimizer_steps"] != 1024 or status["rows_seen"] != 65536 or status["stop_reason"] != "updates":
            raise ValueError("Incomplete imitation")
        recorded = read(folder / "classification.json")
        classification[seed], games[seed], weights = {}, {}, {}
        for point, key in (("initial", "initial_checkpoint"), ("final", "checkpoint")):
            checkpoint = folder / status[key]
            meta = read(checkpoint / "metadata.json")
            if meta.get("training_method") != "behavior-cloning-diagnostic-v1":
                raise ValueError("Wrong training method label")
            expected_updates = 0 if point == "initial" else 1024
            if (meta.get("num_timesteps") != 0 or meta.get("additional_steps") != 0
                    or meta["supervised"].get("optimizer_steps") != expected_updates
                    or meta["supervised"].get("rows_seen") != expected_updates * 64):
                raise ValueError("Checkpoint supervised counters differ")
            if any(meta["config"].get(k) != v for k, v in (spec["train"] | {"seed": seed}).items()):
                raise ValueError("Wrong model seed or config")
            data, weights[point] = model_evidence(checkpoint / "model.zip")
            if data["num_timesteps"] != 0 or data["_n_updates"] != 0:
                raise ValueError("Supervised work mislabeled as PPO steps")
            if data["policy_kwargs"].get("net_arch") != [32,32]:
                raise ValueError("Policy architecture changed")
            model = load_model(checkpoint, env)
            classification[seed][point] = classify(model, dev)
            for metric in ("accuracy", "nll"):
                if abs(recorded[point][metric] - classification[seed][point][metric]) > 1e-5:
                    raise ValueError("Classification metric differs")
            independently_counted = classification[seed][point]
            if recorded[point]["rows"] != independently_counted["rows"] or recorded[point]["per_action"] != {
                k: {"count": v["count"], "accuracy": v["accuracy"]} for k, v in independently_counted["per_action"].items()
            }:
                raise ValueError("Per-action classification differs")
            report = read(root / f"outputs/evaluate-s{seed:03d}-student/{point}.json")
            if report["model_sha256"] != digest(checkpoint / "model.zip") or report["replay_seeds"] != (spec["replay_seeds"] if point == "final" else []):
                raise ValueError("Wrong evaluated checkpoint or replays")
            games[seed][point], count = replay_report(report, env, model, None, spec, heuristic_policy, random_policy)
            actions += count
            if point == "final":
                optimizers[seed] = actor_optimizer(checkpoint / "model.zip", model)
        actor_changed = False
        for name in weights["initial"]:
            same = torch.equal(weights["initial"][name], weights["final"][name])
            if name.startswith(("mlp_extractor.value_net.", "value_net.")) and not same:
                raise ValueError("Critic changed")
            if name.startswith(("mlp_extractor.policy_net.", "action_net.")) and not same:
                actor_changed = True
        if not actor_changed:
            raise ValueError("Actor unchanged")
    baselines = {}
    for policy in ("heuristic", "random"):
        report = read(root / f"outputs/baseline-s007-student/{policy}.json")
        baselines[policy], count = replay_report(report, env, None, policy, spec, heuristic_policy, random_policy)
        actions += count
    env.close()
    verify_files(root)
    result = {"status": "passed", "training_method": "behavior-cloning-diagnostic-v1", "ppo_steps": 0,
              "supervised_updates_per_model": 1024, "dataset_rows": {"train": len(train["actions"]), "dev": len(dev["actions"])},
              "dataset_sha256": {name: digest(dataset / f"{name}.npz") for name in ("train", "dev")},
              "classification": classification, "games": games, "baselines": baselines, "optimizers": optimizers,
              "decision": decide(classification, games), "evaluation_episodes": 160, "policy_replays": 6,
              "replayed_actions": actions, "critic_weights_unchanged": True, "frozen_inputs_unchanged": True,
              "peak_sampled_rss_bytes": peak, "outputs_and_returns_bytes": output_bytes, "audit_seconds": time.monotonic()-started}
    dump(root / "audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run.resolve()), allow_nan=False))
