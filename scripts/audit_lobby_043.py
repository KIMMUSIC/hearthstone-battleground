"""Independent audit for the 043 balanced-imitation comparison."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

from audit_lobby_032 import model_evidence
from audit_lobby_035 import replay_report
from audit_lobby_041 import actor_optimizer, verify_dataset
from lobby_032 import digest, dump, read
from lobby_035 import identity
from run_diversity_009 import size, verify_files


SPEC_NAME = "lobby043_comparison.json"
ARMS = {"control": "uniform-choice", "candidate": "balanced-choice"}
SAMPLING_PROTOCOL = "choice-probabilities-v1"
TRAIN_SEEDS = [7, 17, 27]
DATASET_TRAIN_SEEDS = list(range(100))
DATASET_DEV_SEEDS = list(range(20001, 20021))
EVAL_SEEDS = list(range(20001, 20021))
REPLAY_SEEDS = [20001, 20002]
EXPECTED_IMITATION = {"updates": 1024, "batch_size": 64, "learning_rate": 0.0003, "max_seconds": 60}


def _load_npz(path: Path) -> dict[str, object]:
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _spec_path(root: Path) -> Path:
    path = root / "configs" / SPEC_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing 043 spec: configs/{SPEC_NAME}")
    return path


def _validate_spec(spec: dict) -> None:
    if spec.get("seeds") != TRAIN_SEEDS:
        raise ValueError("043 training seed set changed")
    if spec.get("arms") != ARMS:
        raise ValueError("043 arm definitions changed")
    if spec.get("eval_seeds") != EVAL_SEEDS or spec.get("replay_seeds") != REPLAY_SEEDS:
        raise ValueError("043 evaluation seed sets changed")
    dataset = spec.get("dataset", {})
    if dataset.get("train_seeds") != DATASET_TRAIN_SEEDS or dataset.get("dev_seeds") != DATASET_DEV_SEEDS:
        raise ValueError("043 dataset seed sets changed")
    if dataset.get("max_rounds") != 100:
        raise ValueError("043 dataset max_rounds changed")
    train = spec.get("train", {})
    required_train = {
        "max_steps": 8192,
        "max_seconds": 60.0,
        "threads": 1,
        "seed": 7,
        "n_steps": 32,
        "batch_size": 32,
        "n_epochs": 4,
        "gamma": 1.0,
        "learner_seat": 0,
        "max_rounds": 100,
        "heuristic_opponents": 7,
        "observation_scale": "fixed-v1",
    }
    if any(train.get(key) != value for key, value in required_train.items()):
        raise ValueError("043 shared training settings changed")
    if spec.get("imitation") != EXPECTED_IMITATION:
        raise ValueError("043 supervised settings changed")


def expected_jobs(spec: dict):
    return (
        [("dataset", 7, "student")]
        + [(mode, seed, arm) for mode in ("train", "evaluate") for seed in spec["seeds"] for arm in ("control", "candidate")]
        + [("baseline", 7, "control")]
    )


def sampling_probabilities(actions, mode: str):
    import numpy as np

    labels = np.asarray(actions, dtype=np.int64)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("Training labels must be a nonempty vector")
    if mode == "uniform-choice":
        return np.full(labels.shape[0], 1.0 / labels.shape[0], dtype=np.float64)
    if mode != "balanced-choice":
        raise ValueError(f"Unknown 043 sampling mode: {mode}")
    counts = Counter(int(action) for action in labels)
    classes = len(counts)
    probabilities = np.empty(labels.shape[0], dtype=np.float64)
    for index, action in enumerate(labels):
        probabilities[index] = 0.5 / labels.shape[0] + 0.5 / (classes * counts[int(action)])
    if abs(float(probabilities.sum()) - 1.0) > 1e-12:
        raise ValueError("Sampling probabilities do not sum to 1")
    return probabilities


def expected_sampling_indices(actions, mode: str, seed: int, updates: int, batch_size: int):
    import numpy as np

    probabilities = sampling_probabilities(actions, mode)
    rng = np.random.default_rng(seed)
    indices = np.empty((updates, batch_size), dtype=np.int64)
    for update in range(updates):
        indices[update] = rng.choice(len(probabilities), size=batch_size, replace=True, p=probabilities)
    return indices, probabilities


def _histogram(values) -> dict[str, int]:
    return {str(int(k)): int(v) for k, v in sorted(Counter(int(value) for value in values).items())}


def verify_sampling(folder: Path, actions, mode: str, seed: int, updates: int = 1024, batch_size: int = 64):
    import numpy as np

    path = folder / "sampling.npz"
    data = _load_npz(path)
    if set(data) != {"indices", "probabilities"}:
        raise ValueError("Unexpected sampling proof fields")
    expected_indices, expected_probabilities = expected_sampling_indices(actions, mode, seed, updates, batch_size)
    indices = np.asarray(data["indices"])
    probabilities = np.asarray(data["probabilities"])
    if indices.dtype != np.int64 or indices.shape != expected_indices.shape:
        raise ValueError("Sampling indices shape or dtype mismatch")
    if probabilities.dtype != np.float64 or probabilities.shape != expected_probabilities.shape:
        raise ValueError("Sampling probabilities shape or dtype mismatch")
    if not np.array_equal(indices, expected_indices):
        raise ValueError("Sampling indices differ from deterministic replay")
    if not np.allclose(probabilities, expected_probabilities, rtol=0, atol=1e-15):
        raise ValueError("Sampling probabilities differ from training labels")
    flat = indices.reshape(-1)
    if flat.size != updates * batch_size or flat.min(initial=0) < 0 or flat.max(initial=-1) >= len(expected_probabilities):
        raise ValueError("Sampling indices outside training split")
    sample_actions = np.asarray(actions, dtype=np.int64)[flat]
    sampling_sha256 = digest(path)
    action_histogram = _histogram(sample_actions)
    return {
        "sampling_sha256": sampling_sha256,
        "indices_shape": list(indices.shape),
        "probabilities_sha256": hashlib.sha256(np.ascontiguousarray(probabilities).tobytes()).hexdigest(),
        "probability_sum": float(probabilities.sum()),
        "row_exposure_histogram": _histogram(flat),
        "action_exposure_histogram": action_histogram,
        "metadata": {
            "mode": mode,
            "protocol": SAMPLING_PROTOCOL,
            "hash": sampling_sha256,
            "histogram": action_histogram,
        },
    }


def classify(model, data):
    import numpy as np
    import torch

    correct = 0
    loss = 0.0
    counts: dict[str, int] = {}
    hits: dict[str, int] = {}
    confusion = np.zeros((37, 37), dtype=np.int64)
    actions = np.asarray(data["actions"], dtype=np.int64)
    for start in range(0, len(actions), 256):
        obs = {k[5:]: v[start:start + 256] for k, v in data.items() if k.startswith("obs__")}
        labels = actions[start:start + 256]
        masks = obs["action_mask"].astype(bool)
        predicted = model.predict(obs, deterministic=True, action_masks=masks)[0]
        tensors, _ = model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            _, log_prob, _ = model.policy.evaluate_actions(tensors, torch.as_tensor(labels, dtype=torch.long), action_masks=masks)
        if not torch.isfinite(log_prob).all():
            raise FloatingPointError("Non-finite label log probabilities")
        loss -= float(log_prob.sum())
        correct += int(np.sum(predicted == labels))
        for label, prediction in zip(labels, predicted, strict=True):
            key = str(int(label))
            counts[key] = counts.get(key, 0) + 1
            hits[key] = hits.get(key, 0) + int(label == prediction)
            confusion[int(label), int(prediction)] += 1
    n = len(actions)
    per_action = {key: {"count": count, "accuracy": hits.get(key, 0) / count}
                  for key, count in sorted(counts.items(), key=lambda item: int(item[0]))}
    macro = sum(value["accuracy"] for value in per_action.values()) / len(per_action) if per_action else 0.0
    return {"accuracy": correct / n, "macro_action_accuracy": macro, "nll": loss / n, "rows": n,
            "per_action": per_action, "confusion_matrix": confusion.tolist()}


def _compare_metric(recorded: dict, expected: dict, path: str) -> None:
    for metric in ("accuracy", "macro_action_accuracy", "nll"):
        if metric not in recorded:
            raise ValueError(f"Missing classification metric {path}.{metric}")
        if abs(recorded[metric] - expected[metric]) > 1e-5:
            raise ValueError(f"Classification metric differs: {path}.{metric}")
    for key in ("rows", "per_action", "confusion_matrix"):
        if recorded.get(key) != expected[key]:
            raise ValueError(f"Classification detail differs: {path}.{key}")


def _assert_supervised_metadata(
    meta: dict,
    config: dict,
    updates: int,
    batch_size: int,
    expected_sampling: dict,
) -> None:
    if meta.get("training_method") != "behavior-cloning-diagnostic-v1":
        raise ValueError("Wrong training method label")
    supervised = meta.get("supervised", {})
    if (meta.get("num_timesteps") != 0 or meta.get("additional_steps") != 0
            or supervised.get("optimizer_steps") != updates
            or supervised.get("rows_seen") != updates * batch_size
            or supervised.get("critic_unchanged") is False
            or supervised.get("finite_parameters") is False):
        raise ValueError("Checkpoint supervised counters differ")
    if supervised.get("sampling") != expected_sampling:
        raise ValueError("Checkpoint sampling metadata differs")
    if any(meta.get("config", {}).get(key) != value for key, value in config.items()):
        raise ValueError("Wrong model seed or config")


def _verify_driver_artifacts(root: Path, spec: dict, jobs) -> tuple[int, int]:
    peak = 0
    output_bytes = size(root / "outputs") + size(root / "returns")
    if output_bytes >= spec["limits"]["disk_bytes"]:
        raise ValueError("Disk budget exceeded")
    for job in jobs:
        name = identity(job)
        control = read(root / f"outputs/control-{name}/result.json")
        command = control["commands"][0]
        if (control["status"] != "completed" or len(control["commands"]) != 1 or command["returncode"] != 0
                or command["remaining_pids"] or command["peak_sampled_rss"] >= spec["limits"]["rss_bytes"]
                or control["elapsed_seconds"] > spec["limits"]["phase_seconds"]):
            raise ValueError("Supervisor failure")
        peak = max(peak, command["peak_sampled_rss"])
        worker = json.loads((root / f"outputs/control-{name}/000.log").read_text(encoding="utf-8").splitlines()[0])
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
    return peak, output_bytes


def audit(root: Path):
    started = time.monotonic()
    root = Path(root)
    if (root / "audit.json").exists():
        raise FileExistsError("Preserve previous audit")
    verify_files(root)
    spec = read(_spec_path(root))
    _validate_spec(spec)
    sys.path.insert(0, str(root / "scripts"))
    from lobby_043 import decide, jobs as runner_jobs

    planned_jobs = expected_jobs(spec)
    if runner_jobs(spec) != planned_jobs:
        raise ValueError("043 runner job order differs from audit contract")
    state = read(root / "outputs/driver-status.json")
    if state["status"] != "completed" or state["completed_phases"] != [identity(job) for job in planned_jobs]:
        raise ValueError("Incomplete campaign")
    peak, output_bytes = _verify_driver_artifacts(root, spec, planned_jobs)

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
    train = verify_dataset(dataset / "train.npz", DATASET_TRAIN_SEEDS, env, heuristic_policy)
    dev = verify_dataset(dataset / "dev.npz", DATASET_DEV_SEEDS, env, heuristic_policy)
    train_actions = train["actions"]
    labels: dict[int, dict[str, dict[str, dict[str, dict]]]] = {}
    games: dict[int, dict[str, dict[str, dict]]] = {}
    optimizers: dict[str, dict] = {}
    sampling: dict[str, dict] = {}
    actions = 0
    for seed in TRAIN_SEEDS:
        labels[seed] = {}
        games[seed] = {}
        initial_weights = {}
        for arm, sampling_mode in ARMS.items():
            folder = root / "outputs" / identity(("train", seed, arm))
            status = read(folder / "status.json")
            if (status.get("optimizer_steps") != EXPECTED_IMITATION["updates"]
                    or status.get("rows_seen") != EXPECTED_IMITATION["updates"] * EXPECTED_IMITATION["batch_size"]
                    or status.get("stop_reason") != "updates"
                    or status.get("num_timesteps") != 0
                    or status.get("sb3_n_updates") != 0
                    or status.get("critic_unchanged") is not True
                    or status.get("finite_parameters") is not True):
                raise ValueError("Incomplete imitation")
            proof = verify_sampling(
                folder,
                train_actions,
                sampling_mode,
                seed,
                EXPECTED_IMITATION["updates"],
                EXPECTED_IMITATION["batch_size"],
            )
            if status.get("sampling") != proof["metadata"]:
                raise ValueError("Wrong status sampling metadata")
            sampling[f"s{seed}-{arm}"] = proof
            recorded = read(folder / "classification.json")
            labels[seed][arm] = {}
            games[seed][arm] = {}
            weights = {}
            for point, key in (("initial", "initial_checkpoint"), ("final", "checkpoint")):
                checkpoint = folder / status[key]
                expected_updates = 0 if point == "initial" else EXPECTED_IMITATION["updates"]
                meta = read(checkpoint / "metadata.json")
                _assert_supervised_metadata(
                    meta,
                    spec["train"] | {"seed": seed},
                    expected_updates,
                    EXPECTED_IMITATION["batch_size"],
                    proof["metadata"] if point == "final" else {
                        "mode": sampling_mode,
                        "protocol": SAMPLING_PROTOCOL,
                        "hash": None,
                        "histogram": {},
                    },
                )
                data, weights[point] = model_evidence(checkpoint / "model.zip")
                if data["num_timesteps"] != 0 or data["_n_updates"] != 0:
                    raise ValueError("Supervised work mislabeled as PPO steps")
                if data["policy_kwargs"].get("net_arch") != [32, 32]:
                    raise ValueError("Policy architecture changed")
                model = load_model(checkpoint, env)
                labels[seed][arm][point] = {"train": classify(model, train), "dev": classify(model, dev)}
                for split in ("train", "dev"):
                    _compare_metric(recorded[point][split], labels[seed][arm][point][split], f"s{seed}.{arm}.{point}.{split}")
                report = read(root / "outputs" / identity(("evaluate", seed, arm)) / f"{point}.json")
                if report["model_sha256"] != digest(checkpoint / "model.zip") or report["replay_seeds"] != (REPLAY_SEEDS if point == "final" else []):
                    raise ValueError("Wrong evaluated checkpoint or replays")
                games[seed][arm][point], count = replay_report(report, env, model, None, spec, heuristic_policy, random_policy)
                actions += count
                if point == "final":
                    optimizers[f"s{seed}-{arm}"] = actor_optimizer(checkpoint / "model.zip", model, EXPECTED_IMITATION["updates"])
            actor_changed = False
            for name in weights["initial"]:
                same = torch.equal(weights["initial"][name], weights["final"][name])
                if name.startswith(("mlp_extractor.value_net.", "value_net.")) and not same:
                    raise ValueError("Critic changed")
                if name.startswith(("mlp_extractor.policy_net.", "action_net.")) and not same:
                    actor_changed = True
            if not actor_changed:
                raise ValueError("Actor unchanged")
            initial_weights[arm] = weights["initial"]
        if not all(torch.equal(initial_weights["control"][name], initial_weights["candidate"][name]) for name in initial_weights["control"]):
            raise ValueError("Paired initial policies differ")
    baselines = {}
    for policy in ("heuristic", "random"):
        report = read(root / "outputs/baseline-s007-control" / f"{policy}.json")
        baselines[policy], count = replay_report(report, env, None, policy, spec, heuristic_policy, random_policy)
        actions += count
    env.close()
    verify_files(root)
    result = {
        "status": "passed",
        "training_method": "behavior-cloning-diagnostic-v1",
        "ppo_steps": 0,
        "supervised_updates_per_model": 1024,
        "models": 6,
        "dataset_rows": {"train": len(train["actions"]), "dev": len(dev["actions"])},
        "dataset_sha256": {name: digest(dataset / f"{name}.npz") for name in ("train", "dev")},
        "labels": labels,
        "games": games,
        "baselines": baselines,
        "optimizers": optimizers,
        "sampling": sampling,
        "decision": decide(labels, games),
        "evaluation_episodes": 280,
        "policy_replays": 12,
        "replayed_actions": actions,
        "paired_initial_policies_equal": True,
        "critic_weights_unchanged": True,
        "frozen_inputs_unchanged": True,
        "peak_sampled_rss_bytes": peak,
        "outputs_and_returns_bytes": output_bytes,
        "audit_seconds": time.monotonic() - started,
    }
    dump(root / "audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run.resolve()), allow_nan=False))
