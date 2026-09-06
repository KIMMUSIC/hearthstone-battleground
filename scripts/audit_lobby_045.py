"""Independent audit for the 045 visited-state training comparison."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from audit_lobby_032 import model_evidence
from audit_lobby_035 import replay_report
from audit_lobby_041 import actor_optimizer, verify_dataset
from audit_lobby_043 import (
    EXPECTED_IMITATION,
    REPLAY_SEEDS,
    _assert_supervised_metadata,
    _compare_metric,
    _verify_driver_artifacts,
    classify,
    verify_sampling,
)
from lobby_032 import digest, dump, read
from lobby_035 import identity
from run_diversity_009 import verify_files


SPEC_NAME = "lobby045_comparison.json"
ARMS = {"control": "balanced-choice", "candidate": "balanced-choice"}
TRAIN_SEEDS = [7, 17, 27]
DATASET_TRAIN_SEEDS = list(range(100))
DATASET_DEV_SEEDS = list(range(20001, 20021))
EVAL_SEEDS = list(range(20001, 20021))
BASE_DATASET_ID = "dataset-s007-teacher"
EXPECTED_SRC_FILES = 19


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _spec_path(root: Path) -> Path:
    path = root / "configs" / SPEC_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing 045 spec: configs/{SPEC_NAME}")
    return path


def expected_jobs(spec: dict[str, Any]):
    seeds = spec["seeds"]
    return (
        [("dataset", 7, "teacher")]
        + [("collect", seed, "candidate") for seed in seeds]
        + [("train", seed, arm) for seed in seeds for arm in ("control", "candidate")]
        + [("evaluate", seed, arm) for seed in seeds for arm in ("control", "candidate")]
        + [("baseline", 7, "control")]
    )


def validate_045_spec(spec: dict[str, Any]) -> None:
    if spec.get("seeds") != TRAIN_SEEDS:
        raise ValueError("045 training seed set changed")
    if spec.get("arms") != ARMS:
        raise ValueError("045 arms must both use balanced-choice")
    if spec.get("eval_seeds") != EVAL_SEEDS or spec.get("replay_seeds") != REPLAY_SEEDS:
        raise ValueError("045 evaluation seed sets changed")
    if spec.get("imitation") != EXPECTED_IMITATION:
        raise ValueError("045 supervised settings changed")
    dataset = spec.get("dataset", {})
    if dataset.get("train_seeds") != DATASET_TRAIN_SEEDS or dataset.get("dev_seeds") != DATASET_DEV_SEEDS:
        raise ValueError("045 dataset seed sets changed")
    if dataset.get("max_rounds") != 100:
        raise ValueError("045 dataset max_rounds changed")
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
        raise ValueError("045 shared training settings changed")
    collectors = spec.get("collectors")
    if sorted(int(seed) for seed in collectors or {}) != TRAIN_SEEDS:
        raise ValueError("045 collectors must be declared for all training seeds")
    for seed in TRAIN_SEEDS:
        row = collectors[str(seed)]
        expected_checkpoint = f"runs/lobby-043-local-r1/outputs/train-s{seed:03d}-candidate/checkpoint-final"
        if row.get("checkpoint") != expected_checkpoint:
            raise ValueError("045 collector checkpoint path changed")
        if not _sha_like(row.get("model_sha256")) or not _sha_like(row.get("metadata_sha256")):
            raise ValueError("045 collector hashes missing")


def _sha_like(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _array_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and left.dtype == right.dtype and np.array_equal(left, right)


def _assert_prefix(prefix: dict[str, np.ndarray], augmented: dict[str, np.ndarray], rows: int) -> None:
    if set(prefix) != set(augmented):
        raise ValueError("Augmented dataset keys differ from base dataset")
    for key, value in prefix.items():
        if not _array_equal(value, augmented[key][:rows]):
            raise ValueError(f"Candidate train prefix differs from base teacher train: {key}")


def _episode_list(collection: dict[str, Any]) -> list[dict[str, Any]]:
    episodes = collection.get("episodes") or collection.get("trajectories")
    if not isinstance(episodes, list):
        raise ValueError("collection.json missing episodes")
    return episodes


def _episode_seed(row: dict[str, Any]) -> int:
    return int(row.get("seed", row.get("train_seed")))


def _row_range(row: dict[str, Any]) -> tuple[int, int]:
    value = row.get("row_range", row.get("teacher_row_range", row.get("rows")))
    if value is None and "row_start" in row and "row_end" in row:
        value = [row["row_start"], row["row_end"]]
    if not (isinstance(value, (list, tuple)) and len(value) == 2):
        raise ValueError("collection episode missing row_range")
    start, stop = int(value[0]), int(value[1])
    if stop < start:
        raise ValueError("collection episode row_range is reversed")
    return start, stop


def _actions(row: dict[str, Any], *names: str) -> list[int]:
    for name in names:
        value = row.get(name)
        if value is not None:
            if not isinstance(value, list):
                raise ValueError(f"collection {name} must be a list")
            return [int(action) for action in value]
    raise ValueError(f"collection episode missing one of {names}")


def _rank(row: dict[str, Any]) -> float:
    value = row.get("rank", row.get("natural_rank"))
    if value is None:
        raise ValueError("collection episode missing rank")
    return float(value)


def _reward(row: dict[str, Any]) -> float:
    value = row.get("reward", row.get("total_reward"))
    if value is None:
        raise ValueError("collection episode missing reward")
    return float(value)


def _trace_hash(row: dict[str, Any]) -> str:
    value = row.get("trace_sha256", row.get("trace_hash"))
    if not _sha_like(value):
        raise ValueError("collection episode missing trace hash")
    return value


def _assert_obs_row(data: dict[str, np.ndarray], index: int, obs: dict[str, np.ndarray], teacher_action: int, seed: int) -> None:
    for key, expected in obs.items():
        npz_key = "obs__" + key
        if npz_key not in data or not np.array_equal(data[npz_key][index], expected):
            raise ValueError(f"Collector observation row differs: seed={seed} row={index} key={key}")
    if int(data["actions"][index]) != int(teacher_action) or int(data["seeds"][index]) != int(seed):
        raise ValueError(f"Collector teacher label or seed differs: seed={seed} row={index}")


def _maybe_run_root(output_dir: Path) -> Path | None:
    return output_dir.parent.parent if output_dir.parent.name == "outputs" else None


def _checkpoint_values(collect_dir: Path, collector: dict[str, Any]) -> set[str]:
    value = collector.get("checkpoint")
    if not isinstance(value, str):
        return set()
    values = {value}
    root = _maybe_run_root(collect_dir)
    if root is not None:
        values.add(str(root / value))
    return values


def _assert_optional_dataset_hashes(holder: dict[str, Any], train_sha: str, dev_sha: str, context: str) -> None:
    for train_key in ("dataset_train_sha256", "train_sha256"):
        if train_key in holder and holder[train_key] != train_sha:
            raise ValueError(f"{context} train dataset hash differs")
    for dev_key in ("dataset_dev_sha256", "dev_sha256"):
        if dev_key in holder and holder[dev_key] != dev_sha:
            raise ValueError(f"{context} dev dataset hash differs")


def verify_collector_dataset(
    base_dir: Path,
    collect_dir: Path,
    collector: dict[str, Any],
    env,
    model,
    heuristic_policy,
    train_seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Replay a 045 collector dataset from the parent policy and teacher labels."""
    train_seeds = DATASET_TRAIN_SEEDS if train_seeds is None else train_seeds
    base_dir = Path(base_dir)
    collect_dir = Path(collect_dir)
    base_train_path = base_dir / "train.npz"
    base_dev_path = base_dir / "dev.npz"
    train_path = collect_dir / "train.npz"
    dev_path = collect_dir / "dev.npz"
    base_train = load_npz(base_train_path)
    base_dev = load_npz(base_dev_path)
    augmented_train = load_npz(train_path)
    augmented_dev = load_npz(dev_path)
    base_train_sha = digest(base_train_path)
    base_dev_sha = digest(base_dev_path)
    train_sha = digest(train_path)
    dev_sha = digest(dev_path)
    if base_dev_sha != dev_sha:
        raise ValueError("Candidate dev split must be byte-identical to base dev split")
    for key, value in base_dev.items():
        if not _array_equal(value, augmented_dev[key]):
            raise ValueError(f"Candidate dev array differs from base: {key}")
    base_rows = len(base_train["actions"])
    _assert_prefix(base_train, augmented_train, base_rows)

    manifest = read(collect_dir / "manifest.json")
    if manifest.get("config", {}).get("train_seeds") != train_seeds or manifest.get("config", {}).get("dev_seeds") != DATASET_DEV_SEEDS:
        raise ValueError("Collector manifest split seeds changed")
    if manifest.get("split_counts", {}).get("train") != len(augmented_train["actions"]):
        raise ValueError("Collector manifest train count mismatch")
    if manifest.get("split_hashes", {}).get("train") != train_sha:
        raise ValueError("Collector manifest train hash mismatch")
    if manifest.get("split_hashes", {}).get("dev") != dev_sha:
        raise ValueError("Collector manifest dev hash mismatch")

    collection = read(collect_dir / "collection.json")
    episodes = sorted(_episode_list(collection), key=_episode_seed)
    if [_episode_seed(row) for row in episodes] != train_seeds:
        raise ValueError("Collector did not use exactly the training seeds")
    expected_input_hashes = {
        "base_train": base_train_sha,
        "base_dev": base_dev_sha,
        "checkpoint_model": collector["model_sha256"],
        "checkpoint_metadata": collector["metadata_sha256"],
    }
    if collection.get("checkpoint_model_sha256") != collector["model_sha256"]:
        raise ValueError("Collection model hash differs from spec")
    if collection.get("checkpoint_metadata_sha256") != collector["metadata_sha256"]:
        raise ValueError("Collection metadata hash differs from spec")
    expected_checkpoints = _checkpoint_values(collect_dir, collector)
    if expected_checkpoints and collection.get("checkpoint") not in expected_checkpoints:
        raise ValueError("Collection checkpoint path differs from spec")
    if collection.get("base_train_rows") != base_rows:
        raise ValueError("Collection base row count differs from actual base dataset")
    if collection.get("added_rows") != len(augmented_train["actions"]) - base_rows:
        raise ValueError("Collection added row count differs from augmented dataset")
    if collection.get("input_hashes_before") != expected_input_hashes or collection.get("input_hashes_after") != expected_input_hashes:
        raise ValueError("Collection input hash proof differs from actual inputs")
    if collection.get("output_hashes") != {"train": train_sha, "dev": dev_sha}:
        raise ValueError("Collection output hashes differ from actual dataset files")

    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Collector manifest missing provenance")
    if provenance.get("kind") != "lobby-045-visited-training":
        raise ValueError("Collector provenance kind changed")
    if provenance.get("model_seed") != collection.get("model_seed"):
        raise ValueError("Collector provenance model_seed differs from collection")
    if expected_checkpoints and provenance.get("checkpoint") not in expected_checkpoints:
        raise ValueError("Collector provenance checkpoint differs from spec")
    for key in ("checkpoint_model_sha256", "checkpoint_metadata_sha256", "base_train_rows", "added_rows", "input_hashes_before", "input_hashes_after"):
        if provenance.get(key) != collection.get(key):
            raise ValueError(f"Collector provenance {key} differs from collection")
    if provenance.get("trajectory") != {
        "prefix": "teacher-train",
        "append": "current-parent-model-visited-states-with-heuristic-labels",
        "row_start": base_rows,
        "row_end": len(augmented_train["actions"]),
    }:
        raise ValueError("Collector provenance trajectory differs from actual row layout")

    cursor = base_rows
    ranks: dict[str, float] = {}
    trace_hashes: dict[str, str] = {}
    total_rows = 0
    teacher_hist: Counter[str] = Counter()
    executed_hist: Counter[str] = Counter()
    model_matches = 0
    for episode in episodes:
        seed = _episode_seed(episode)
        expected_start, expected_stop = _row_range(episode)
        if expected_start != cursor:
            raise ValueError("Collector row ranges are not contiguous after the teacher prefix")
        recorded_actions = _actions(episode, "executed_actions", "actions", "decisions", "parent_pred_actions")
        recorded_labels = _actions(episode, "teacher_actions", "teacher_labels", "labels")
        obs, _ = env.reset(seed=seed)
        rewards: list[float] = []
        predicted_actions: list[int] = []
        teacher_actions: list[int] = []
        done = False
        row = expected_start
        while not done:
            mask = env.action_masks()
            if done or not env.observation_space.contains(obs):
                raise ValueError("Collector replay reached invalid observation")
            view = env.game.view(env.learner_seat)
            teacher_action = int(heuristic_policy(view))
            predicted_action = int(model.predict(obs, deterministic=True, action_masks=mask)[0])
            if not bool(mask[predicted_action]) or not bool(mask[teacher_action]):
                raise ValueError("Collector or teacher produced illegal action")
            _assert_obs_row(augmented_train, row, obs, teacher_action, seed)
            predicted_actions.append(predicted_action)
            teacher_actions.append(teacher_action)
            model_matches += int(predicted_action == recorded_actions[len(predicted_actions) - 1])
            teacher_hist[str(teacher_action)] += 1
            executed_hist[str(predicted_action)] += 1
            obs, reward, terminated, truncated, _ = env.step(predicted_action)
            env.game.assert_conservation()
            rewards.append(float(reward))
            done = bool(terminated or truncated)
            row += 1
            if truncated:
                raise ValueError("Collector trajectory truncated")
        if predicted_actions != recorded_actions or teacher_actions != recorded_labels:
            raise ValueError(f"Collector recorded actions or teacher labels differ: seed={seed}")
        if row != expected_stop:
            raise ValueError(f"Collector row range stop differs: seed={seed}")
        rank = float(env.game.players[env.learner_seat].rank)
        reward_sum = float(sum(rewards))
        trace_sha = json_digest(env.game.trace)
        if rank != _rank(episode) or reward_sum != _reward(episode) or trace_sha != _trace_hash(episode):
            raise ValueError(f"Collector rank/reward/trace differs: seed={seed}")
        ranks[str(seed)] = rank
        trace_hashes[str(seed)] = trace_sha
        total_rows += len(predicted_actions)
        cursor = row
    if cursor != len(augmented_train["actions"]):
        raise ValueError("Collector train file has unaccounted rows")
    return {
        "base_teacher_rows": base_rows,
        "collected_rows": total_rows,
        "augmented_train_rows": len(augmented_train["actions"]),
        "dev_sha256": dev_sha,
        "train_sha256": train_sha,
        "model_prediction_matches_collection": model_matches,
        "model_prediction_rows": total_rows,
        "teacher_action_histogram": dict(sorted(teacher_hist.items(), key=lambda item: int(item[0]))),
        "executed_action_histogram": dict(sorted(executed_hist.items(), key=lambda item: int(item[0]))),
        "natural_ranks": ranks,
        "trace_sha256": trace_hashes,
        "row_range": [base_rows, cursor],
    }


def _empty_prediction_metrics() -> dict[str, Any]:
    return {"rows": 0, "correct": 0, "confusion_matrix": np.zeros((37, 37), dtype=np.int64)}


def _record_prediction(metrics: dict[str, Any], teacher_action: int, predicted_action: int) -> None:
    metrics["rows"] += 1
    metrics["correct"] += int(teacher_action == predicted_action)
    metrics["confusion_matrix"][int(teacher_action), int(predicted_action)] += 1


def _summarize_prediction_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    matrix = metrics["confusion_matrix"]
    rows = int(metrics["rows"])
    per_action = {}
    for action in range(37):
        count = int(matrix[action].sum())
        if count:
            per_action[str(action)] = {"count": count, "accuracy": int(matrix[action, action]) / count}
    macro = float(np.mean([row["accuracy"] for row in per_action.values()])) if per_action else None
    return {
        "rows": rows,
        "correct": int(metrics["correct"]),
        "accuracy": int(metrics["correct"]) / rows if rows else None,
        "macro_action_accuracy": macro,
        "per_action": per_action,
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def visited_state_metrics_from_reports(root: Path, spec: dict[str, Any], env, load_model_fn, heuristic_policy) -> dict[str, Any]:
    """Compute teacher agreement on final evaluation states for all six 045 models."""
    result: dict[str, Any] = {}
    total_episodes = 0
    total_rows = 0
    for seed in spec["seeds"]:
        result[str(seed)] = {}
        for arm in ("control", "candidate"):
            train_dir = root / "outputs" / identity(("train", seed, arm))
            status = read(train_dir / "status.json")
            checkpoint = train_dir / status["checkpoint"]
            model = load_model_fn(checkpoint, env)
            report = read(root / "outputs" / identity(("evaluate", seed, arm)) / "final.json")
            if report["model_sha256"] != digest(checkpoint / "model.zip"):
                raise ValueError("Visited-state report points at wrong model")
            metrics = _empty_prediction_metrics()
            for episode in report["episodes"]:
                obs, _ = env.reset(seed=episode["seed"])
                done = False
                for index, action in enumerate(episode["decisions"]):
                    mask = env.action_masks()
                    if done or not bool(mask[int(action)]) or not env.observation_space.contains(obs):
                        raise ValueError("Invalid visited-state replay action")
                    predicted = int(model.predict(obs, deterministic=True, action_masks=mask)[0])
                    if predicted != int(action):
                        raise ValueError("Visited-state saved action differs from loaded model")
                    teacher = int(heuristic_policy(env.game.view(env.learner_seat)))
                    _record_prediction(metrics, teacher, predicted)
                    obs, _, terminated, truncated, _ = env.step(predicted)
                    env.game.assert_conservation()
                    done = bool(terminated or truncated)
                    if truncated:
                        raise ValueError("Visited-state replay truncated")
                    if index == len(episode["decisions"]) - 1 and not done:
                        raise ValueError("Visited-state episode ended early in report")
                if env.game.players[env.learner_seat].rank != episode["rank"]:
                    raise ValueError("Visited-state rank differs")
                total_episodes += 1
            summary = _summarize_prediction_metrics(metrics) | {
                "episodes": len(report["episodes"]),
                "denominator_source": "final model visited states",
            }
            recorded = read(root / "outputs" / identity(("evaluate", seed, arm)) / "visited.json")
            for key, value in summary.items():
                if key == "per_action":
                    actual = {a: {k: recorded[key][a][k] for k in v} for a, v in value.items()}
                else:
                    actual = recorded[key]
                if actual != value:
                    raise ValueError(f"Visited-state metrics differ from runner output: {key}")
            result[str(seed)][arm] = summary
            total_rows += summary["rows"]
    result["summary"] = {"episodes": total_episodes, "rows": total_rows}
    expected_episodes = len(spec["seeds"]) * 2 * len(spec["eval_seeds"])
    if total_episodes != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} final visited-state games, got {total_episodes}")
    return result


def _call_decide(decide, labels: dict, games: dict, visited: dict) -> dict:
    parameters = inspect.signature(decide).parameters
    if len(parameters) >= 3:
        return decide(labels, games, visited)
    return decide(labels, games)


def _collector_checkpoint(root: Path, spec: dict[str, Any], seed: int) -> Path:
    row = spec["collectors"][str(seed)]
    checkpoint = root / row["checkpoint"]
    if digest(checkpoint / "model.zip") != row["model_sha256"]:
        raise ValueError("Collector model hash differs from spec")
    if digest(checkpoint / "metadata.json") != row["metadata_sha256"]:
        raise ValueError("Collector metadata hash differs from spec")
    return checkpoint


def verify_provenance(base, collected, checkpoint, seed):
    record = read(collected / "collection.json")
    manifest = read(collected / "manifest.json")
    original = read(base / "manifest.json")
    expected = {"base_train": digest(base / "train.npz"), "base_dev": digest(base / "dev.npz"),
                "checkpoint_model": digest(checkpoint / "model.zip"),
                "checkpoint_metadata": digest(checkpoint / "metadata.json")}
    for data in (record, manifest["provenance"]):
        if data["input_hashes_before"] != expected or data["input_hashes_after"] != expected:
            raise ValueError("Collection input provenance differs")
        if data["model_seed"] != seed or data["base_train_rows"] != original["split_counts"]["train"]:
            raise ValueError("Collection source identity differs")
        if data["added_rows"] != manifest["split_counts"]["train"] - original["split_counts"]["train"]:
            raise ValueError("Collection provenance row count differs")
        if data["checkpoint_model_sha256"] != expected["checkpoint_model"] or data["checkpoint_metadata_sha256"] != expected["checkpoint_metadata"]:
            raise ValueError("Collection checkpoint provenance differs")
    if record["output_hashes"] != manifest["split_hashes"]:
        raise ValueError("Collection output hashes differ")
    for key in ("config", "core_config", "compatibility"):
        if manifest[key] != original[key]:
            raise ValueError("Collection base contract differs")


def _training_dataset(root: Path, seed: int, arm: str) -> Path:
    if arm == "control":
        return root / "outputs" / BASE_DATASET_ID
    return root / "outputs" / identity(("collect", seed, "candidate"))


def _expected_initial_sampling(sampling_mode: str) -> dict[str, Any]:
    return {"mode": sampling_mode, "protocol": "choice-probabilities-v1", "hash": None, "histogram": {}}


def audit(root: Path) -> dict[str, Any]:
    started = time.monotonic()
    root = Path(root)
    if (root / "audit.json").exists():
        raise FileExistsError("Preserve previous audit")
    verify_files(root)
    spec = read(_spec_path(root))
    validate_045_spec(spec)
    sys.path.insert(0, str(root / "scripts"))
    from lobby_045 import decide, jobs as runner_jobs

    planned_jobs = expected_jobs(spec)
    if runner_jobs(spec) != planned_jobs:
        raise ValueError("045 runner job order differs from audit contract")
    state = read(root / "outputs/driver-status.json")
    if state["status"] != "completed" or state["completed_phases"] != [identity(job) for job in planned_jobs]:
        raise ValueError("Incomplete 045 campaign")
    peak, output_bytes = _verify_driver_artifacts(root, spec, planned_jobs)

    sys.path.insert(0, str(root / "src"))
    import torch
    from hearthstone_ai import lobby_env
    from hearthstone_ai.lobby_policies import heuristic_policy, random_policy
    from hearthstone_ai.lobby_training import load_model

    if not Path(lobby_env.__file__).resolve().is_relative_to(root / "src"):
        raise ValueError("Use frozen source")
    src_files = list((root / "src").rglob("*.py"))
    if len(src_files) != EXPECTED_SRC_FILES:
        raise ValueError("045 source file count changed")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    env = lobby_env.LobbyEnv(observation_scale="fixed-v1")
    base_dataset = root / "outputs" / BASE_DATASET_ID
    base_train = verify_dataset(base_dataset / "train.npz", DATASET_TRAIN_SEEDS, env, heuristic_policy)
    base_dev = verify_dataset(base_dataset / "dev.npz", DATASET_DEV_SEEDS, env, heuristic_policy)

    collectors: dict[str, dict] = {}
    input_hashes_before = {}
    for seed in TRAIN_SEEDS:
        checkpoint = _collector_checkpoint(root, spec, seed)
        input_hashes_before[f"collector-s{seed:03d}-model"] = digest(checkpoint / "model.zip")
        input_hashes_before[f"collector-s{seed:03d}-metadata"] = digest(checkpoint / "metadata.json")
        verify_provenance(base_dataset, root / "outputs" / identity(("collect", seed, "candidate")), checkpoint, seed)
        collector_model = load_model(checkpoint, env)
        collectors[str(seed)] = verify_collector_dataset(
            base_dataset,
            root / "outputs" / identity(("collect", seed, "candidate")),
            spec["collectors"][str(seed)],
            env,
            collector_model,
            heuristic_policy,
        )

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
            dataset = _training_dataset(root, seed, arm)
            train_data = load_npz(dataset / "train.npz")
            dev_data = load_npz(dataset / "dev.npz")
            if digest(dataset / "dev.npz") != digest(base_dataset / "dev.npz"):
                raise ValueError("Training dev split is not byte-identical to base dev")
            status = read(folder / "status.json")
            if read(folder / "dataset_manifest.json") != read(dataset / "manifest.json"):
                raise ValueError("Training dataset manifest differs from selected source")
            if (
                status.get("optimizer_steps") != EXPECTED_IMITATION["updates"]
                or status.get("rows_seen") != EXPECTED_IMITATION["updates"] * EXPECTED_IMITATION["batch_size"]
                or status.get("stop_reason") != "updates"
                or status.get("num_timesteps") != 0
                or status.get("sb3_n_updates") != 0
                or status.get("critic_unchanged") is not True
                or status.get("finite_parameters") is not True
            ):
                raise ValueError("Incomplete 045 imitation")
            proof = verify_sampling(
                folder,
                train_data["actions"],
                sampling_mode,
                seed,
                EXPECTED_IMITATION["updates"],
                EXPECTED_IMITATION["batch_size"],
            )
            if status.get("sampling") != proof["metadata"]:
                raise ValueError("Wrong 045 status sampling metadata")
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
                    proof["metadata"] if point == "final" else _expected_initial_sampling(sampling_mode),
                )
                supervised = meta.get("supervised", {})
                if supervised.get("dataset_train_rows") != len(train_data["actions"]):
                    raise ValueError("Checkpoint recorded wrong training dataset size")
                if supervised.get("dataset_dev_rows") != len(dev_data["actions"]):
                    raise ValueError("Checkpoint recorded wrong dev dataset size")
                _assert_optional_dataset_hashes(supervised, digest(dataset / "train.npz"), digest(dataset / "dev.npz"), "Checkpoint supervised metadata")
                data, weights[point] = model_evidence(checkpoint / "model.zip")
                if data["num_timesteps"] != 0 or data["_n_updates"] != 0:
                    raise ValueError("Supervised work mislabeled as PPO steps")
                if data["policy_kwargs"].get("net_arch") != [32, 32]:
                    raise ValueError("Policy architecture changed")
                model = load_model(checkpoint, env)
                labels[seed][arm][point] = {"train": classify(model, train_data), "dev": classify(model, dev_data)}
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
    visited = visited_state_metrics_from_reports(root, spec, env, load_model, heuristic_policy)
    env.close()
    input_hashes_after = {}
    for seed in TRAIN_SEEDS:
        checkpoint = _collector_checkpoint(root, spec, seed)
        input_hashes_after[f"collector-s{seed:03d}-model"] = digest(checkpoint / "model.zip")
        input_hashes_after[f"collector-s{seed:03d}-metadata"] = digest(checkpoint / "metadata.json")
    if input_hashes_before != input_hashes_after:
        raise ValueError("Collector checkpoint hashes changed during audit")
    verify_files(root)
    result = {
        "status": "passed",
        "training_method": "behavior-cloning-diagnostic-v1",
        "ppo_steps": 0,
        "supervised_updates_per_model": 1024,
        "models": 6,
        "collection_episodes": 300,
        "dataset_rows": {
            "base_train": len(base_train["actions"]),
            "dev": len(base_dev["actions"]),
            "candidate_train": {seed: collectors[str(seed)]["augmented_train_rows"] for seed in map(str, TRAIN_SEEDS)},
        },
        "dataset_sha256": {"base_train": digest(base_dataset / "train.npz"), "dev": digest(base_dataset / "dev.npz")},
        "collectors": collectors,
        "labels": labels,
        "games": games,
        "visited_state_teacher_agreement": visited,
        "baselines": baselines,
        "optimizers": optimizers,
        "sampling": sampling,
        "decision": _call_decide(decide, labels, games, visited),
        "evaluation_episodes": 280,
        "policy_replays": 12,
        "replayed_actions": actions,
        "paired_initial_policies_equal": True,
        "critic_weights_unchanged": True,
        "collector_inputs_unchanged": True,
        "frozen_inputs_unchanged": True,
        "source_py_files": EXPECTED_SRC_FILES,
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
