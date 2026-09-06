"""Read-only visited-state diagnosis for lobby 043 final imitation models."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ACTION_KINDS = (
    ["end", "reroll", "upgrade", "freeze"]
    + ["buy"] * 7
    + ["sell"] * 7
    + ["play"] * 10
    + ["swap"] * 6
    + ["discover"] * 3
)
ACTION_INDICES = (
    [None, None, None, None]
    + list(range(7))
    + list(range(7))
    + list(range(10))
    + list(range(6))
    + list(range(3))
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json_fresh(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Preserve previous diagnostic: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(root: Path) -> None:
    entries = read_json(root / "FILES.json")
    for relative_name, expected in entries.items():
        path = (root / relative_name).resolve()
        if not path.is_relative_to(root) or (root / relative_name).is_symlink():
            raise ValueError(f"Unsafe frozen input path: {relative_name}")
        if digest(path) != expected:
            raise ValueError(f"Frozen manifest hash mismatch: {relative_name}")


def checkpoint_for(root: Path, seed: int, arm: str) -> Path:
    status = read_json(root / f"outputs/train-s{seed:03d}-{arm}/status.json")
    return root / f"outputs/train-s{seed:03d}-{arm}" / status["checkpoint"]


def input_hashes(root: Path, spec: dict[str, Any]) -> dict[str, str]:
    paths = [
        "FILES.json",
        "audit.json",
        "configs/lobby043_comparison.json",
        "outputs/dataset-s007-student/dev.npz",
    ]
    for seed in spec["seeds"]:
        for arm in sorted(spec["arms"]):
            paths.append(f"outputs/evaluate-s{seed:03d}-{arm}/final.json")
            paths.append(str((checkpoint_for(root, seed, arm) / "model.zip").relative_to(root)).replace("\\", "/"))
    return {name: digest(root / name) for name in sorted(paths)}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def obs_batch(data: dict[str, np.ndarray], start: int, stop: int) -> dict[str, np.ndarray]:
    return {key.removeprefix("obs__"): value[start:stop] for key, value in data.items() if key.startswith("obs__")}


def action_kind(action: int) -> str:
    return ACTION_KINDS[int(action)]


def action_index(action: int) -> int | None:
    return ACTION_INDICES[int(action)]


def semantic_error(true_action: int, pred_action: int) -> str:
    true_kind = action_kind(true_action)
    pred_kind = action_kind(pred_action)
    if true_kind == "end" and pred_kind != "end":
        return "missing_end_turn"
    if true_kind != "end" and pred_kind == "end":
        return "early_end_turn"
    if true_kind == pred_kind and action_index(true_action) != action_index(pred_action):
        return f"{true_kind}_slot_confusion"
    if true_kind == pred_kind:
        return f"{true_kind}_same_kind_other"
    return f"{true_kind}_as_{pred_kind}"


def confusion_to_json(matrix: np.ndarray) -> list[list[int]]:
    return [[int(value) for value in row] for row in matrix.tolist()]


def empty_metrics() -> dict[str, Any]:
    return {"rows": 0, "correct": 0, "confusion_matrix": np.zeros((37, 37), dtype=np.int64)}


def record_prediction(metrics: dict[str, Any], teacher_action: int, predicted_action: int) -> None:
    metrics["rows"] += 1
    metrics["correct"] += int(teacher_action == predicted_action)
    metrics["confusion_matrix"][int(teacher_action), int(predicted_action)] += 1


def summarize_prediction_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    matrix = metrics["confusion_matrix"]
    rows = int(metrics["rows"])
    per_action = {}
    kind_confusion: dict[str, Counter] = defaultdict(Counter)
    semantic_errors: Counter[str] = Counter()
    for true_action in range(37):
        count = int(matrix[true_action].sum())
        if count:
            correct = int(matrix[true_action, true_action])
            per_action[str(true_action)] = {
                "kind": action_kind(true_action),
                "count": count,
                "correct": correct,
                "errors": count - correct,
                "accuracy": correct / count if count else None,
            }
        for pred_action in range(37):
            cell = int(matrix[true_action, pred_action])
            if not cell:
                continue
            kind_confusion[action_kind(true_action)][action_kind(pred_action)] += cell
            if true_action != pred_action:
                semantic_errors[semantic_error(true_action, pred_action)] += cell
    action_accuracies = [row["accuracy"] for row in per_action.values() if row["accuracy"] is not None]
    return {
        "rows": rows,
        "correct": int(metrics["correct"]),
        "incorrect": rows - int(metrics["correct"]),
        "accuracy": int(metrics["correct"]) / rows if rows else None,
        "actual_teacher_classes": sorted(per_action, key=lambda value: int(value)),
        "macro_action_accuracy": float(np.mean(action_accuracies)) if action_accuracies else None,
        "per_action": per_action,
        "confusion_total": int(matrix.sum()),
        "confusion_matrix": confusion_to_json(matrix),
        "kind_confusion": {
            true: {pred: int(count) for pred, count in sorted(counter.items())}
            for true, counter in sorted(kind_confusion.items())
        },
        "semantic_errors": {key: int(value) for key, value in sorted(semantic_errors.items())},
    }


def stratum_for_round(round_number: int) -> str:
    if round_number <= 3:
        return "round_1_3"
    if round_number <= 6:
        return "round_4_6"
    return "round_7_plus"


def decode_fixed_state_features(obs: dict[str, np.ndarray]) -> dict[str, int]:
    player = np.asarray(obs["player"], dtype=np.float32)
    board = np.asarray(obs["board"], dtype=np.float32)
    hand = np.asarray(obs["hand"], dtype=np.float32)
    if player.shape != (9,) or board.shape[0] != 7 or hand.shape[0] != 10:
        raise ValueError("Unexpected fixed-v1 observation shapes")
    gold = int(round(float(player[3]) * 10))
    tier = int(round(float(player[4]) * 2))
    board_count = int(np.sum(board[:, 0] < 0.5))
    hand_count = int(np.sum(hand[:, 0] < 0.5))
    if not 0 <= gold <= 10 or not 1 <= tier <= 6 or not 0 <= board_count <= 7 or not 0 <= hand_count <= 10:
        raise ValueError(f"Decoded fixed-v1 feature out of range: gold={gold} tier={tier} board={board_count} hand={hand_count}")
    return {"gold": gold, "tier": tier, "board_count": board_count, "hand_count": hand_count}


def feature_accumulator() -> dict[str, Any]:
    return {"rows": 0, "sums": Counter(), "histograms": defaultdict(Counter)}


def record_features(acc: dict[str, Any], features: dict[str, int]) -> None:
    acc["rows"] += 1
    for key, value in features.items():
        acc["sums"][key] += value
        acc["histograms"][key][str(value)] += 1


def summarize_features(acc: dict[str, Any]) -> dict[str, Any]:
    rows = int(acc["rows"])
    return {
        "rows": rows,
        "means": {key: acc["sums"][key] / rows if rows else None for key in sorted(acc["sums"])},
        "histograms": {
            key: {bucket: int(count) for bucket, count in sorted(counter.items(), key=lambda item: int(item[0]))}
            for key, counter in sorted(acc["histograms"].items())
        },
    }


def validate_rank(rank: float) -> float:
    doubled = float(rank) * 2.0
    if abs(doubled - round(doubled)) > 1e-9:
        raise ValueError(f"Rank is not integer/half-integer: {rank}")
    return float(rank)


def load_model(root: Path, seed: int, arm: str, env):
    from hearthstone_ai.lobby_training import load_model as strict_load_model

    checkpoint = checkpoint_for(root, seed, arm)
    before = digest(checkpoint / "model.zip")
    model = strict_load_model(checkpoint, env)
    if digest(checkpoint / "model.zip") != before:
        raise ValueError("Model load mutated checkpoint")
    return model, before


def dataset_metrics(model, data: dict[str, np.ndarray]) -> dict[str, Any]:
    metrics = empty_metrics()
    actions = data["actions"].astype(np.int64, copy=False)
    seed_metrics: dict[str, dict[str, int]] = defaultdict(lambda: {"rows": 0, "correct": 0})
    model.policy.set_training_mode(False)
    for start in range(0, len(actions), 256):
        stop = min(start + 256, len(actions))
        obs = obs_batch(data, start, stop)
        labels = actions[start:stop]
        masks = obs["action_mask"].astype(bool, copy=False)
        preds = model.predict(obs, deterministic=True, action_masks=masks)[0].astype(np.int64)
        for label, pred, seed in zip(labels, preds, data["seeds"][start:stop], strict=True):
            record_prediction(metrics, int(label), int(pred))
            row = seed_metrics[str(int(seed))]
            row["rows"] += 1
            row["correct"] += int(label == pred)
    result = summarize_prediction_metrics(metrics)
    for row in seed_metrics.values():
        row["accuracy"] = row["correct"] / row["rows"] if row["rows"] else None
    result["seed_metrics"] = dict(sorted(seed_metrics.items(), key=lambda item: int(item[0])))
    result["denominator_source"] = "teacher dev NPZ rows"
    return result


def dataset_features(data: dict[str, np.ndarray]) -> dict[str, Any]:
    acc = feature_accumulator()
    rows = len(data["actions"])
    for index in range(rows):
        obs = {key.removeprefix("obs__"): value[index] for key, value in data.items() if key.startswith("obs__")}
        record_features(acc, decode_fixed_state_features(obs))
    summary = summarize_features(acc)
    summary["denominator_source"] = "teacher dev NPZ rows"
    return summary


def validate_dataset(data: dict[str, np.ndarray], expected_rows: int) -> None:
    required = {"actions", "seeds"} | {
        f"obs__{key}" for key in ("player", "public_players", "shop", "board", "hand", "discover", "action_mask")
    }
    if set(data) != required:
        raise ValueError("Unexpected dev dataset fields")
    if len(data["actions"]) != expected_rows or any(len(value) != expected_rows for value in data.values()):
        raise ValueError("Unexpected dev dataset length")
    mask = data["obs__action_mask"]
    actions = data["actions"].astype(np.int64, copy=False)
    if mask.shape != (expected_rows, 37) or not np.isin(mask, [0, 1]).all():
        raise ValueError("Bad action mask")
    if not mask[np.arange(expected_rows), actions].all():
        raise ValueError("Dev labels must be legal under mask")


def replay_arm(root: Path, spec: dict[str, Any], seed: int, arm: str, model) -> dict[str, Any]:
    from hearthstone_ai.lobby_env import LobbyEnv
    from hearthstone_ai.lobby_policies import heuristic_policy

    report = read_json(root / f"outputs/evaluate-s{seed:03d}-{arm}/final.json")
    if [episode["seed"] for episode in report["episodes"]] != spec["eval_seeds"]:
        raise ValueError(f"Unexpected eval seeds for s{seed:03d}-{arm}")
    metrics = empty_metrics()
    strata = {name: empty_metrics() for name in ("round_1_3", "round_4_6", "round_7_plus")}
    first_mismatch_kinds: Counter[str] = Counter()
    first_mismatch_actions: Counter[str] = Counter()
    feature_acc = feature_accumulator()
    ranks = {}
    actions = Counter()
    episodes = 0
    states = 0
    saved_matches = 0
    for episode in report["episodes"]:
        env = LobbyEnv(
            learner_seat=spec["train"]["learner_seat"],
            max_rounds=spec["train"]["max_rounds"],
            heuristic_opponents=spec["train"]["heuristic_opponents"],
            observation_scale=spec["train"]["observation_scale"],
        )
        obs, _ = env.reset(seed=episode["seed"])
        done = False
        rewards = []
        saw_mismatch = False
        for step_index, action in enumerate(episode["decisions"]):
            mask = env.action_masks()
            if done or not mask[int(action)] or not env.observation_space.contains(obs):
                raise ValueError(f"Invalid saved state/action: s{seed:03d}-{arm} eval={episode['seed']} step={step_index}")
            student_prediction = int(model.predict(obs, deterministic=True, action_masks=mask)[0])
            if student_prediction != int(action):
                raise ValueError(
                    f"Stored action differs from frozen model prediction: s{seed:03d}-{arm} "
                    f"eval={episode['seed']} step={step_index} stored={action} predicted={student_prediction}"
                )
            saved_matches += 1
            view = env.game.view(env.learner_seat)
            teacher_action = int(heuristic_policy(view))
            record_prediction(metrics, teacher_action, student_prediction)
            record_prediction(strata[stratum_for_round(int(env.game.round))], teacher_action, student_prediction)
            if teacher_action != student_prediction and not saw_mismatch:
                first_mismatch_kinds[f"{action_kind(teacher_action)}->{action_kind(student_prediction)}"] += 1
                first_mismatch_actions[f"{teacher_action}->{student_prediction}"] += 1
                saw_mismatch = True
            record_features(feature_acc, decode_fixed_state_features(obs))
            actions[str(action)] += 1
            obs, reward, terminated, truncated, info = env.step(int(action))
            env.game.assert_conservation()
            rewards.append(reward)
            done = bool(terminated or truncated)
        rank = validate_rank(env.game.players[env.learner_seat].rank)
        if (
            not done
            or rank != episode["rank"]
            or sum(rewards) != episode["reward"]
            or terminated != episode["terminated"]
            or truncated != episode["truncated"]
            or info != episode["info"]
            or json.loads(json.dumps(env.game.trace)) != episode["trace"]
        ):
            raise ValueError(f"Saved trace replay mismatch: s{seed:03d}-{arm} eval={episode['seed']}")
        episodes += 1
        states += len(episode["decisions"])
        ranks[str(episode["seed"])] = rank
        env.close()
    metrics_summary = summarize_prediction_metrics(metrics)
    return {
        "training_seed": seed,
        "arm": arm,
        "episodes": episodes,
        "states": states,
        "saved_action_model_agreement": {"matches": saved_matches, "denominator": states, "accuracy": saved_matches / states if states else None},
        "visited_state_teacher_agreement": {**metrics_summary, "denominator_source": "model visited states from saved final traces"},
        "round_strata": {name: summarize_prediction_metrics(row) for name, row in strata.items()},
        "first_mismatch_kinds": {key: int(value) for key, value in sorted(first_mismatch_kinds.items())},
        "first_mismatch_actions": {key: int(value) for key, value in sorted(first_mismatch_actions.items())},
        "visited_state_features": {**summarize_features(feature_acc), "denominator_source": "model visited states from saved final traces"},
        "actions": {key: int(value) for key, value in sorted(actions.items(), key=lambda item: int(item[0]))},
        "ranks": ranks,
        "mean_rank": sum(ranks.values()) / len(ranks) if ranks else None,
        "report_model_sha256": report["model_sha256"],
        "report_mean_rank": report["mean_rank"],
        "report_mean_reward": report["mean_reward"],
        "report_top4_fraction_ranked": report["top4_fraction_ranked"],
        "report_truncated": report["truncated"],
    }


def diagnose(run_root: Path, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    if output.exists():
        raise FileExistsError(f"Preserve previous diagnostic: {output}")
    run_root = run_root.resolve()
    verify_manifest(run_root)
    spec = read_json(run_root / "configs/lobby043_comparison.json")
    before_hashes = input_hashes(run_root, spec)
    sys.path.insert(0, str(run_root / "src"))

    import torch
    from hearthstone_ai import lobby_env
    from hearthstone_ai.lobby_env import LobbyEnv

    if not Path(lobby_env.__file__).resolve().is_relative_to(run_root / "src"):
        raise ValueError("Frozen run source was not imported")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    dev = load_npz(run_root / "outputs/dataset-s007-student/dev.npz")
    validate_dataset(dev, 2236)
    dev_feature_summary = dataset_features(dev)

    models: dict[tuple[int, str], Any] = {}
    model_hashes = {}
    dev_metrics: dict[str, Any] = {}
    env = LobbyEnv(
        learner_seat=spec["train"]["learner_seat"],
        max_rounds=spec["train"]["max_rounds"],
        heuristic_opponents=spec["train"]["heuristic_opponents"],
        observation_scale=spec["train"]["observation_scale"],
    )
    try:
        for seed in spec["seeds"]:
            dev_metrics[str(seed)] = {}
            for arm in sorted(spec["arms"]):
                model, model_sha = load_model(run_root, seed, arm, env)
                models[(seed, arm)] = model
                model_hashes[f"s{seed:03d}-{arm}-final"] = model_sha
                dev_metrics[str(seed)][arm] = dataset_metrics(model, dev)
    finally:
        env.close()

    replay: dict[str, Any] = {}
    gate_by_seed = {}
    for seed in spec["seeds"]:
        replay[str(seed)] = {}
        candidate_gap = None
        for arm in sorted(spec["arms"]):
            row = replay_arm(run_root, spec, seed, arm, models[(seed, arm)])
            replay[str(seed)][arm] = row
            dev_acc = dev_metrics[str(seed)][arm]["accuracy"]
            visited_acc = row["visited_state_teacher_agreement"]["accuracy"]
            row["teacher_dev_accuracy_comparison"] = {
                "dev_accuracy": dev_acc,
                "dev_denominator": dev_metrics[str(seed)][arm]["rows"],
                "dev_actual_teacher_classes": dev_metrics[str(seed)][arm]["actual_teacher_classes"],
                "visited_accuracy": visited_acc,
                "visited_denominator": row["states"],
                "visited_actual_teacher_classes": row["visited_state_teacher_agreement"]["actual_teacher_classes"],
                "accuracy_gap_dev_minus_visited": dev_acc - visited_acc,
            }
            if arm == "candidate":
                candidate_gap = dev_acc - visited_acc
        gate_by_seed[str(seed)] = {
            "candidate_accuracy_gap_dev_minus_visited": candidate_gap,
            "gap_at_least_20pp": bool(candidate_gap is not None and candidate_gap >= 0.20),
        }

    after_hashes = input_hashes(run_root, spec)
    verify_manifest(run_root)
    if before_hashes != after_hashes:
        raise ValueError("Frozen model/data/trace hashes changed during diagnosis")

    gate_count = sum(int(row["gap_at_least_20pp"]) for row in gate_by_seed.values())
    result = {
        "status": "passed",
        "diagnostic": "lobby044-visited-state-diagnosis",
        "run": str(run_root),
        "frozen_inputs_unchanged": True,
        "input_hashes_before": before_hashes,
        "input_hashes_after": after_hashes,
        "cpu_threads": {"torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads()},
        "dataset": {"dev": {"rows": int(len(dev["actions"])), "sha256": digest(run_root / "outputs/dataset-s007-student/dev.npz")}},
        "model_hashes": model_hashes,
        "teacher_dev_features": dev_feature_summary,
        "teacher_dev_model_metrics": dev_metrics,
        "final_trace_replay": replay,
        "candidate_gap_gate": {
            "definition": "candidate teacher-dev action accuracy minus candidate visited-state teacher agreement >= 0.20 in at least 2 of 3 training seeds",
            "by_seed": gate_by_seed,
            "passing_seed_count": gate_count,
            "gate_met": gate_count >= 2,
        },
        "claim_boundary": [
            "Read-only diagnostic of frozen 043 final models, dev teacher NPZ, and saved final traces.",
            "Visited-state metrics use model-visited states from saved final traces; teacher-dev metrics use the separate 2,236-row dev NPZ.",
            "State-distribution differences are descriptive only and do not establish causality or performance improvement.",
            "No training, model modification, or reserved final evaluation data use is performed.",
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    total_episodes = sum(row[arm]["episodes"] for row in replay.values() for arm in row)
    if total_episodes != 120:
        raise ValueError(f"Expected 120 final trace episodes, got {total_episodes}")
    write_json_fresh(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/lobby-043-local-r1"))
    parser.add_argument("--output", type=Path, default=Path("experiments/diagnostics/lobby044_visited_states.json"))
    args = parser.parse_args()
    print(json.dumps(diagnose(args.run, args.output), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
