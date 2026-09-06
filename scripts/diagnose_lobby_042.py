"""Read-only classification error diagnosis for lobby 041 imitation models."""

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


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dump_fresh(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(f"Preserve previous diagnostic: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_files(root: Path) -> None:
    entries = read(root / "FILES.json")
    for name, expected in entries.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or (root / name).is_symlink():
            raise ValueError(f"Unsafe frozen input path: {name}")
        if digest(path) != expected:
            raise ValueError(f"Frozen input hash mismatch: {name}")


def source_hashes(root: Path) -> dict[str, str]:
    names = [
        "FILES.json",
        "audit.json",
        "src/hearthstone_ai/lobby_imitation.py",
        "src/hearthstone_ai/lobby_env.py",
        "src/hearthstone_ai/lobby_training.py",
        "src/hearthstone_ai/lobby_policies.py",
        "src/hearthstone_ai/actions.py",
        "configs/lobby041_imitation.json",
        "outputs/dataset-s007-student/train.npz",
        "outputs/dataset-s007-student/dev.npz",
    ]
    return {name: digest(root / name) for name in names}


def frozen_input_hashes(root: Path, spec: dict[str, Any]) -> dict[str, str]:
    names = [
        "outputs/dataset-s007-student/train.npz",
        "outputs/dataset-s007-student/dev.npz",
    ]
    for seed in spec["seeds"]:
        for point in ("initial", "final"):
            names.append(str((checkpoint_for(root, seed, point) / "model.zip").relative_to(root)).replace("\\", "/"))
        names.append(f"outputs/evaluate-s{seed:03d}-student/final.json")
    return {name: digest(root / name) for name in sorted(names)}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def obs_batch(data: dict[str, np.ndarray], start: int, stop: int) -> dict[str, np.ndarray]:
    return {key.removeprefix("obs__"): value[start:stop] for key, value in data.items() if key.startswith("obs__")}


def action_kind(action: int) -> str:
    from hearthstone_ai.actions import decode_action

    return decode_action(int(action)).kind


def confusion_to_json(matrix: np.ndarray) -> list[list[int]]:
    return [[int(value) for value in row] for row in matrix.tolist()]


def summarize_matrix(matrix: np.ndarray) -> dict[str, Any]:
    per_action = {}
    present = [action for action in range(37) if int(matrix[action].sum())]
    for action in present:
        count = int(matrix[action].sum())
        correct = int(matrix[action, action])
        per_action[str(action)] = {
            "kind": action_kind(action),
            "count": count,
            "correct": correct,
            "errors": count - correct,
            "accuracy": correct / count if count else None,
        }
    kind_counts: dict[str, Counter] = defaultdict(Counter)
    semantic_errors: dict[str, int] = Counter()
    for true_action in range(37):
        true_kind = action_kind(true_action)
        for pred_action in range(37):
            count = int(matrix[true_action, pred_action])
            if not count:
                continue
            pred_kind = action_kind(pred_action)
            kind_counts[true_kind][pred_kind] += count
            if true_action == pred_action:
                continue
            semantic_errors[classify_semantic_error(true_action, pred_action)] += count
    action_accuracies = [row["accuracy"] for row in per_action.values() if row["accuracy"] is not None]
    return {
        "confusion_matrix": confusion_to_json(matrix),
        "confusion_total": int(matrix.sum()),
        "per_action": per_action,
        "macro_action_accuracy": float(np.mean(action_accuracies)) if action_accuracies else None,
        "kind_confusion": {
            true: {pred: int(count) for pred, count in sorted(counter.items())}
            for true, counter in sorted(kind_counts.items())
        },
        "semantic_errors": {key: int(value) for key, value in sorted(semantic_errors.items())},
        "error_contribution_by_action": {
            action: per_action[action]["errors"] for action in sorted(per_action, key=lambda x: int(x))
        },
    }


def summarize_confusion(actions: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    matrix = np.zeros((37, 37), dtype=np.int64)
    for label, pred in zip(actions, predicted, strict=True):
        matrix[int(label), int(pred)] += 1
    return summarize_matrix(matrix)


def classify_semantic_error(true_action: int, pred_action: int) -> str:
    from hearthstone_ai.actions import decode_action

    true = decode_action(int(true_action))
    pred = decode_action(int(pred_action))
    if true.kind == "end" and pred.kind != "end":
        return "missing_end_turn"
    if true.kind != "end" and pred.kind == "end":
        return "early_end_turn"
    if true.kind == pred.kind == "buy" and true.index != pred.index:
        return "buy_slot_confusion"
    if true.kind == pred.kind == "play" and true.index != pred.index:
        return "play_slot_confusion"
    if true.kind == pred.kind == "sell" and true.index != pred.index:
        return "sell_slot_confusion"
    if true.kind == pred.kind == "discover" and true.index != pred.index:
        return "discover_slot_confusion"
    if true.kind == pred.kind == "swap" and true.index != pred.index:
        return "swap_slot_confusion"
    if true.kind == pred.kind:
        return f"{true.kind}_same_kind_other"
    return f"{true.kind}_as_{pred.kind}"


def classify_dataset(model, data: dict[str, np.ndarray], *, include_predictions: bool = False) -> dict[str, Any]:
    import torch

    actions = data["actions"].astype(np.int64, copy=False)
    predictions = []
    nll = 0.0
    correct = 0
    seed_rows: dict[str, dict[str, int]] = defaultdict(lambda: {"rows": 0, "correct": 0})
    model.policy.set_training_mode(False)
    for start in range(0, len(actions), 256):
        stop = min(start + 256, len(actions))
        obs = obs_batch(data, start, stop)
        labels = actions[start:stop]
        masks = obs["action_mask"].astype(bool, copy=False)
        predicted = model.predict(obs, deterministic=True, action_masks=masks)[0].astype(np.int64)
        tensors, _ = model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            distribution = model.policy.get_distribution(tensors, action_masks=masks)
            log_prob = distribution.log_prob(torch.as_tensor(labels, dtype=torch.long, device=model.policy.device))
        if not torch.isfinite(log_prob).all():
            raise FloatingPointError("Non-finite label log probabilities")
        predictions.append(predicted)
        nll += float((-log_prob).sum().detach().cpu().item())
        correct += int(np.sum(predicted == labels))
        for seed, hit in zip(data["seeds"][start:stop], predicted == labels, strict=True):
            row = seed_rows[str(int(seed))]
            row["rows"] += 1
            row["correct"] += int(hit)
    predicted_all = np.concatenate(predictions) if predictions else np.zeros(0, dtype=np.int64)
    confusion = summarize_confusion(actions, predicted_all)
    rows = int(len(actions))
    for row in seed_rows.values():
        row["accuracy"] = row["correct"] / row["rows"] if row["rows"] else None
    result = {
        "rows": rows,
        "accuracy": correct / rows if rows else None,
        "nll": nll / rows if rows else None,
        "correct": correct,
        "incorrect": rows - correct,
        "seed_metrics": dict(sorted(seed_rows.items(), key=lambda item: int(item[0]))),
        **confusion,
    }
    if include_predictions:
        result["_predictions"] = predicted_all.astype(int).tolist()
    return result


def common_dev_errors(actions: np.ndarray, predictions_by_seed: dict[int, list[int]]) -> dict[str, Any]:
    labels = actions.astype(np.int64, copy=False)
    seed_order = sorted(predictions_by_seed)
    pred_arrays = {seed: np.asarray(predictions_by_seed[seed], dtype=np.int64) for seed in seed_order}
    for seed, predictions in pred_arrays.items():
        if predictions.shape != labels.shape:
            raise ValueError(f"Prediction length mismatch for seed {seed}")
    wrong_by_seed = {seed: predictions != labels for seed, predictions in pred_arrays.items()}
    union_wrong = np.zeros(labels.shape, dtype=bool)
    intersection_wrong = np.ones(labels.shape, dtype=bool)
    for wrong in wrong_by_seed.values():
        union_wrong |= wrong
        intersection_wrong &= wrong
    common_matrix = np.zeros((37, 37), dtype=np.int64)
    for row_index in np.flatnonzero(intersection_wrong):
        label = int(labels[row_index])
        for predictions in pred_arrays.values():
            common_matrix[label, int(predictions[row_index])] += 1
    per_label_common_rows = {
        str(action): int(np.sum(intersection_wrong & (labels == action)))
        for action in range(37)
        if int(np.sum(intersection_wrong & (labels == action)))
    }
    return {
        "rows": int(len(labels)),
        "training_seeds": seed_order,
        "union_wrong_rows": int(np.sum(union_wrong)),
        "union_wrong_fraction": float(np.mean(union_wrong)) if len(labels) else None,
        "intersection_wrong_rows": int(np.sum(intersection_wrong)),
        "intersection_wrong_fraction": float(np.mean(intersection_wrong)) if len(labels) else None,
        "per_label_intersection_wrong_rows": per_label_common_rows,
        "common_error_prediction_instances": summarize_matrix(common_matrix),
        "definition": "Union/intersection are computed on identical dev row indices across the three final models.",
    }


def validate_rank(rank: float) -> float:
    doubled = float(rank) * 2.0
    if abs(doubled - round(doubled)) > 1e-9:
        raise ValueError(f"Rank is not integer/half-integer: {rank}")
    return float(rank)

def validate_dataset(data: dict[str, np.ndarray], expected_rows: int) -> None:
    required = {"actions", "seeds"} | {
        f"obs__{key}" for key in ("player", "public_players", "shop", "board", "hand", "discover", "action_mask")
    }
    if set(data) != required:
        raise ValueError("Unexpected dataset fields")
    if len(data["actions"]) != expected_rows:
        raise ValueError("Dataset row count mismatch")
    if any(len(value) != expected_rows for value in data.values()):
        raise ValueError("Dataset array length mismatch")
    mask = data["obs__action_mask"]
    actions = data["actions"].astype(np.int64, copy=False)
    if mask.shape != (expected_rows, 37) or not np.isin(mask, [0, 1]).all():
        raise ValueError("Bad action mask")
    if not mask[np.arange(expected_rows), actions].all():
        raise ValueError("Dataset labels must be legal under mask")


def checkpoint_for(root: Path, seed: int, point: str) -> Path:
    folder = root / f"outputs/train-s{seed:03d}-student"
    status = read(folder / "status.json")
    return folder / (status["initial_checkpoint"] if point == "initial" else status["checkpoint"])


def load_student(root: Path, seed: int, point: str, env):
    from hearthstone_ai.lobby_training import load_model

    checkpoint = checkpoint_for(root, seed, point)
    before = digest(checkpoint / "model.zip")
    model = load_model(checkpoint, env)
    if digest(checkpoint / "model.zip") != before:
        raise ValueError("Model load mutated checkpoint")
    return model, before


def replay_final_games(root: Path, spec: dict[str, Any], models: dict[int, Any]) -> dict[str, Any]:
    from hearthstone_ai.lobby_env import LobbyEnv
    from hearthstone_ai.lobby_policies import heuristic_policy

    result = {}
    total_states = 0
    total_teacher_state_matches = 0
    total_saved_model_matches = 0
    for seed in spec["seeds"]:
        model = models[seed]
        report = read(root / f"outputs/evaluate-s{seed:03d}-student/final.json")
        if [episode["seed"] for episode in report["episodes"]] != spec["eval_seeds"]:
            raise ValueError("Unexpected final eval seeds")
        seed_rows = {
            "episodes": 0,
            "states": 0,
            "student_state_teacher_agreement": {"matches": 0, "denominator": 0, "accuracy": None},
            "saved_action_model_agreement": {"matches": 0, "denominator": 0, "accuracy": None},
            "ranks": {},
            "actions": Counter(),
            "teacher_actions_on_student_states": Counter(),
        }
        for episode in report["episodes"]:
            env = LobbyEnv(
                learner_seat=spec["train"]["learner_seat"],
                max_rounds=spec["train"]["max_rounds"],
                heuristic_opponents=7,
                observation_scale="fixed-v1",
            )
            obs, _ = env.reset(seed=episode["seed"])
            done = False
            rewards = []
            for action in episode["decisions"]:
                if done or not env.action_masks()[action] or not env.observation_space.contains(obs):
                    raise ValueError("Saved final trace contains invalid state/action")
                teacher_action = heuristic_policy(env.game.view(env.learner_seat))
                student_prediction = int(model.predict(obs, deterministic=True, action_masks=env.action_masks())[0])
                if student_prediction != action:
                    raise ValueError(
                        f"Saved final trace action differs from frozen model prediction: "
                        f"training_seed={seed} eval_seed={episode['seed']} saved={action} predicted={student_prediction}"
                    )
                hit = int(student_prediction == teacher_action)
                seed_rows["student_state_teacher_agreement"]["matches"] += hit
                seed_rows["student_state_teacher_agreement"]["denominator"] += 1
                seed_rows["saved_action_model_agreement"]["matches"] += 1
                seed_rows["saved_action_model_agreement"]["denominator"] += 1
                seed_rows["actions"][str(action)] += 1
                seed_rows["teacher_actions_on_student_states"][str(teacher_action)] += 1
                obs, reward, terminated, truncated, info = env.step(action)
                env.game.assert_conservation()
                rewards.append(reward)
                done = terminated or truncated
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
                raise ValueError("Saved final trace replay mismatch")
            seed_rows["episodes"] += 1
            seed_rows["states"] += len(episode["decisions"])
            seed_rows["ranks"][str(episode["seed"])] = rank
            env.close()
        agreement = seed_rows["student_state_teacher_agreement"]
        agreement["accuracy"] = agreement["matches"] / agreement["denominator"] if agreement["denominator"] else None
        saved_agreement = seed_rows["saved_action_model_agreement"]
        saved_agreement["accuracy"] = (
            saved_agreement["matches"] / saved_agreement["denominator"] if saved_agreement["denominator"] else None
        )
        seed_rows["actions"] = dict(seed_rows["actions"])
        seed_rows["teacher_actions_on_student_states"] = dict(seed_rows["teacher_actions_on_student_states"])
        result[str(seed)] = seed_rows
        total_states += seed_rows["states"]
        total_teacher_state_matches += agreement["matches"]
        total_saved_model_matches += saved_agreement["matches"]
    return {
        "by_training_seed": result,
        "total_episodes": sum(row["episodes"] for row in result.values()),
        "total_states": total_states,
        "student_state_teacher_agreement": {
            "matches": total_teacher_state_matches,
            "denominator": total_states,
            "accuracy": total_teacher_state_matches / total_states if total_states else None,
        },
        "saved_action_model_agreement": {
            "matches": total_saved_model_matches,
            "denominator": total_states,
            "accuracy": total_saved_model_matches / total_states if total_states else None,
        },
    }


def diagnose(run_root: Path, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    if output.exists():
        raise FileExistsError(f"Preserve previous diagnostic: {output}")
    run_root = run_root.resolve()
    verify_files(run_root)
    spec = read(run_root / "configs/lobby041_imitation.json")
    input_hashes_before = frozen_input_hashes(run_root, spec)
    sys.path.insert(0, str(run_root / "src"))

    import torch
    from hearthstone_ai import lobby_env
    from hearthstone_ai.lobby_env import LobbyEnv

    if not Path(lobby_env.__file__).resolve().is_relative_to(run_root / "src"):
        raise ValueError("Frozen run source was not imported")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    env = LobbyEnv(observation_scale="fixed-v1")
    dataset_dir = run_root / "outputs/dataset-s007-student"
    train = load_npz(dataset_dir / "train.npz")
    dev = load_npz(dataset_dir / "dev.npz")
    validate_dataset(train, 10752)
    validate_dataset(dev, 2236)

    classification: dict[str, Any] = {}
    final_models = {}
    model_hashes = {}
    final_dev_predictions: dict[int, list[int]] = {}
    for seed in spec["seeds"]:
        classification[str(seed)] = {}
        for point in ("initial", "final"):
            model, model_sha = load_student(run_root, seed, point, env)
            model_hashes[f"s{seed:03d}-{point}"] = model_sha
            dev_metrics = classify_dataset(model, dev, include_predictions=point == "final")
            if point == "final":
                final_dev_predictions[seed] = dev_metrics.pop("_predictions")
            classification[str(seed)][point] = {
                "train": classify_dataset(model, train),
                "dev": dev_metrics,
            }
            if point == "final":
                final_models[seed] = model
    games = replay_final_games(run_root, spec, final_models)
    verify_files(run_root)
    input_hashes_after = frozen_input_hashes(run_root, spec)
    if input_hashes_before != input_hashes_after:
        raise ValueError("Frozen model/dataset/trace hashes changed during diagnosis")

    result = {
        "status": "passed",
        "diagnostic": "lobby042-classification-error-diagnosis",
        "run": str(run_root),
        "frozen_inputs_unchanged": True,
        "frozen_input_hashes_before": input_hashes_before,
        "frozen_input_hashes_after": input_hashes_after,
        "cpu_threads": {"torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads()},
        "source_hashes": source_hashes(run_root),
        "model_hashes": model_hashes,
        "dataset": {
            "train": {"rows": int(len(train["actions"])), "sha256": digest(dataset_dir / "train.npz")},
            "dev": {"rows": int(len(dev["actions"])), "sha256": digest(dataset_dir / "dev.npz")},
        },
        "classification": classification,
        "final_dev_common_errors_across_models": common_dev_errors(dev["actions"], final_dev_predictions),
        "final_game_replay": games,
        "claim_boundary": [
            "Read-only diagnostic of saved 041 datasets, models, and final traces.",
            "Metrics use masked predictions and teacher-action NLL; no training or model modification is performed.",
            "Game replay separates teacher-trajectory dataset accuracy from teacher agreement on student-visited states.",
            "This does not establish a causal explanation or performance improvement.",
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    if games["total_episodes"] != 60:
        raise ValueError("Expected 60 final game traces")
    dump_fresh(output, result)
    env.close()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/lobby-041-local-r1"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/diagnostics/lobby042_classification.json"),
    )
    args = parser.parse_args()
    print(json.dumps(diagnose(args.run, args.output), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

