"""Source-specific read-only diagnosis for lobby 045 post-collection results."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from lobby_035 import identity  # noqa: E402

SEEDS = [7, 17, 27]
ARMS = ["control", "candidate"]
BASE_DATASET_ID = "dataset-s007-teacher"
SPEC_NAME = "lobby045_comparison.json"
DIAGNOSTIC_PROTOCOL = "lobby-046-source-diagnosis-v1"
ACTION_COUNT = 37


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json_fresh(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Preserve previous diagnostic: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def expected_jobs(spec: dict[str, Any]):
    return (
        [("dataset", 7, "teacher")]
        + [("collect", seed, "candidate") for seed in spec["seeds"]]
        + [("train", seed, arm) for seed in spec["seeds"] for arm in ("control", "candidate")]
        + [("evaluate", seed, arm) for seed in spec["seeds"] for arm in ("control", "candidate")]
        + [("baseline", 7, "control")]
    )


def verify_driver_receipts(run_root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    from audit_lobby_043 import _verify_driver_artifacts

    peak, output_bytes = _verify_driver_artifacts(run_root, spec, expected_jobs(spec))
    return {"peak_sampled_rss_bytes": peak, "outputs_and_returns_bytes": output_bytes}


def validate_spec(spec: dict[str, Any]) -> None:
    if spec.get("seeds") != SEEDS:
        raise ValueError("046 requires 045 seeds [7, 17, 27]")
    if sorted(spec.get("arms", {})) != sorted(ARMS):
        raise ValueError("046 requires control and candidate arms")
    if spec.get("eval_seeds") != list(range(20001, 20021)):
        raise ValueError("046 requires the saved 045 eval seed set 20001..20020")
    dataset = spec.get("dataset", {})
    if dataset.get("train_seeds") != list(range(100)) or dataset.get("dev_seeds") != list(range(20001, 20021)):
        raise ValueError("046 requires the 045 teacher dataset seed sets")
    train = spec.get("train", {})
    expected_train = {
        "threads": 1,
        "learner_seat": 0,
        "max_rounds": 100,
        "heuristic_opponents": 7,
        "observation_scale": "fixed-v1",
        "gamma": 1.0,
    }
    if any(train.get(key) != value for key, value in expected_train.items()):
        raise ValueError("046 requires fixed 045 train/environment settings")
    if spec.get("arms") != {"control": "balanced-choice", "candidate": "balanced-choice"}:
        raise ValueError("046 requires the 045 balanced-choice arms")


def checkpoint_for(run_root: Path, seed: int, arm: str) -> Path:
    train_dir = run_root / "outputs" / identity(("train", seed, arm))
    status = read_json(train_dir / "status.json")
    return train_dir / status["checkpoint"]


def used_input_paths(run_root: Path, _spec: dict[str, Any]) -> dict[str, Path]:
    paths = {}
    for path in run_root.rglob("*"):
        if path.is_file():
            relative = path.relative_to(run_root).as_posix()
            paths[relative] = path
    return dict(sorted(paths.items()))


def hash_inputs(paths: dict[str, Path]) -> dict[str, str]:
    return {name: digest(path) for name, path in paths.items()}


def verify_frozen_files_manifest(run_root: Path) -> None:
    entries = read_json(run_root / "FILES.json")
    for relative_name, expected in entries.items():
        path = (run_root / relative_name).resolve()
        if not path.is_relative_to(run_root.resolve()) or (run_root / relative_name).is_symlink():
            raise ValueError(f"Unsafe frozen FILES path: {relative_name}")
        if digest(path) != expected:
            raise ValueError(f"Frozen FILES hash mismatch: {relative_name}")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def validate_npz(data: dict[str, np.ndarray], *, rows: int | None = None, name: str = "dataset") -> None:
    required = {
        "obs__player",
        "obs__public_players",
        "obs__shop",
        "obs__board",
        "obs__hand",
        "obs__discover",
        "obs__action_mask",
        "actions",
        "seeds",
    }
    if set(data) != required:
        raise ValueError(f"{name} keys mismatch")
    if rows is not None and len(data["actions"]) != rows:
        raise ValueError(f"{name} row count mismatch")
    row_count = len(data["actions"])
    if row_count == 0:
        raise ValueError(f"{name} must not be empty")
    for key, value in data.items():
        if len(value) != row_count:
            raise ValueError(f"{name} {key} length mismatch")
        if key.startswith("obs__") and not np.isfinite(value).all():
            raise ValueError(f"{name} {key} contains nonfinite values")
    if data["actions"].ndim != 1 or data["seeds"].ndim != 1:
        raise ValueError(f"{name} actions/seeds shape mismatch")
    actions = data["actions"].astype(np.int64, copy=False)
    mask = data["obs__action_mask"]
    if mask.shape != (row_count, ACTION_COUNT) or not np.isin(mask, [0, 1]).all():
        raise ValueError(f"{name} action mask mismatch")
    if (actions < 0).any() or (actions >= ACTION_COUNT).any() or not mask[np.arange(row_count), actions].all():
        raise ValueError(f"{name} teacher actions must be legal")


def assert_prefix(base: dict[str, np.ndarray], augmented: dict[str, np.ndarray], rows: int) -> None:
    if rows <= 0:
        raise ValueError("Base train slice must be nonempty")
    if set(base) != set(augmented):
        raise ValueError("Augmented candidate train keys differ from base")
    for key, value in base.items():
        if not np.array_equal(value, augmented[key][:rows]):
            raise ValueError(f"Candidate train prefix differs from base: {key}")


def slice_npz(data: dict[str, np.ndarray], start: int, stop: int, name: str) -> dict[str, np.ndarray]:
    if not (0 <= start < stop <= len(data["actions"])):
        raise ValueError(f"Invalid nonempty source slice for {name}: [{start}, {stop})")
    result = {key: value[start:stop] for key, value in data.items()}
    validate_npz(result, rows=stop - start, name=name)
    return result


def collection_episodes(collection: dict[str, Any]) -> list[dict[str, Any]]:
    episodes = collection.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("collection.json missing nonempty episodes")
    return episodes


def validate_half_rank(rank: Any) -> float:
    value = float(rank)
    if abs(value * 2.0 - round(value * 2.0)) > 1e-9:
        raise ValueError(f"Rank is not integer/half-integer: {rank}")
    return value


def validate_collection_bounds(base_dir: Path, collect_dir: Path) -> dict[str, Any]:
    base_manifest = read_json(base_dir / "manifest.json")
    manifest = read_json(collect_dir / "manifest.json")
    collection = read_json(collect_dir / "collection.json")
    base_rows = int(base_manifest["split_counts"]["train"])
    augmented_rows = int(manifest["split_counts"]["train"])
    added_rows = int(collection["added_rows"])
    if augmented_rows != base_rows + added_rows or added_rows <= 0:
        raise ValueError("Collection train row counts are inconsistent")
    if manifest["split_hashes"]["train"] != digest(collect_dir / "train.npz"):
        raise ValueError("Collection train hash mismatch")
    if manifest["split_hashes"]["dev"] != digest(collect_dir / "dev.npz"):
        raise ValueError("Collection dev hash mismatch")
    if digest(base_dir / "dev.npz") != digest(collect_dir / "dev.npz"):
        raise ValueError("Collection dev split is not byte-identical to base dev")
    base_train = load_npz(base_dir / "train.npz")
    augmented_train = load_npz(collect_dir / "train.npz")
    validate_npz(base_train, rows=base_rows, name="base train")
    validate_npz(augmented_train, rows=augmented_rows, name="augmented train")
    assert_prefix(base_train, augmented_train, base_rows)
    cursor = base_rows
    for episode in collection_episodes(collection):
        start, stop = int(episode["row_start"]), int(episode["row_end"])
        if start != cursor or stop <= start:
            raise ValueError("Collection episode row ranges are not contiguous")
        teacher_actions = episode.get("teacher_actions")
        executed_actions = episode.get("executed_actions")
        if not isinstance(teacher_actions, list) or not isinstance(executed_actions, list):
            raise ValueError("Collection episode actions missing")
        if len(teacher_actions) != stop - start or len(executed_actions) != stop - start:
            raise ValueError("Collection episode action lengths differ from row range")
        if not np.array_equal(augmented_train["actions"][start:stop].astype(np.int64), np.asarray(teacher_actions, dtype=np.int64)):
            raise ValueError("Collection teacher labels differ from sliced rows")
        if not np.array_equal(augmented_train["seeds"][start:stop].astype(np.int64), np.full(stop - start, int(episode["seed"]), dtype=np.int64)):
            raise ValueError("Collection seeds differ from sliced rows")
        validate_half_rank(episode["rank"])
        trace_hash = episode.get("trace_sha256")
        if not (isinstance(trace_hash, str) and len(trace_hash) == 64):
            raise ValueError("Collection episode trace hash missing")
        cursor = stop
    if cursor != augmented_rows:
        raise ValueError("Collection row ranges do not cover all added rows")
    return {
        "base_train_rows": base_rows,
        "added_rows": added_rows,
        "augmented_train_rows": augmented_rows,
        "row_range": [base_rows, augmented_rows],
        "episode_count": len(collection_episodes(collection)),
    }




def validate_audit_evidence(run_root: Path, audit: dict[str, Any], collection_bounds: dict[str, Any] | None = None) -> None:
    if audit.get("status") != "passed":
        raise ValueError("046 requires passed 045 audit")
    if audit.get("decision", {}).get("candidate_keep") is not False:
        raise ValueError("046 expects the frozen 045 candidate_keep rejection")
    base_dir = run_root / "outputs" / BASE_DATASET_ID
    dataset_hashes = audit.get("dataset_sha256", {})
    if dataset_hashes.get("base_train") != digest(base_dir / "train.npz"):
        raise ValueError("045 audit base train hash differs from diagnosed input")
    if dataset_hashes.get("dev") != digest(base_dir / "dev.npz"):
        raise ValueError("045 audit dev hash differs from diagnosed input")
    if collection_bounds is not None:
        collectors = audit.get("collectors", {})
        for seed, bounds in collection_bounds.items():
            collect_dir = run_root / "outputs" / identity(("collect", int(seed), "candidate"))
            recorded = collectors.get(str(seed), {})
            if recorded.get("base_teacher_rows") != bounds["base_train_rows"]:
                raise ValueError("045 audit collector base row count differs")
            if recorded.get("augmented_train_rows") != bounds["augmented_train_rows"]:
                raise ValueError("045 audit collector augmented row count differs")
            if recorded.get("collected_rows") != bounds["added_rows"]:
                raise ValueError("045 audit collector added row count differs")
            if recorded.get("train_sha256") != digest(collect_dir / "train.npz"):
                raise ValueError("045 audit collector train hash differs")
            if recorded.get("dev_sha256") != digest(collect_dir / "dev.npz"):
                raise ValueError("045 audit collector dev hash differs")

def action_histogram(actions: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(actions.astype(np.int64, copy=False), return_counts=True)
    return {str(int(action)): int(count) for action, count in zip(unique, counts, strict=True)}


def model_metrics(model, data: dict[str, np.ndarray], denominator_source: str, classify_fn) -> dict[str, Any]:
    metrics = classify_fn(model, data)
    metrics["denominator_source"] = denominator_source
    metrics["teacher_action_histogram"] = action_histogram(data["actions"])
    return metrics


def summarize_matrix(matrix: np.ndarray) -> dict[str, Any]:
    rows = int(matrix.sum())
    correct = int(np.trace(matrix))
    per_action = {}
    for action in range(ACTION_COUNT):
        count = int(matrix[action].sum())
        if count:
            per_action[str(action)] = {"count": count, "accuracy": float(matrix[action, action] / count)}
    return {
        "rows": rows,
        "correct": correct,
        "incorrect": rows - correct,
        "accuracy": float(correct / rows) if rows else 0.0,
        "macro_action_accuracy": float(np.mean([row["accuracy"] for row in per_action.values()])) if per_action else 0.0,
        "per_action": per_action,
        "confusion_matrix": matrix.astype(int).tolist(),
        "teacher_action_histogram": {str(k): int(v) for k, v in sorted(Counter({i: int(matrix[i].sum()) for i in range(ACTION_COUNT) if matrix[i].sum()}).items())},
    }


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


def action_kind(action: int) -> str:
    return ACTION_KINDS[int(action)]


def action_index(action: int) -> int | None:
    return ACTION_INDICES[int(action)]


def error_type(true_action: int, pred_action: int) -> str:
    if action_kind(true_action) != action_kind(pred_action):
        return "kind"
    if int(true_action) != int(pred_action):
        return "slot"
    return "correct"


def stratum_for_round(round_number: int) -> str:
    if round_number <= 3:
        return "round_1_3"
    if round_number <= 6:
        return "round_4_6"
    return "round_7_plus"


def replay_saved_final(run_root: Path, spec: dict[str, Any], seed: int, arm: str, model) -> dict[str, Any]:
    from hearthstone_ai.lobby_env import LobbyEnv
    from hearthstone_ai.lobby_policies import heuristic_policy

    report_path = run_root / "outputs" / identity(("evaluate", seed, arm)) / "final.json"
    report = read_json(report_path)
    checkpoint = checkpoint_for(run_root, seed, arm)
    if report.get("model_sha256") != digest(checkpoint / "model.zip"):
        raise ValueError("Final report model hash differs from checkpoint")
    if [episode["seed"] for episode in report["episodes"]] != spec["eval_seeds"]:
        raise ValueError("Final report episode seeds differ from 045 spec")
    matrix = np.zeros((ACTION_COUNT, ACTION_COUNT), dtype=np.int64)
    strata = {name: np.zeros((ACTION_COUNT, ACTION_COUNT), dtype=np.int64) for name in ("round_1_3", "round_4_6", "round_7_plus")}
    first_errors: Counter[str] = Counter()
    first_error_pairs: Counter[str] = Counter()
    all_error_types: Counter[str] = Counter()
    all_error_pairs: Counter[str] = Counter()
    saved_action_matches = 0
    states = 0
    ranks: dict[str, float] = {}
    episodes = 0
    for episode in report["episodes"]:
        env = LobbyEnv(
            learner_seat=spec["train"]["learner_seat"],
            max_rounds=spec["train"]["max_rounds"],
            heuristic_opponents=spec["train"]["heuristic_opponents"],
            observation_scale=spec["train"]["observation_scale"],
        )
        try:
            obs, _ = env.reset(seed=int(episode["seed"]))
            done = False
            rewards: list[float] = []
            saw_mismatch = False
            for step_index, saved_action in enumerate(episode["decisions"]):
                action = int(saved_action)
                mask = env.action_masks()
                if done or not bool(mask[action]) or not env.observation_space.contains(obs):
                    raise ValueError(f"Invalid saved state/action seed={seed} arm={arm} eval={episode['seed']} step={step_index}")
                prediction = int(model.predict(obs, deterministic=True, action_masks=mask)[0])
                if prediction != action:
                    raise ValueError("Saved action differs from deterministic final model prediction")
                saved_action_matches += 1
                assert env.game is not None
                teacher = int(heuristic_policy(env.game.view(env.learner_seat)))
                matrix[teacher, prediction] += 1
                strata[stratum_for_round(int(env.game.round))][teacher, prediction] += 1
                if teacher != prediction:
                    kind_pair = f"{action_kind(teacher)}->{action_kind(prediction)}"
                    action_pair = f"{teacher}->{prediction}"
                    all_error_types[error_type(teacher, prediction)] += 1
                    all_error_pairs[kind_pair] += 1
                    if not saw_mismatch:
                        first_errors[error_type(teacher, prediction)] += 1
                        first_error_pairs[action_pair] += 1
                        saw_mismatch = True
                obs, reward, terminated, truncated, info = env.step(action)
                env.game.assert_conservation()
                rewards.append(float(reward))
                done = bool(terminated or truncated)
                states += 1
            rank = validate_half_rank(env.game.players[env.learner_seat].rank)
            expected_reward = float(sum(rewards))
            if (
                not done
                or rank != float(episode["rank"])
                or expected_reward != float(episode["reward"])
                or bool(terminated) != bool(episode["terminated"])
                or bool(truncated) != bool(episode["truncated"])
                or info != episode["info"]
                or json.loads(json.dumps(env.game.trace, sort_keys=True)) != episode["trace"]
                or canonical_sha256(env.game.trace) != canonical_sha256(episode["trace"])
            ):
                raise ValueError("Saved final trace replay mismatch")
            ranks[str(episode["seed"])] = rank
            episodes += 1
        finally:
            env.close()
    summary = summarize_matrix(matrix)
    summary["denominator_source"] = "saved final model visited states"
    return {
        "training_seed": seed,
        "arm": arm,
        "episodes": episodes,
        "states": states,
        "saved_action_model_agreement": {
            "matches": saved_action_matches,
            "denominator": states,
            "accuracy": float(saved_action_matches / states) if states else 0.0,
        },
        "visited_state_teacher_agreement": summary,
        "round_strata": {name: summarize_matrix(value) for name, value in strata.items()},
        "error_types": {key: int(value) for key, value in sorted(all_error_types.items())},
        "error_kind_pairs": {key: int(value) for key, value in sorted(all_error_pairs.items())},
        "first_error_types": {key: int(value) for key, value in sorted(first_errors.items())},
        "first_error_action_pairs": {key: int(value) for key, value in sorted(first_error_pairs.items())},
        "ranks": ranks,
        "mean_rank": float(report["mean_rank"]),
        "mean_reward": float(report["mean_reward"]),
        "top4_fraction_ranked": float(report["top4_fraction_ranked"]),
        "truncated": int(report["truncated"]),
        "report_sha256": digest(report_path),
    }


def pair_gate(replay: dict[str, Any]) -> dict[str, Any]:
    by_seed = {}
    qualifying_counts: Counter[str] = Counter()
    for seed in SEEDS:
        errors = replay[str(seed)]["candidate"].get("error_kind_pairs", {})
        total = int(sum(errors.values()))
        qualifying = {pair: int(count) for pair, count in sorted(errors.items()) if total and count / total >= 0.5}
        for pair in qualifying:
            qualifying_counts[pair] += 1
        by_seed[str(seed)] = {
            "errors": total,
            "qualifying_pairs": {
                pair: {"count": count, "share": float(count / total)}
                for pair, count in qualifying.items()
            },
            "has_dominant_pair": bool(qualifying),
        }
    repeated_pairs = {pair: int(count) for pair, count in sorted(qualifying_counts.items()) if count >= 2}
    return {
        "source": "candidate_final_visited_errors",
        "definition": "same true-kind->pred-kind pair accounts for at least 50% of candidate final visited-state errors in at least 2 seeds; correct rows excluded",
        "by_seed": by_seed,
        "repeated_qualifying_pairs": repeated_pairs,
        "passing_seed_count": max(repeated_pairs.values(), default=0),
        "gate_met": bool(repeated_pairs),
    }


def decision_priorities(source_metrics: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    conflict = []
    added_failed = []
    gaps = []
    for seed in SEEDS:
        row = source_metrics[str(seed)]
        control_base = row["control"]["base_train"]["accuracy"]
        candidate_base = row["candidate"]["base_train"]["accuracy"]
        control_added = row["control"]["collector_added"]["accuracy"]
        candidate_added = row["candidate"]["collector_added"]["accuracy"]
        conflict.append(candidate_added > control_added and candidate_base < control_base)
        added_failed.append(candidate_added <= control_added)
        dev_accuracy = row["candidate"]["dev"]["accuracy"]
        visited_accuracy = replay[str(seed)]["candidate"]["visited_state_teacher_agreement"]["accuracy"]
        gaps.append(dev_accuracy - visited_accuracy >= 0.20)
    if sum(conflict) >= 2:
        branch = "source_accuracy_tradeoff"
        reason = "candidate collector-added micro improves while base-train micro regresses in at least 2 of 3 seeds"
    elif sum(added_failed) >= 2:
        branch = "added_state_learning_failure"
        reason = "candidate collector-added micro does not improve over control in at least 2 of 3 seeds"
    elif sum(gaps) >= 2:
        branch = "remaining_state_distribution_gap"
        reason = "candidate dev micro exceeds candidate visited-state teacher agreement by at least 20pp in at least 2 of 3 seeds"
    else:
        branch = "insufficient_evidence"
        reason = "none of the exclusive 046 priority gates passed"
    return {
        "branch": branch,
        "reason": reason,
        "exclusive_priority_order": [
            "source_accuracy_tradeoff",
            "added_state_learning_failure",
            "remaining_state_distribution_gap",
            "insufficient_evidence",
        ],
        "by_seed": {
            str(seed): {
                "source_accuracy_tradeoff": bool(conflict[index]),
                "added_state_learning_failure": bool(added_failed[index]),
                "dev_minus_visited_gap_at_least_20pp": bool(gaps[index]),
            }
            for index, seed in enumerate(SEEDS)
        },
        "dominant_error_pair_gate": pair_gate(replay),
    }


def diagnose(run_root: Path, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    run_root = Path(run_root).resolve()
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Preserve previous diagnostic: {output}")
    verify_frozen_files_manifest(run_root)
    spec = read_json(run_root / "configs" / SPEC_NAME)
    validate_spec(spec)
    audit = read_json(run_root / "audit.json")
    validate_audit_evidence(run_root, audit)
    receipts = verify_driver_receipts(run_root, spec)
    input_paths = used_input_paths(run_root, spec)
    before_hashes = hash_inputs(input_paths)
    src_path = run_root / "src"
    sys.path.insert(0, str(src_path))

    import torch
    from hearthstone_ai import lobby_env
    from hearthstone_ai.actions import ACTIONS as frozen_actions
    from hearthstone_ai.lobby_env import LobbyEnv
    from hearthstone_ai.lobby_training import load_model
    from audit_lobby_043 import classify as classify_fn

    if not Path(lobby_env.__file__).resolve().is_relative_to(src_path):
        raise ValueError("Frozen 045 source was not imported")
    if [(action.kind, action.index) for action in frozen_actions] != list(zip(ACTION_KINDS, ACTION_INDICES, strict=True)):
        raise ValueError("Frozen action registry differs from 046 action taxonomy")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    base_dir = run_root / "outputs" / BASE_DATASET_ID
    base_train = load_npz(base_dir / "train.npz")
    base_dev = load_npz(base_dir / "dev.npz")
    base_manifest = read_json(base_dir / "manifest.json")
    base_rows = int(base_manifest["split_counts"]["train"])
    dev_rows = int(base_manifest["split_counts"]["dev"])
    validate_npz(base_train, rows=base_rows, name="base train")
    validate_npz(base_dev, rows=dev_rows, name="dev")

    env = LobbyEnv(
        learner_seat=spec["train"]["learner_seat"],
        max_rounds=spec["train"]["max_rounds"],
        heuristic_opponents=spec["train"]["heuristic_opponents"],
        observation_scale=spec["train"]["observation_scale"],
    )
    source_metrics: dict[str, Any] = {}
    replay_metrics: dict[str, Any] = {}
    collection_bounds_for_audit: dict[str, Any] = {}
    source_bounds: dict[str, Any] = {"base_train": {"row_range": [0, base_rows], "rows": base_rows}, "dev": {"rows": dev_rows}}
    try:
        for seed in SEEDS:
            collect_dir = run_root / "outputs" / identity(("collect", seed, "candidate"))
            bounds = validate_collection_bounds(base_dir, collect_dir)
            source_bounds[f"collect-s{seed:03d}-candidate"] = bounds
            collection_bounds_for_audit[str(seed)] = bounds
            augmented = load_npz(collect_dir / "train.npz")
            added = slice_npz(augmented, bounds["base_train_rows"], bounds["augmented_train_rows"], f"collector added s{seed:03d}")
            source_metrics[str(seed)] = {}
            replay_metrics[str(seed)] = {}
            for arm in ARMS:
                checkpoint = checkpoint_for(run_root, seed, arm)
                model = load_model(checkpoint, env)
                source_metrics[str(seed)][arm] = {
                    "base_train": model_metrics(model, base_train, "teacher base train rows [0, base_train_rows)", classify_fn),
                    "collector_added": model_metrics(model, added, "collector-added train rows [base_train_rows, augmented_train_rows)", classify_fn),
                    "dev": model_metrics(model, base_dev, "teacher dev NPZ rows", classify_fn),
                }
                replay_metrics[str(seed)][arm] = replay_saved_final(run_root, spec, seed, arm, model)
    finally:
        env.close()

    validate_audit_evidence(run_root, audit, collection_bounds_for_audit)
    after_hashes = hash_inputs(input_paths)
    verify_frozen_files_manifest(run_root)
    if before_hashes != after_hashes:
        raise RuntimeError("046 used input hashes changed during diagnosis")
    total_classifications = sum(len(arms) * 3 for arms in source_metrics.values())
    total_episodes = sum(replay_metrics[str(seed)][arm]["episodes"] for seed in SEEDS for arm in ARMS)
    if total_classifications != 18:
        raise RuntimeError(f"Expected 6 models x 3 corpora classifications, got {total_classifications}")
    if total_episodes != 120:
        raise RuntimeError(f"Expected 120 saved final episodes, got {total_episodes}")
    result = {
        "status": "passed",
        "diagnostic": "lobby-046-post-collection-source-diagnosis",
        "protocol": DIAGNOSTIC_PROTOCOL,
        "run": str(run_root),
        "cpu_threads": {"torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads()},
        "receipts": receipts,
        "input_hashes_before": before_hashes,
        "input_hashes_after": after_hashes,
        "frozen_inputs_unchanged": True,
        "source_bounds": source_bounds,
        "classification": source_metrics,
        "final_saved_episode_replay": replay_metrics,
        "decision_priorities": decision_priorities(source_metrics, replay_metrics),
        "candidate_keep_changed": False,
        "claim_boundary": [
            "Read-only diagnosis of frozen 045 artifacts after a passed 045 audit.",
            "No new training, sampling, model mutation, or evaluation games are performed.",
            "Base train, collector-added train, dev, and saved final visited-state denominators stay separate.",
            "Repeated dev seeds 20001..20020 are not treated as a final holdout.",
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json_fresh(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.run, args.output), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
