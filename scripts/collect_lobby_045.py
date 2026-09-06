"""Collect model-visited lobby states with heuristic teacher labels for 045."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hearthstone_ai.artifacts import atomic_json, file_hash  # noqa: E402
from hearthstone_ai.lobby_env import LobbyEnv  # noqa: E402
from hearthstone_ai.lobby_imitation import _load_manifest  # noqa: E402
from hearthstone_ai.lobby_policies import heuristic_policy  # noqa: E402
from hearthstone_ai.lobby_training import load_model  # noqa: E402


def _fresh_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError("045 collection output must be empty; use a new directory")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _validate_seeds(base_manifest: dict[str, Any], seeds: list[int]) -> None:
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int or not 0 <= seed <= 9999 for seed in seeds)
    ):
        raise ValueError("seeds must be a nonempty list of integers in 0..9999")
    if seeds != base_manifest.get("config", {}).get("train_seeds"):
        raise ValueError("collector seeds must equal base manifest train_seeds")


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(arrays):
            raise RuntimeError("Saved augmented NPZ keys differ from source arrays")


def _collection_rows(model, seed: int, max_rounds: int, row_start: int):
    env = LobbyEnv(
        learner_seat=0,
        max_rounds=max_rounds,
        heuristic_opponents=7,
        observation_scale="fixed-v1",
    )
    rows: list[dict[str, np.ndarray]] = []
    teacher_actions: list[int] = []
    executed_actions: list[int] = []
    reward_total = 0.0
    try:
        obs, _ = env.reset(seed=seed)
        while True:
            assert env.game is not None
            view = env.game.view(env.learner_seat)
            teacher = int(heuristic_policy(view))
            if teacher not in view.legal_actions or not env.observation_space.contains(obs):
                raise ValueError("Collector produced an illegal or out-of-space row")
            prediction, _ = model.predict(
                obs,
                deterministic=True,
                action_masks=obs["action_mask"].astype(bool),
            )
            executed = int(np.asarray(prediction).item())
            if executed not in view.legal_actions:
                raise ValueError("Parent model predicted an illegal collector action")
            rows.append({key: value.copy() for key, value in obs.items()})
            teacher_actions.append(teacher)
            executed_actions.append(executed)
            obs, reward, terminated, truncated, _ = env.step(executed)
            reward_total += float(reward)
            env.game.assert_conservation()
            if terminated or truncated:
                rank = env.game.players[env.learner_seat].rank
                if truncated or rank is None:
                    raise RuntimeError(f"Collector seed {seed} did not finish with a natural rank")
                rank = float(rank)
                expected_reward = float((4.5 - rank) / 3.5)
                if not np.isclose(reward_total, expected_reward):
                    raise RuntimeError("Collector reward total does not match terminal rank formula")
                return (
                    rows,
                    teacher_actions,
                    [seed] * len(teacher_actions),
                    {
                        "seed": seed,
                        "executed_actions": executed_actions,
                        "teacher_actions": teacher_actions,
                        "row_start": row_start,
                        "row_end": row_start + len(teacher_actions),
                        "rank": rank,
                        "reward": reward_total,
                        "trace_sha256": _canonical_sha256(env.game.trace),
                    },
                )
    finally:
        env.close()


def _concat_rows(base: dict[str, np.ndarray], rows: list[dict[str, np.ndarray]], actions, seeds):
    if not rows:
        raise RuntimeError("Collector produced no rows")
    additions = {
        f"obs__{key}": np.stack([row[key] for row in rows]).astype(base[f"obs__{key}"].dtype, copy=False)
        for key in rows[0]
    }
    additions["actions"] = np.asarray(actions, dtype=base["actions"].dtype)
    additions["seeds"] = np.asarray(seeds, dtype=base["seeds"].dtype)
    return {key: np.concatenate([base[key], additions[key]], axis=0) for key in base}


def collect(
    base_dir: Path,
    checkpoint: Path,
    output: Path,
    *,
    model_seed: int,
    seeds: list[int],
    max_rounds: int = 100,
) -> dict[str, Any]:
    """Append current-model visited training states with heuristic labels to a base dataset."""
    base_dir = Path(base_dir)
    checkpoint = Path(checkpoint)
    output = Path(output)
    if type(model_seed) is not int or not 0 <= model_seed <= 9999:
        raise ValueError("model_seed must be an integer in 0..9999")
    if type(max_rounds) is not int or max_rounds < 1:
        raise ValueError("max_rounds must be a positive integer")
    _fresh_dir(output)

    before_hashes = {
        "base_train": file_hash(base_dir / "train.npz"),
        "base_dev": file_hash(base_dir / "dev.npz"),
        "checkpoint_model": file_hash(checkpoint / "model.zip"),
        "checkpoint_metadata": file_hash(checkpoint / "metadata.json"),
    }
    base_manifest = _load_manifest(base_dir)
    _validate_seeds(base_manifest, seeds)
    if max_rounds != base_manifest["core_config"]["max_rounds"]:
        raise ValueError("max_rounds must match base manifest")

    load_env = LobbyEnv(
        learner_seat=base_manifest["core_config"]["learner_seat"],
        max_rounds=max_rounds,
        heuristic_opponents=base_manifest["core_config"]["heuristic_opponents"],
        observation_scale=base_manifest["core_config"]["observation_scale"],
    )
    model = load_model(checkpoint, load_env)
    try:
        base_train = _load_npz(base_dir / "train.npz")
        base_train_rows = int(base_train["actions"].shape[0])
        all_rows: list[dict[str, np.ndarray]] = []
        all_teacher_actions: list[int] = []
        all_seeds: list[int] = []
        episodes = []
        row_start = base_train_rows
        for seed in seeds:
            rows, teacher_actions, seed_rows, episode = _collection_rows(
                model,
                seed,
                max_rounds,
                row_start,
            )
            all_rows.extend(rows)
            all_teacher_actions.extend(teacher_actions)
            all_seeds.extend(seed_rows)
            episodes.append(episode)
            row_start = episode["row_end"]
        augmented_train = _concat_rows(base_train, all_rows, all_teacher_actions, all_seeds)
        _save_npz(output / "train.npz", augmented_train)
        shutil.copyfile(base_dir / "dev.npz", output / "dev.npz")

        after_hashes = {
            "base_train": file_hash(base_dir / "train.npz"),
            "base_dev": file_hash(base_dir / "dev.npz"),
            "checkpoint_model": file_hash(checkpoint / "model.zip"),
            "checkpoint_metadata": file_hash(checkpoint / "metadata.json"),
        }
        if after_hashes != before_hashes:
            raise RuntimeError("Collector input hashes changed during collection")
        train_hash = file_hash(output / "train.npz")
        dev_hash = file_hash(output / "dev.npz")
        added_rows = int(len(all_teacher_actions))
        provenance = {
            "kind": "lobby-045-visited-training",
            "model_seed": model_seed,
            "checkpoint": str(checkpoint),
            "checkpoint_model_sha256": before_hashes["checkpoint_model"],
            "checkpoint_metadata_sha256": before_hashes["checkpoint_metadata"],
            "base_train_rows": base_train_rows,
            "added_rows": added_rows,
            "trajectory": {
                "prefix": "teacher-train",
                "append": "current-parent-model-visited-states-with-heuristic-labels",
                "row_start": base_train_rows,
                "row_end": base_train_rows + added_rows,
            },
            "input_hashes_before": before_hashes,
            "input_hashes_after": after_hashes,
        }
        manifest = {
            **base_manifest,
            "split_counts": {
                **base_manifest["split_counts"],
                "train": base_train_rows + added_rows,
            },
            "split_hashes": {
                **base_manifest["split_hashes"],
                "train": train_hash,
                "dev": dev_hash,
            },
            "splits": {
                **base_manifest.get("splits", {}),
                "train": {
                    **base_manifest.get("splits", {}).get("train", {}),
                    "rows": base_train_rows + added_rows,
                    "sha256": train_hash,
                },
                "dev": {
                    **base_manifest.get("splits", {}).get("dev", {}),
                    "sha256": dev_hash,
                },
            },
            "provenance": provenance,
        }
        collection = {
            "model_seed": model_seed,
            "checkpoint": str(checkpoint),
            "checkpoint_model_sha256": before_hashes["checkpoint_model"],
            "checkpoint_metadata_sha256": before_hashes["checkpoint_metadata"],
            "base_train_rows": base_train_rows,
            "added_rows": added_rows,
            "episodes": episodes,
            "input_hashes_before": before_hashes,
            "input_hashes_after": after_hashes,
            "output_hashes": {
                "train": train_hash,
                "dev": dev_hash,
            },
        }
        atomic_json(output / "manifest.json", manifest)
        atomic_json(output / "collection.json", collection)
        return {
            "status": "completed",
            "base_train_rows": base_train_rows,
            "added_rows": added_rows,
            "split_counts": manifest["split_counts"],
            "split_hashes": manifest["split_hashes"],
            "provenance": provenance,
            "collection": collection,
        }
    finally:
        model.env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--max-rounds", type=int, default=100)
    args = parser.parse_args()
    print(
        json.dumps(
            collect(
                args.base_dir,
                args.checkpoint,
                args.output,
                model_seed=args.model_seed,
                seeds=args.seeds,
                max_rounds=args.max_rounds,
            ),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
