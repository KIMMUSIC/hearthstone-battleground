"""Behavior-cloning diagnostics for fixed-scale lobby teacher trajectories."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from .artifacts import atomic_json, environment_report, file_hash
from .lobby_env import LobbyEnv
from .lobby_policies import heuristic_policy
from .lobby_training import REWARD_CONTRACT, _config, lobby_compatibility_signature


TRAINING_METHOD = "behavior-cloning-diagnostic-v1"
SAMPLING_PROTOCOL_LEGACY = "legacy-integers-v1"
SAMPLING_PROTOCOL_CHOICE = "choice-probabilities-v1"
SAMPLING_MODES = {"uniform-choice", "balanced-choice"}
DEFAULT_TRAIN_SEEDS = list(range(100))
DEFAULT_DEV_SEEDS = list(range(20001, 20021))


def _dataset_config(values: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "train_seeds": DEFAULT_TRAIN_SEEDS,
        "dev_seeds": DEFAULT_DEV_SEEDS,
        "max_rounds": 100,
    }
    unknown = set(values) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown imitation dataset settings: {sorted(unknown)}")
    cfg = defaults | values
    for key in ("train_seeds", "dev_seeds"):
        seeds = cfg[key]
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(type(seed) is not int or seed < 0 for seed in seeds)
        ):
            raise ValueError(f"{key} must be a nonempty list of nonnegative integers")
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"{key} must not contain duplicates")
    if set(cfg["train_seeds"]) & set(cfg["dev_seeds"]):
        raise ValueError("train_seeds and dev_seeds must be disjoint")
    if type(cfg["max_rounds"]) is not int or cfg["max_rounds"] < 1:
        raise ValueError("max_rounds must be a positive integer")
    return cfg


def _imitation_config(values: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "updates": 1024,
        "batch_size": 64,
        "learning_rate": 0.0003,
        "max_seconds": 60.0,
    }
    unknown = set(values) - (set(defaults) | {"sampling"})
    if unknown:
        raise ValueError(f"Unknown imitation training settings: {sorted(unknown)}")
    cfg = defaults | values
    if "sampling" not in values:
        cfg["sampling"] = "legacy-integers"
    elif cfg["sampling"] not in SAMPLING_MODES:
        raise ValueError(f"sampling must be one of {sorted(SAMPLING_MODES)}")
    if type(cfg["updates"]) is not int or cfg["updates"] < 1:
        raise ValueError("updates must be a positive integer")
    if type(cfg["batch_size"]) is not int or cfg["batch_size"] < 1:
        raise ValueError("batch_size must be a positive integer")
    for key in ("learning_rate", "max_seconds"):
        value = cfg[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{key} must be finite and positive")
    return cfg


def _core_config(values: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = _config({"heuristic_opponents": 7, "observation_scale": "fixed-v1", **(values or {})})
    if cfg["heuristic_opponents"] != 7:
        raise ValueError("heuristic_opponents must be 7 for imitation diagnostics")
    if cfg["observation_scale"] != "fixed-v1":
        raise ValueError("observation_scale must be fixed-v1 for imitation diagnostics")
    if cfg["threads"] != 1:
        raise ValueError("threads must be exactly 1 for imitation diagnostics")
    return cfg


def _fresh_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError("Imitation output must be empty; use a new directory")


def _trajectory(seed: int, max_rounds: int) -> tuple[list[dict[str, np.ndarray]], list[int], list[int], float]:
    env = LobbyEnv(
        learner_seat=0,
        max_rounds=max_rounds,
        heuristic_opponents=7,
        observation_scale="fixed-v1",
    )
    observations: list[dict[str, np.ndarray]] = []
    actions: list[int] = []
    seeds: list[int] = []
    try:
        obs, _ = env.reset(seed=seed)
        while True:
            assert env.game is not None
            view = env.game.view(env.learner_seat)
            action = heuristic_policy(view)
            if action not in view.legal_actions or not env.observation_space.contains(obs):
                raise ValueError("Teacher produced an illegal or out-of-space observation row")
            observations.append({key: value.copy() for key, value in obs.items()})
            actions.append(action)
            seeds.append(seed)
            obs, _, terminated, truncated, _ = env.step(action)
            env.game.assert_conservation()
            if terminated or truncated:
                if truncated or env.game.players[env.learner_seat].rank is None:
                    raise RuntimeError(f"Seed {seed} did not finish with a natural learner rank")
                return observations, actions, seeds, float(env.game.players[env.learner_seat].rank)
    finally:
        env.close()


def _write_split(path: Path, seed_values: list[int], max_rounds: int) -> dict[str, Any]:
    rows: list[dict[str, np.ndarray]] = []
    actions: list[int] = []
    seeds: list[int] = []
    ranks: dict[str, float] = {}
    for seed in seed_values:
        episode_rows, episode_actions, episode_seeds, rank = _trajectory(seed, max_rounds)
        rows.extend(episode_rows)
        actions.extend(episode_actions)
        seeds.extend(episode_seeds)
        ranks[str(seed)] = rank
    if not rows:
        raise RuntimeError("No imitation rows were generated")
    arrays = {
        f"obs__{key}": np.stack([row[key] for row in rows]).astype(rows[0][key].dtype, copy=False)
        for key in rows[0]
    }
    arrays["actions"] = np.asarray(actions, dtype=np.int64)
    arrays["seeds"] = np.asarray(seeds, dtype=np.int64)
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != set(arrays):
            raise RuntimeError("Saved dataset keys do not match generated arrays")
    return {
        "rows": len(actions),
        "episodes": len(seed_values),
        "seeds": list(seed_values),
        "natural_ranks": ranks,
        "sha256": file_hash(path),
    }


def build_dataset(config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Create fresh train/dev teacher-label NPZ files for the BC diagnostic."""
    cfg = _dataset_config(config)
    output = Path(output)
    _fresh_dir(output)
    train = _write_split(output / "train.npz", cfg["train_seeds"], cfg["max_rounds"])
    dev = _write_split(output / "dev.npz", cfg["dev_seeds"], cfg["max_rounds"])
    core = _core_config({"max_rounds": cfg["max_rounds"]})
    manifest = {
        "config": cfg,
        "core_config": {
            "heuristic_opponents": core["heuristic_opponents"],
            "observation_scale": core["observation_scale"],
            "learner_seat": core["learner_seat"],
            "max_rounds": core["max_rounds"],
        },
        "compatibility": lobby_compatibility_signature(core),
        "split_counts": {
            "train": train["rows"],
            "dev": dev["rows"],
        },
        "split_hashes": {
            "train": train["sha256"],
            "dev": dev["sha256"],
        },
        "splits": {
            "train": train,
            "dev": dev,
        },
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest


def _load_manifest(dataset_dir: Path) -> dict[str, Any]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg = _dataset_config(manifest.get("config", {}))
    if set(cfg["train_seeds"]) & set(cfg["dev_seeds"]):
        raise ValueError("Dataset train/dev seeds overlap")
    core = _core_config({"max_rounds": cfg["max_rounds"]})
    expected_core = {
        "heuristic_opponents": core["heuristic_opponents"],
        "observation_scale": core["observation_scale"],
        "learner_seat": core["learner_seat"],
        "max_rounds": core["max_rounds"],
    }
    if manifest.get("core_config") != expected_core:
        raise ValueError("Dataset core_config mismatch")
    if manifest.get("compatibility") != lobby_compatibility_signature(core):
        raise ValueError("Dataset compatibility mismatch")
    for split in ("train", "dev"):
        file = dataset_dir / f"{split}.npz"
        if manifest.get("split_hashes", {}).get(split) != file_hash(file):
            raise ValueError(f"Dataset {split} hash mismatch")
        data = _load_npz(file)
        _validate_npz_split(split, data, cfg[f"{split}_seeds"], manifest["split_counts"][split])
    return manifest


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def sampling_probabilities(actions, mode: str) -> np.ndarray:
    """Return per-row sampling probabilities for explicit 043 choice modes."""
    if mode not in SAMPLING_MODES:
        raise ValueError(f"sampling mode must be one of {sorted(SAMPLING_MODES)}")
    labels = np.asarray(actions)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("actions must be a nonempty 1D array")
    labels = labels.astype(np.int64, copy=False)
    if mode == "uniform-choice":
        return np.full(labels.shape[0], 1.0 / labels.shape[0], dtype=np.float64)
    unique, counts = np.unique(labels, return_counts=True)
    per_label = dict(zip(unique.tolist(), counts.tolist(), strict=True))
    count_for_row = np.asarray([per_label[int(action)] for action in labels], dtype=np.float64)
    probabilities = 0.5 / labels.shape[0] + 0.5 / (len(unique) * count_for_row)
    return probabilities.astype(np.float64, copy=False)


def _validate_npz_split(
    split: str,
    data: dict[str, np.ndarray],
    expected_seeds: list[int] | None = None,
    expected_rows: int | None = None,
) -> None:
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
        raise ValueError(f"Dataset {split} keys mismatch")
    actions = data["actions"]
    seeds = data["seeds"]
    if actions.dtype.kind not in {"i", "u"} or seeds.dtype.kind not in {"i", "u"}:
        raise ValueError(f"Dataset {split} actions and seeds must be integer arrays")
    if actions.ndim != 1 or seeds.ndim != 1 or len(actions) != len(seeds):
        raise ValueError(f"Dataset {split} actions and seeds shape mismatch")
    rows = len(actions)
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"Dataset {split} row count mismatch")
    slot_width = None
    for key, expected_shape in {
        "obs__player": (rows, 9),
        "obs__public_players": (rows, 8, 4),
        "obs__action_mask": (rows, 37),
    }.items():
        if data[key].shape != expected_shape:
            raise ValueError(f"Dataset {split} {key} shape mismatch")
        if not np.isfinite(data[key]).all():
            raise ValueError(f"Dataset {split} {key} contains nonfinite values")
    for key, capacity in {
        "obs__shop": 7,
        "obs__board": 7,
        "obs__hand": 10,
        "obs__discover": 3,
    }.items():
        value = data[key]
        if value.ndim != 3 or value.shape[:2] != (rows, capacity):
            raise ValueError(f"Dataset {split} {key} shape mismatch")
        slot_width = value.shape[2] if slot_width is None else slot_width
        if value.shape[2] != slot_width:
            raise ValueError(f"Dataset {split} card slot width mismatch")
        if not np.isfinite(value).all():
            raise ValueError(f"Dataset {split} {key} contains nonfinite values")
    mask = data["obs__action_mask"]
    if not np.isin(mask, [0, 1]).all():
        raise ValueError(f"Dataset {split} action masks must be binary")
    if (actions < 0).any() or (actions >= mask.shape[1]).any():
        raise ValueError(f"Dataset {split} actions outside action space")
    if not mask[np.arange(rows), actions.astype(np.int64)].all():
        raise ValueError(f"Dataset {split} teacher actions must be legal under masks")
    if expected_seeds is not None:
        expected = np.asarray(expected_seeds, dtype=np.int64)
        observed = np.unique(seeds.astype(np.int64))
        if not np.array_equal(observed, expected):
            raise ValueError(f"Dataset {split} seeds mismatch")


def _obs_batch(data: dict[str, np.ndarray], indices: np.ndarray | slice, device: torch.device):
    return {
        key.removeprefix("obs__"): torch.as_tensor(data[key][indices], device=device)
        for key in data
        if key.startswith("obs__")
    }


def _actor_parameters(policy) -> list[torch.nn.Parameter]:
    return [
        parameter
        for name, parameter in policy.named_parameters()
        if not name.startswith("mlp_extractor.value_net") and not name.startswith("value_net")
    ]


def _critic_state(policy) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in policy.named_parameters()
        if name.startswith("mlp_extractor.value_net") or name.startswith("value_net")
    }


def _critic_unchanged(policy, before: dict[str, torch.Tensor]) -> bool:
    return all(torch.equal(parameter.detach().cpu(), before[name]) for name, parameter in policy.named_parameters() if name in before)


def _finite_parameters(policy) -> bool:
    return all(torch.isfinite(parameter).all().item() for parameter in policy.parameters())


def _sampling_protocol(mode: str) -> str:
    return SAMPLING_PROTOCOL_LEGACY if mode == "legacy-integers" else SAMPLING_PROTOCOL_CHOICE


def _sampling_histogram(actions: np.ndarray, indices: np.ndarray) -> dict[str, int]:
    if indices.size == 0:
        return {}
    sampled = actions[indices.reshape(-1)].astype(np.int64, copy=False)
    unique, counts = np.unique(sampled, return_counts=True)
    return {str(int(action)): int(count) for action, count in zip(unique, counts, strict=True)}


def _write_sampling_record(
    output: Path,
    indices: np.ndarray,
    probabilities: np.ndarray,
) -> str:
    np.savez_compressed(
        output / "sampling.npz",
        indices=indices.astype(np.int64, copy=False),
        probabilities=probabilities.astype(np.float64, copy=False),
    )
    with np.load(output / "sampling.npz", allow_pickle=False) as data:
        if data["indices"].dtype != np.int64 or data["probabilities"].dtype != np.float64:
            raise RuntimeError("Saved sampling record has unexpected dtypes")
    return file_hash(output / "sampling.npz")


def _save_checkpoint(
    output: Path,
    name: str,
    model: MaskablePPO,
    compatibility: dict[str, Any],
    config: dict[str, Any],
    supervised: dict[str, Any],
) -> None:
    destination = output / name
    if destination.exists():
        raise FileExistsError(f"Checkpoint already exists: {name}")
    temporary = output / ("." + name + ".tmp")
    temporary.mkdir()
    try:
        model.save(temporary / "model.zip")
        atomic_json(
            temporary / "metadata.json",
            {
                "compatibility": compatibility,
                "config": config,
                "num_timesteps": int(model.num_timesteps),
                "additional_steps": 0,
                "reward_contract": REWARD_CONTRACT,
                "training_method": TRAINING_METHOD,
                "supervised": supervised,
            },
        )
        os.rename(temporary, destination)
    except Exception:
        for path in sorted(temporary.glob("*")):
            path.unlink(missing_ok=True)
        temporary.rmdir()
        raise


def evaluate_labels(model: MaskablePPO, dataset_file: Path) -> dict[str, Any]:
    """Evaluate masked teacher-action classification metrics for a saved NPZ split."""
    data = _load_npz(Path(dataset_file))
    _validate_npz_split(Path(dataset_file).stem, data)
    actions = data["actions"].astype(np.int64, copy=False)
    device = model.policy.device
    model.policy.set_training_mode(False)
    with torch.no_grad():
        obs = _obs_batch(data, slice(None), device)
        masks = data["obs__action_mask"].astype(bool, copy=False)
        dist = model.policy.get_distribution(obs, action_masks=masks)
        action_tensor = torch.as_tensor(actions, device=device)
        log_prob = dist.log_prob(action_tensor)
        if not torch.isfinite(log_prob).all():
            raise FloatingPointError("Non-finite label log probabilities")
        predicted = dist.mode().detach().cpu().numpy().astype(np.int64)
    correct = predicted == actions
    per_action: dict[str, dict[str, float | int]] = {}
    for action in sorted(set(actions.tolist())):
        action_mask = actions == action
        count = int(action_mask.sum())
        per_action[str(action)] = {
            "count": count,
            "accuracy": float(correct[action_mask].mean()) if count else None,
        }
    confusion = np.zeros((37, 37), dtype=np.int64)
    for actual, guess in zip(actions, predicted, strict=True):
        confusion[int(actual), int(guess)] += 1
    macro_accuracies = [row["accuracy"] for row in per_action.values()]
    return {
        "rows": int(len(actions)),
        "accuracy": float(correct.mean()) if len(actions) else 0.0,
        "macro_action_accuracy": float(np.mean(macro_accuracies)) if macro_accuracies else 0.0,
        "nll": float((-log_prob).mean().detach().cpu().item()) if len(actions) else 0.0,
        "per_action": per_action,
        "confusion_matrix": confusion.tolist(),
    }


def train_imitation(
    config: dict[str, Any],
    imitation_config: dict[str, Any],
    dataset_dir: Path,
    output: Path,
) -> dict[str, Any]:
    """Train the policy head by masked teacher-action behavior cloning."""
    cfg = _core_config(config)
    imitate = _imitation_config(imitation_config)
    dataset_dir = Path(dataset_dir)
    output = Path(output)
    _fresh_dir(output)
    started = time.monotonic()
    status_started = False
    original_threads = torch.get_num_threads()
    env = None
    try:
        status_started = True
        torch.set_num_threads(cfg["threads"])
        manifest = _load_manifest(dataset_dir)
        core = manifest["core_config"]
        if cfg["learner_seat"] != core["learner_seat"] or cfg["max_rounds"] != core["max_rounds"]:
            raise ValueError("Training config must match dataset learner_seat and max_rounds")
        atomic_json(output / "environment.json", environment_report())
        atomic_json(output / "dataset_manifest.json", manifest)
        train_data = _load_npz(dataset_dir / "train.npz")
        _validate_npz_split(
            "train",
            train_data,
            manifest["config"]["train_seeds"],
            manifest["split_counts"]["train"],
        )
        train_rows = int(train_data["actions"].shape[0])
        if train_rows < 1:
            raise ValueError("Training split must contain at least one row")
        if imitate["sampling"] == "legacy-integers":
            probabilities = np.full(train_rows, 1.0 / train_rows, dtype=np.float64)
        else:
            probabilities = sampling_probabilities(train_data["actions"], imitate["sampling"])
        if not np.isfinite(probabilities).all() or not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("Sampling probabilities must be finite and sum to 1")
        env = LobbyEnv(
            learner_seat=cfg["learner_seat"],
            max_rounds=cfg["max_rounds"],
            heuristic_opponents=cfg["heuristic_opponents"],
            observation_scale=cfg["observation_scale"],
        )
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            device="cpu",
            seed=cfg["seed"],
            n_steps=cfg["n_steps"],
            batch_size=cfg["batch_size"],
            n_epochs=cfg["n_epochs"],
            gamma=cfg["gamma"],
            learning_rate=imitate["learning_rate"],
            policy_kwargs={"net_arch": [32, 32]},
            verbose=0,
        )
        compatibility = lobby_compatibility_signature(cfg)
        supervised = {
            "training_method": TRAINING_METHOD,
            "dataset_train_rows": train_rows,
            "dataset_dev_rows": int(manifest["split_counts"]["dev"]),
            "optimizer_steps": 0,
            "rows_seen": 0,
            "finite_parameters": _finite_parameters(model.policy),
            "sampling": {
                "mode": imitate["sampling"],
                "protocol": _sampling_protocol(imitate["sampling"]),
                "hash": None,
                "histogram": {},
            },
        }
        _save_checkpoint(output, "checkpoint-initial", model, compatibility, cfg, supervised)

        for name, parameter in model.policy.named_parameters():
            if name.startswith("mlp_extractor.value_net") or name.startswith("value_net"):
                parameter.requires_grad_(False)
        actor_parameters = _actor_parameters(model.policy)
        critic_before = _critic_state(model.policy)
        rng = np.random.default_rng(cfg["seed"])
        optimizer_steps = 0
        rows_seen = 0
        sampled_indices: list[np.ndarray] = []
        stop_reason = "updates"
        model.policy.set_training_mode(True)
        for _ in range(imitate["updates"]):
            if time.monotonic() - started >= imitate["max_seconds"]:
                stop_reason = "time"
                break
            if imitate["sampling"] == "legacy-integers":
                indices = rng.integers(0, train_rows, size=imitate["batch_size"])
            else:
                indices = rng.choice(
                    train_rows,
                    size=imitate["batch_size"],
                    replace=True,
                    p=probabilities,
                )
            indices = indices.astype(np.int64, copy=False)
            sampled_indices.append(indices.copy())
            obs = _obs_batch(train_data, indices, model.policy.device)
            masks = train_data["obs__action_mask"][indices].astype(bool, copy=False)
            actions = torch.as_tensor(train_data["actions"][indices], device=model.policy.device)
            distribution = model.policy.get_distribution(obs, action_masks=masks)
            loss = -distribution.log_prob(actions).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite imitation loss")
            model.policy.optimizer.zero_grad()
            loss.backward()
            for parameter in actor_parameters:
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError("Non-finite imitation gradient")
            torch.nn.utils.clip_grad_norm_(actor_parameters, 0.5)
            model.policy.optimizer.step()
            optimizer_steps += 1
            rows_seen += imitate["batch_size"]
        model.num_timesteps = 0
        model._n_updates = 0
        critic_unchanged = _critic_unchanged(model.policy, critic_before)
        finite_parameters = _finite_parameters(model.policy)
        if not critic_unchanged:
            raise FloatingPointError("Critic parameters changed during imitation training")
        if not finite_parameters:
            raise FloatingPointError("Non-finite final imitation parameters")
        index_array = (
            np.stack(sampled_indices).astype(np.int64, copy=False)
            if sampled_indices
            else np.zeros((0, imitate["batch_size"]), dtype=np.int64)
        )
        sampling_hash = _write_sampling_record(output, index_array, probabilities)
        sampling = {
            "mode": imitate["sampling"],
            "protocol": _sampling_protocol(imitate["sampling"]),
            "hash": sampling_hash,
            "histogram": _sampling_histogram(train_data["actions"], index_array),
        }
        supervised = {
            **supervised,
            "optimizer_steps": optimizer_steps,
            "rows_seen": rows_seen,
            "critic_unchanged": critic_unchanged,
            "finite_parameters": finite_parameters,
            "sampling": sampling,
        }
        _save_checkpoint(output, "checkpoint-final", model, compatibility, cfg, supervised)
        elapsed = time.monotonic() - started
        status = {
            "status": "completed" if stop_reason == "updates" else "stopped",
            "stop_reason": stop_reason,
            "initial_checkpoint": "checkpoint-initial",
            "checkpoint": "checkpoint-final",
            "checkpoints": ["checkpoint-initial", "checkpoint-final"],
            "optimizer_steps": optimizer_steps,
            "rows_seen": rows_seen,
            "num_timesteps": int(model.num_timesteps),
            "sb3_n_updates": int(getattr(model, "_n_updates", 0)),
            "elapsed_seconds": elapsed,
            "training_method": TRAINING_METHOD,
            "critic_unchanged": supervised["critic_unchanged"],
            "finite_parameters": supervised["finite_parameters"],
            "sampling": sampling,
            "dataset_train_rows": train_rows,
            "dataset_dev_rows": int(manifest["split_counts"]["dev"]),
        }
        atomic_json(output / "status.json", status)
        return status
    except Exception as error:
        if status_started:
            (output / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
            atomic_json(
                output / "status.json",
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
        raise
    finally:
        if env is not None:
            env.close()
        torch.set_num_threads(original_threads)
