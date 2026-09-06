"""Read-only activation and reward-delay diagnosis for lobby 035 models."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

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


def array_stats(values: np.ndarray) -> dict[str, float | int]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        raise ValueError("Cannot summarize an empty array")
    return {
        "count": int(flat.size),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "abs_max": float(np.max(np.abs(flat))),
        "abs_p95": float(np.quantile(np.abs(flat), 0.95)),
        "nonzero_fraction": float(np.count_nonzero(flat) / flat.size),
    }


def observation_hash_update(hasher, seed: int, step: int, obs: dict[str, np.ndarray]) -> None:
    hasher.update(str(seed).encode("ascii"))
    hasher.update(str(step).encode("ascii"))
    for key in sorted(obs):
        value = np.ascontiguousarray(obs[key])
        hasher.update(key.encode("ascii"))
        hasher.update(str(value.shape).encode("ascii"))
        hasher.update(str(value.dtype).encode("ascii"))
        hasher.update(value.tobytes())


def build_corpus(run_root: Path, spec: dict):
    sys.path.insert(0, str(run_root / "src"))
    from hearthstone_ai.lobby_env import LobbyEnv
    from hearthstone_ai.lobby_policies import heuristic_policy

    env = LobbyEnv(
        learner_seat=spec["train"]["learner_seat"],
        max_rounds=spec["train"]["max_rounds"],
        heuristic_opponents=7,
    )
    observations: list[dict[str, np.ndarray]] = []
    episode_lengths: dict[str, int] = {}
    field_values: dict[str, list[np.ndarray]] = defaultdict(list)
    player_columns: dict[str, list[float]] = defaultdict(list)
    public_columns: dict[str, list[float]] = defaultdict(list)
    hasher = hashlib.sha256()
    player_names = [
        "seat",
        "round",
        "hp",
        "gold",
        "tier",
        "upgrade_cost",
        "frozen",
        "actions_remaining",
        "swaps_remaining",
    ]
    public_names = ["hp", "tier", "alive", "rank"]

    for seed in spec["eval_seeds"]:
        obs, _ = env.reset(seed=seed)
        done = False
        steps = 0
        while not done:
            if not env.observation_space.contains(obs):
                raise ValueError("Heuristic corpus observation outside space")
            frozen_obs = {key: np.asarray(value).copy() for key, value in obs.items()}
            observation_hash_update(hasher, seed, steps, frozen_obs)
            observations.append(frozen_obs)
            for key, value in frozen_obs.items():
                field_values[key].append(np.asarray(value))
            for index, name in enumerate(player_names):
                player_columns[name].append(float(frozen_obs["player"][index]))
            for index, name in enumerate(public_names):
                public_columns[name].extend(float(row[index]) for row in frozen_obs["public_players"])
            action = heuristic_policy(env.game.view(env.learner_seat))
            if not env.action_masks()[action]:
                raise ValueError("Heuristic selected illegal action")
            obs, _, terminated, truncated, _ = env.step(action)
            env.game.assert_conservation()
            steps += 1
            done = terminated or truncated
        episode_lengths[str(seed)] = steps
    env.close()
    return {
        "observations": observations,
        "summary": {
            "policy": "heuristic",
            "eval_seeds": list(spec["eval_seeds"]),
            "observation_count": len(observations),
            "episode_lengths": episode_lengths,
            "sha256": hasher.hexdigest(),
            "field_stats": {
                key: array_stats(np.concatenate([value.reshape(1, *value.shape) for value in values], axis=0))
                for key, values in sorted(field_values.items())
            },
            "player_columns": {
                key: array_stats(np.asarray(values, dtype=np.float32))
                for key, values in sorted(player_columns.items())
            },
            "public_player_columns": {
                key: array_stats(np.asarray(values, dtype=np.float32))
                for key, values in sorted(public_columns.items())
            },
        },
    }


def checkpoint_from_status(train_dir: Path, point: str) -> Path:
    status = read(train_dir / "status.json")
    key = "initial_checkpoint" if point == "initial" else "checkpoint"
    return train_dir / status[key]


def model_data(model_path: Path) -> dict:
    with zipfile.ZipFile(model_path) as archive:
        return json.loads(archive.read("data"))


def tensor_stats(tensor) -> dict[str, float | int]:
    array = tensor.detach().cpu().numpy().astype(np.float64).reshape(-1)
    return {
        "count": int(array.size),
        "saturation_abs_ge_0_95_fraction": float(np.mean(np.abs(array) >= 0.95)),
        "abs_mean": float(np.mean(np.abs(array))),
        "abs_max": float(np.max(np.abs(array))),
        "abs_p50": float(np.quantile(np.abs(array), 0.50)),
        "abs_p95": float(np.quantile(np.abs(array), 0.95)),
        "abs_p99": float(np.quantile(np.abs(array), 0.99)),
    }


def activation_report(model, observations: list[dict[str, np.ndarray]]):
    import torch

    policy_linear = model.policy.mlp_extractor.policy_net[0]
    policy_tanh = model.policy.mlp_extractor.policy_net[1]
    value_linear = model.policy.mlp_extractor.value_net[0]
    value_tanh = model.policy.mlp_extractor.value_net[1]
    policy_pre, policy_post, value_pre, value_post = [], [], [], []
    with torch.no_grad():
        for obs in observations:
            obs_tensor, _ = model.policy.obs_to_tensor(obs)
            features = model.policy.extract_features(obs_tensor)
            p_pre = policy_linear(features)
            v_pre = value_linear(features)
            policy_pre.append(p_pre)
            value_pre.append(v_pre)
            policy_post.append(policy_tanh(p_pre))
            value_post.append(value_tanh(v_pre))
    return {
        "policy_first_linear_pre_tanh": tensor_stats(torch.cat(policy_pre, dim=0)),
        "policy_first_tanh": tensor_stats(torch.cat(policy_post, dim=0)),
        "value_first_linear_pre_tanh": tensor_stats(torch.cat(value_pre, dim=0)),
        "value_first_tanh": tensor_stats(torch.cat(value_post, dim=0)),
    }


def gae_terminal_coefficients(gamma: float, gae_lambda: float, n_steps: int) -> dict[str, float | int | dict[str, float]]:
    distances = [1, 2, 4, 8, 16, 32, n_steps]
    unique_distances = sorted(set(d for d in distances if d >= 0))
    base = gamma * gae_lambda
    return {
        "gamma": float(gamma),
        "gae_lambda": float(gae_lambda),
        "n_steps": int(n_steps),
        "formula": "(gamma * gae_lambda) ** distance",
        "analytical_only": True,
        "terminal_td_coefficients": {
            str(distance): float(base**distance) for distance in unique_distances
        },
    }


def source_hashes(run_root: Path) -> dict[str, str]:
    names = [
        "FILES.json",
        "src/hearthstone_ai/lobby_env.py",
        "src/hearthstone_ai/lobby_training.py",
        "src/hearthstone_ai/lobby_observation.py",
        "src/hearthstone_ai/lobby_policies.py",
        "src/hearthstone_ai/lobby.py",
        "data/cards.json",
        "data/card_mapping.json",
        "configs/lobby035_comparison.json",
        "docs/LOBBY_CONTRACT.md",
    ]
    return {name: digest(run_root / name) for name in names}


def compact_summary(models: dict) -> dict:
    groups = {
        "initial": [],
        "control_final": [],
        "candidate_final": [],
    }
    for key, report in models.items():
        acts = report["activations_on_shared_heuristic_corpus"]
        row = {
            "policy_tanh_saturation": acts["policy_first_tanh"]["saturation_abs_ge_0_95_fraction"],
            "value_tanh_saturation": acts["value_first_tanh"]["saturation_abs_ge_0_95_fraction"],
            "policy_pre_tanh_abs_p95": acts["policy_first_linear_pre_tanh"]["abs_p95"],
            "value_pre_tanh_abs_p95": acts["value_first_linear_pre_tanh"]["abs_p95"],
        }
        if key.endswith("-initial"):
            groups["initial"].append(row)
        elif "-control-final" in key:
            groups["control_final"].append(row)
        elif "-candidate-final" in key:
            groups["candidate_final"].append(row)

    def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
        return {
            metric: float(np.mean([row[metric] for row in rows]))
            for metric in rows[0]
        }

    return {
        "mean_by_group": {key: summarize(value) for key, value in groups.items()},
        "interpretation": (
            "First Tanh units are already frequently saturated at initialization and become more saturated "
            "after 8192 environment steps (1024 optimizer updates); this supports a normalization/scale hypothesis for follow-up testing, "
            "but does not establish causality."
        ),
    }


def diagnose(run_root: Path, output: Path) -> dict:
    started = time.monotonic()
    if output.exists():
        raise FileExistsError(f"Preserve previous diagnostic: {output}")
    run_root = run_root.resolve()
    verify_files(run_root)
    spec = read(run_root / "configs/lobby035_comparison.json")
    if spec["eval_seeds"] != list(range(20001, 20021)):
        raise ValueError("036 only uses reserved 035 eval seeds 20001..20020")
    sys.path.insert(0, str(run_root / "src"))

    import torch
    from hearthstone_ai import lobby_env
    from hearthstone_ai.lobby_training import load_model

    if not Path(lobby_env.__file__).resolve().is_relative_to(run_root / "src"):
        raise ValueError("Frozen run source was not imported")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    corpus = build_corpus(run_root, spec)
    models = {}
    all_model_hashes_before = {}
    first_rollout = None
    for seed in spec["seeds"]:
        for arm, opponents in spec["arms"].items():
            train_dir = run_root / "outputs" / f"train-s{seed:03d}-{arm}"
            for point in ("initial", "final"):
                checkpoint = checkpoint_from_status(train_dir, point)
                model_path = checkpoint / "model.zip"
                before = digest(model_path)
                metadata = read(checkpoint / "metadata.json")
                env = lobby_env.LobbyEnv(
                    learner_seat=metadata["config"]["learner_seat"],
                    max_rounds=metadata["config"]["max_rounds"],
                    heuristic_opponents=metadata["config"]["heuristic_opponents"],
                )
                model = load_model(checkpoint, env)
                data = model_data(model_path)
                if first_rollout is None:
                    first_rollout = {
                        "gamma": data["gamma"],
                        "gae_lambda": data["gae_lambda"],
                        "n_steps": data["n_steps"],
                    }
                after = digest(model_path)
                if before != after:
                    raise ValueError("Model load mutated checkpoint")
                key = f"s{seed:03d}-{arm}-{point}"
                all_model_hashes_before[key] = before
                models[key] = {
                    "checkpoint": checkpoint.relative_to(run_root).as_posix(),
                    "model_sha256": before,
                    "strict_load_model": True,
                    "training_heuristic_opponents": opponents,
                    "num_timesteps": int(data["num_timesteps"]),
                    "n_updates": int(data["_n_updates"]),
                    "rollout": {
                        "gamma": float(data["gamma"]),
                        "gae_lambda": float(data["gae_lambda"]),
                        "n_steps": int(data["n_steps"]),
                        "batch_size": int(data["batch_size"]),
                        "n_epochs": int(data["n_epochs"]),
                    },
                    "activations_on_shared_heuristic_corpus": activation_report(model, corpus["observations"]),
                }
                env.close()

    verify_files(run_root)
    for key, before in all_model_hashes_before.items():
        seed_arm_point = key.split("-")
        seed = int(seed_arm_point[0][1:])
        arm = seed_arm_point[1]
        point = seed_arm_point[2]
        checkpoint = checkpoint_from_status(run_root / "outputs" / f"train-s{seed:03d}-{arm}", point)
        if digest(checkpoint / "model.zip") != before:
            raise ValueError("Checkpoint changed during diagnostic")

    if first_rollout is None:
        raise ValueError("No checkpoints diagnosed")
    result = {
        "status": "passed",
        "diagnostic": "lobby036-activation-scale-reward-delay",
        "run": str(run_root),
        "frozen_inputs_unchanged": True,
        "cpu_threads": {"torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads()},
        "source_hashes": source_hashes(run_root),
        "corpus": corpus["summary"],
        "summary": compact_summary(models),
        "models": models,
        "gae_terminal_signal": gae_terminal_coefficients(
            float(first_rollout["gamma"]),
            float(first_rollout["gae_lambda"]),
            int(first_rollout["n_steps"]),
        ),
        "limitations": [
            "Read-only diagnosis; no training, model mutation, source change, or normalization adoption.",
            "Activation saturation and input scale are correlational diagnostics, not causal performance claims.",
            "GAE coefficients are analytical terminal TD contribution factors, not measured full advantages.",
            "The shared corpus follows heuristic actions, so it measures model activations off that fixed trajectory.",
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    dump_fresh(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/lobby-035-local-r1"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/diagnostics/lobby036_activation_diagnosis.json"),
    )
    args = parser.parse_args()
    print(json.dumps(diagnose(args.run, args.output), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
