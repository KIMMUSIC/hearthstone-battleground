"""Bounded training entry points for the eight-player lobby adapter."""

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
from stable_baselines3.common.callbacks import BaseCallback

from .artifacts import ROOT, atomic_json, compatibility_signature, environment_report, file_hash
from .lobby_env import LobbyEnv, OPPONENT_PROTOCOL
from .lobby_observation import LobbyObservation, OBSERVATION_SCALES


REWARD_CONTRACT = "lobby-terminal-rank-v1"
TRAIN_SEED_MIN = 0
TRAIN_SEED_MAX = 9999
EVAL_SEEDS = tuple(range(20001, 20021))


def _config(values: dict[str, Any]) -> dict[str, Any]:
    defaults = dict(
        max_steps=8192,
        max_seconds=60.0,
        threads=1,
        seed=7,
        n_steps=32,
        batch_size=32,
        n_epochs=4,
        checkpoint_interval=1024,
        learner_seat=0,
        max_rounds=100,
        heuristic_opponents=7,
        observation_scale="raw",
        gamma=1.0,
    )
    unknown = set(values) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown lobby training settings: {sorted(unknown)}")
    cfg = defaults | values
    for key, value in cfg.items():
        if key == "max_seconds":
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("max_seconds must be finite and positive")
        elif key == "gamma":
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
                or value > 1
            ):
                raise ValueError("gamma must be finite with 0 < gamma <= 1")
        elif key == "observation_scale":
            if value not in OBSERVATION_SCALES:
                raise ValueError(f"observation_scale must be one of {sorted(OBSERVATION_SCALES)}")
        elif (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < (0 if key in {"seed", "learner_seat", "heuristic_opponents"} else 1)
        ):
            raise ValueError(f"{key} must be a valid integer")
    if cfg["max_steps"] > 8192:
        raise ValueError("max_steps must be <= 8192 for the bounded lobby smoke API")
    if cfg["max_seconds"] > 60:
        raise ValueError("max_seconds must be <= 60 for the bounded lobby smoke API")
    if cfg["threads"] != 1:
        raise ValueError("threads must be exactly 1 for reproducible lobby smoke training")
    if cfg["seed"] > TRAIN_SEED_MAX:
        raise ValueError("seed must stay inside the 0..9999 training seed range")
    if not 0 <= cfg["learner_seat"] < 8:
        raise ValueError("learner_seat must be in range 0..7")
    if not 0 <= cfg["heuristic_opponents"] <= 7:
        raise ValueError("heuristic_opponents must be in range 0..7")
    if cfg["n_steps"] < 2 or cfg["batch_size"] < 2 or cfg["n_steps"] % cfg["batch_size"]:
        raise ValueError("n_steps must be divisible by batch_size, both at least 2")
    if cfg["max_steps"] % cfg["n_steps"]:
        raise ValueError("max_steps must be a multiple of n_steps")
    return cfg


def lobby_compatibility_signature(config: dict[str, Any]) -> dict[str, Any]:
    """Compatibility contract for model deserialization and evaluation."""
    cfg = _config(config)
    signature: dict[str, Any] = {
        **compatibility_signature(),
        "observation_version": LobbyObservation.version,
        "lobby_contract_sha256": file_hash(ROOT / "docs/LOBBY_CONTRACT.md"),
        "reward_contract": REWARD_CONTRACT,
        "learner_seat": cfg["learner_seat"],
        "max_rounds": cfg["max_rounds"],
        "opponent_protocol": OPPONENT_PROTOCOL,
        "heuristic_opponents": cfg["heuristic_opponents"],
        "observation_scale": cfg["observation_scale"],
        "training_seed_range": [TRAIN_SEED_MIN, TRAIN_SEED_MAX],
        "eval_seeds": list(EVAL_SEEDS),
    }
    return signature


def _checkpoint_path(path: Path) -> Path:
    return path / "model.zip" if path.is_dir() else path


def _checkpoint_metadata(checkpoint: Path) -> dict[str, Any]:
    model_path = _checkpoint_path(Path(checkpoint))
    return json.loads(model_path.with_name("metadata.json").read_text(encoding="utf-8"))


def _make_env(config: dict[str, Any], *, seed: int | None = None):
    env = LobbyEnv(
        learner_seat=config["learner_seat"],
        max_rounds=config["max_rounds"],
        heuristic_opponents=config["heuristic_opponents"],
        observation_scale=config["observation_scale"],
    )
    if seed is not None:
        env.reset(seed=seed)
    return env


def load_model(checkpoint: Path, env, *, allow_opponent_shift: bool = False) -> MaskablePPO:
    """Load a lobby model only after the saved contract matches this environment."""
    model_path = _checkpoint_path(Path(checkpoint))
    metadata = _checkpoint_metadata(model_path)
    config = metadata.get("config", {})
    expected = lobby_compatibility_signature(config)
    if metadata.get("compatibility") != expected:
        raise ValueError("Lobby checkpoint compatibility mismatch; retrain under this contract")
    for key in ("learner_seat", "max_rounds"):
        if config.get(key) != getattr(env, key, None):
            raise ValueError(f"Lobby checkpoint environment mismatch: {key}")
    if config.get("observation_scale") != getattr(env, "observation_scale", None):
        raise ValueError("Lobby checkpoint environment mismatch: observation_scale")
    if (
        config.get("heuristic_opponents") != getattr(env, "heuristic_opponents", None)
        and not allow_opponent_shift
    ):
        raise ValueError("Lobby checkpoint environment mismatch: heuristic_opponents")
    return MaskablePPO.load(model_path, env=env, device="cpu", force_reset=True)


def train(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Run one bounded CPU lobby training pass into a fresh output directory."""
    cfg = _config(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / ".lobby-training.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    env = None
    started = time.monotonic()
    original_threads = torch.get_num_threads()
    status_started = False

    def log(event: str, **fields: Any) -> None:
        with (output_dir / "train.log").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {"event": event, "elapsed_seconds": time.monotonic() - started, **fields},
                    allow_nan=False,
                    sort_keys=True,
                )
                + "\n"
            )

    try:
        if any(path != lock for path in output_dir.iterdir()):
            raise FileExistsError("Lobby training output must be empty; use a new directory")
        status_started = True
        torch.set_num_threads(cfg["threads"])
        log("start", pid=os.getpid())
        atomic_json(output_dir / "environment.json", environment_report())
        signature = lobby_compatibility_signature(cfg)
        env = _make_env(cfg)
        atomic_json(
            output_dir / "manifest.json",
            {
                "config": cfg,
                "compatibility": signature,
                "opponent_spec": env.opponent_spec,
                "opponents": env.opponent_spec,
                "training_seed_range": [TRAIN_SEED_MIN, TRAIN_SEED_MAX],
                "eval_seeds_reserved": list(EVAL_SEEDS),
            },
        )
        atomic_json(output_dir / "status.json", {"status": "running", "pid": os.getpid()})
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            device="cpu",
            seed=cfg["seed"],
            n_steps=cfg["n_steps"],
            batch_size=cfg["batch_size"],
            n_epochs=cfg["n_epochs"],
            gamma=cfg["gamma"],
            policy_kwargs={"net_arch": [32, 32]},
            verbose=0,
        )
        checkpoints: list[str] = []
        episode_ranks: list[float] = []
        episode_truncations = 0
        episode_records: list[dict[str, Any]] = []

        def save() -> None:
            if any(not torch.isfinite(parameter).all() for parameter in model.policy.parameters()):
                raise FloatingPointError("Non-finite lobby policy parameters")
            name = f"checkpoint-{int(model.num_timesteps):012d}"
            destination = output_dir / name
            if destination.exists():
                name += f"-{len(checkpoints):04d}"
                destination = output_dir / name
            temporary = output_dir / ("." + name + ".tmp")
            temporary.mkdir()
            model.save(temporary / "model.zip")
            atomic_json(
                temporary / "metadata.json",
                {
                    "compatibility": signature,
                    "config": cfg,
                    "opponent_spec": env.opponent_spec,
                    "num_timesteps": int(model.num_timesteps),
                    "additional_steps": int(model.num_timesteps),
                    "reward_contract": REWARD_CONTRACT,
                },
            )
            os.rename(temporary, destination)
            checkpoints.append(name)
            elapsed = time.monotonic() - started
            metrics = {
                "checkpoint": name,
                "elapsed_seconds": elapsed,
                "num_timesteps": int(model.num_timesteps),
                "additional_steps": int(model.num_timesteps),
                "steps_per_second": int(model.num_timesteps) / max(elapsed, 1e-9),
                "episode_count": len(episode_ranks),
                "episode_mean_rank": sum(episode_ranks) / len(episode_ranks)
                if episode_ranks
                else None,
                "episode_truncations": episode_truncations,
            }
            with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics, allow_nan=False, sort_keys=True) + "\n")
            log("checkpoint", checkpoint=name, num_timesteps=int(model.num_timesteps))

        class BudgetCallback(BaseCallback):
            reason = "steps"
            last_saved = 0

            def _on_step(self) -> bool:
                nonlocal episode_truncations
                observations = self.locals.get("new_obs", {})
                arrays = observations.values() if isinstance(observations, dict) else [observations]
                if (
                    any(not np.isfinite(array).all() for array in arrays)
                    or not np.isfinite(self.locals["rewards"]).all()
                ):
                    raise FloatingPointError("Non-finite lobby training observation or reward")
                dones = self.locals.get("dones", [False] * len(self.locals.get("infos", [])))
                for done, info in zip(dones, self.locals.get("infos", []), strict=False):
                    if not done:
                        continue
                    truncated = bool(info.get("truncated") or info.get("TimeLimit.truncated"))
                    rank = info.get("rank")
                    if "rank" in info and info["rank"] is not None:
                        episode_ranks.append(float(rank))
                    if truncated:
                        episode_truncations += 1
                    episode_record = {
                        "event": "episode",
                        "num_timesteps": int(self.num_timesteps),
                        "seed": info.get("seed"),
                        "rank": rank,
                        "truncated": truncated,
                        "termination_reason": info.get("termination_reason"),
                    }
                    episode_records.append(episode_record)
                    log(**episode_record)
                if self.num_timesteps - self.last_saved >= cfg["checkpoint_interval"]:
                    save()
                    self.last_saved = self.num_timesteps
                if time.monotonic() - started >= cfg["max_seconds"]:
                    self.reason = "time"
                    return False
                return True

        callback = BudgetCallback()
        save()
        if time.monotonic() - started < cfg["max_seconds"]:
            model.learn(
                total_timesteps=cfg["max_steps"], callback=callback, reset_num_timesteps=False
            )
        else:
            callback.reason = "time"
        save()
        elapsed = time.monotonic() - started
        summary = {
            "status": "completed",
            "stop_reason": callback.reason,
            "initial_steps": 0,
            "num_timesteps": int(model.num_timesteps),
            "additional_steps": int(model.num_timesteps),
            "elapsed_seconds": elapsed,
            "steps_per_second": int(model.num_timesteps) / max(elapsed, 1e-9),
            "threads": cfg["threads"],
            "initial_checkpoint": checkpoints[0],
            "checkpoint": checkpoints[-1],
            "checkpoints": checkpoints,
            "episode_count": len(episode_ranks),
            "episode_mean_rank": sum(episode_ranks) / len(episode_ranks)
            if episode_ranks
            else None,
            "episode_truncations": episode_truncations,
            "episodes": episode_records,
            "training_metrics": {
                key: float(value)
                for key, value in getattr(getattr(model, "_logger", None), "name_to_value", {}).items()
                if isinstance(value, (int, float, np.number)) and math.isfinite(float(value))
            },
        }
        atomic_json(output_dir / "status.json", summary)
        log("stop", status="completed", reason=callback.reason, num_timesteps=int(model.num_timesteps))
        return summary
    except KeyboardInterrupt:
        if status_started:
            atomic_json(
                output_dir / "status.json",
                {
                    "status": "interrupted",
                    "stop_reason": "keyboard_interrupt",
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
            log("stop", status="interrupted", reason="keyboard_interrupt")
        raise
    except Exception as error:
        if status_started:
            (output_dir / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
            atomic_json(
                output_dir / "status.json",
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
            log("stop", status="failed", reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        if env is not None:
            env.close()
        torch.set_num_threads(original_threads)
        lock.unlink(missing_ok=True)
