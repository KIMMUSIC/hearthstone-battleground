"""Bounded CPU training with immutable, compatibility-checked checkpoints."""

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

from .artifacts import atomic_json as _json, compatibility_signature, environment_report
from .env import BgEnv
from .game import OPPONENT_MODES


def _config(values: dict[str, Any]) -> dict[str, Any]:
    defaults = dict(
        max_steps=64,
        max_seconds=60.0,
        threads=1,
        seed=7,
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        checkpoint_interval=32,
        max_turns=8,
        max_actions=24,
        opponent_mode="fixed-v1",
        gamma=0.99,
    )
    unknown = set(values) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown training settings: {sorted(unknown)}")
    cfg = defaults | values
    for key in defaults:
        value = cfg[key]
        if key == "opponent_mode":
            if value not in OPPONENT_MODES:
                raise ValueError(f"opponent_mode must be one of {sorted(OPPONENT_MODES)}")
        elif key == "max_seconds":
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
        elif (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < (0 if key == "seed" else 1)
        ):
            raise ValueError(f"{key} must be a valid integer")
    if cfg["n_steps"] < 2 or cfg["batch_size"] < 2 or cfg["n_steps"] % cfg["batch_size"]:
        raise ValueError("n_steps must be divisible by batch_size, both at least 2")
    if cfg["max_steps"] % cfg["n_steps"]:
        raise ValueError("max_steps must be a multiple of n_steps")
    return cfg


def _checkpoint_path(path: Path) -> Path:
    return path / "model.zip" if path.is_dir() else path


def _checkpoint_metadata(checkpoint: Path) -> dict[str, Any]:
    model_path = _checkpoint_path(Path(checkpoint))
    return json.loads(model_path.with_name("metadata.json").read_text(encoding="utf-8"))


def load_model(checkpoint: Path, env: BgEnv) -> MaskablePPO:
    """Load trusted local model files only after checking their runtime contract."""
    model_path = _checkpoint_path(Path(checkpoint))
    metadata = _checkpoint_metadata(model_path)
    if metadata.get("compatibility") != compatibility_signature():
        raise ValueError("Checkpoint compatibility mismatch; retrain under the current contract")
    for key in ("max_turns", "max_actions"):
        if metadata.get("config", {}).get(key) != getattr(env, key):
            raise ValueError(f"Checkpoint environment mismatch: {key}")
    return MaskablePPO.load(model_path, env=env, device="cpu", force_reset=True)


def train(config: dict[str, Any], output_dir: Path, resume: Path | None = None) -> dict[str, Any]:
    """Spend max_steps additional transitions, restarting episodes when resuming.

    Restores policy, optimizer and timestep count, not the prior environment or
    complete RNG state. Resume is deliberately not trajectory-identical.
    Time budget is checked cooperatively between steps/updates, not a hard kill.
    """
    cfg = _config(config)  # Reject invalid settings before creating output files.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / ".training.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    env = None
    started = time.monotonic()
    original_threads = torch.get_num_threads()
    status_started = False
    summary: dict[str, Any] = {}

    def log(event: str, **fields: Any) -> None:
        with (output_dir / "train.log").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {"event": event, "elapsed_seconds": time.monotonic() - started, **fields},
                    allow_nan=False,
                )
                + "\n"
            )

    try:
        if any(path != lock for path in output_dir.iterdir()):
            raise FileExistsError("Training output must be empty; use a new directory for resume")
        status_started = True
        log("start", pid=os.getpid(), resume_from=str(resume) if resume else None)
        _json(output_dir / "environment.json", environment_report())
        signature = compatibility_signature()
        torch.set_num_threads(cfg["threads"])
        env = BgEnv(
            opponent_mode=cfg["opponent_mode"],
            max_turns=cfg["max_turns"],
            max_actions=cfg["max_actions"],
        )
        _json(
            output_dir / "manifest.json",
            {
                "config": cfg,
                "compatibility": signature,
                "opponent_distribution": env.opponent_spec,
                "resume_from": str(resume) if resume else None,
                "resume_semantics": "episode restart; optimizer and counters restored; RNG trajectory not restored",
            },
        )
        _json(output_dir / "status.json", {"status": "running", "pid": os.getpid()})
        if resume is None:
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
        else:
            resume_config = _checkpoint_metadata(resume).get("config", {})
            if resume_config.get("opponent_mode", "fixed-v1") != cfg["opponent_mode"]:
                raise ValueError("Resume opponent_mode must match checkpoint")
            if resume_config.get("gamma", 0.99) != cfg["gamma"]:
                raise ValueError("Resume gamma must match checkpoint")
            model = load_model(resume, env)
            if (
                model.n_steps != cfg["n_steps"]
                or model.batch_size != cfg["batch_size"]
                or model.n_epochs != cfg["n_epochs"]
            ):
                raise ValueError("Resume rollout settings must match checkpoint")
            model.set_random_seed(cfg["seed"])
        initial_steps = int(model.num_timesteps)
        checkpoints: list[str] = []

        def save() -> None:
            if any(not torch.isfinite(parameter).all() for parameter in model.policy.parameters()):
                raise FloatingPointError("Non-finite policy parameters")
            name = f"checkpoint-{int(model.num_timesteps):012d}"
            destination = output_dir / name
            if destination.exists():
                # A final trained policy can differ from its pre-update checkpoint.
                name += f"-{len(checkpoints):04d}"
                destination = output_dir / name
            temporary = output_dir / ("." + name + ".tmp")
            temporary.mkdir()
            model.save(temporary / "model.zip")
            _json(
                temporary / "metadata.json",
                {
                    "compatibility": signature,
                    "config": cfg,
                    "opponent_distribution": env.opponent_spec,
                    "num_timesteps": int(model.num_timesteps),
                    "additional_steps": int(model.num_timesteps) - initial_steps,
                },
            )
            os.rename(temporary, destination)
            checkpoints.append(name)
            elapsed = time.monotonic() - started
            rss = None
            try:
                import psutil

                rss = psutil.Process().memory_info().rss
            except (ImportError, OSError):
                pass
            metrics = {
                "checkpoint": name,
                "elapsed_seconds": elapsed,
                "num_timesteps": int(model.num_timesteps),
                "additional_steps": int(model.num_timesteps) - initial_steps,
                "steps_per_second": (int(model.num_timesteps) - initial_steps) / max(elapsed, 1e-9),
                "process_rss_bytes": rss,
            }
            with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            log("checkpoint", checkpoint=name, num_timesteps=int(model.num_timesteps))

        class BudgetCallback(BaseCallback):
            reason = "steps"
            last_saved = initial_steps

            def _on_step(self) -> bool:
                observations = self.locals.get("new_obs", {})
                arrays = observations.values() if isinstance(observations, dict) else [observations]
                if (
                    any(not np.isfinite(array).all() for array in arrays)
                    or not np.isfinite(self.locals["rewards"]).all()
                ):
                    raise FloatingPointError("Non-finite training observation or reward")
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
        if any(not torch.isfinite(parameter).all() for parameter in model.policy.parameters()):
            raise FloatingPointError("Non-finite policy parameters")
        save()
        summary = {
            "status": "completed",
            "stop_reason": callback.reason,
            "initial_steps": initial_steps,
            "num_timesteps": int(model.num_timesteps),
            "additional_steps": int(model.num_timesteps) - initial_steps,
            "elapsed_seconds": time.monotonic() - started,
            "threads": cfg["threads"],
            "initial_checkpoint": checkpoints[0],
            "checkpoint": checkpoints[-1],
            "checkpoints": checkpoints,
        }
        summary["training_metrics"] = {
            key: float(value)
            for key, value in getattr(getattr(model, "_logger", None), "name_to_value", {}).items()
            if isinstance(value, (int, float, np.number)) and math.isfinite(float(value))
        }
        _json(output_dir / "status.json", summary)
        log(
            "stop",
            status="completed",
            reason=callback.reason,
            num_timesteps=int(model.num_timesteps),
        )
        return summary
    except KeyboardInterrupt:
        if status_started:
            _json(
                output_dir / "status.json",
                {
                    "status": "interrupted",
                    "stop_reason": "keyboard_interrupt",
                    "elapsed_seconds": time.monotonic() - started,
                    "checkpoint_policy": "Only previously committed checkpoint directories are resumable",
                },
            )
            log("stop", status="interrupted", reason="keyboard_interrupt")
        raise
    except Exception as error:
        if status_started:
            (output_dir / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
            _json(
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
        lock.unlink()
