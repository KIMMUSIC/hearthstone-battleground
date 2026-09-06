"""Run the bounded 019 expansion-environment train/evaluate measurement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


CAMPAIGN_ID = "expansion-019"
FULL_STEPS = 8192
EVALUATION_GAMES = 20
TRAIN_CONFIG = Path("configs/expansion019_train.json")
EVAL_CONFIG = Path("configs/expansion019_eval.json")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def require_empty(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {path}")


def configure_torch_threads() -> None:
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def load_project(checkout: Path, expected_source_sha256: str) -> tuple[Path, dict[str, str]]:
    root = checkout.resolve()
    sys.path.insert(0, str(root / "src"))
    from hearthstone_ai.artifacts import compatibility_signature

    signature = compatibility_signature()
    if signature["source_sha256"] != expected_source_sha256:
        raise ValueError(
            "Source hash mismatch: "
            f"expected {expected_source_sha256}, got {signature['source_sha256']}"
        )
    return root, signature


def load_training_config(root: Path) -> dict[str, Any]:
    config = read_json(root / TRAIN_CONFIG)
    if config.get("max_steps") != FULL_STEPS:
        raise ValueError("expansion019_train.json must declare max_steps=8192")
    max_seconds = config.get("max_seconds")
    if (
        isinstance(max_seconds, bool)
        or not isinstance(max_seconds, (int, float))
        or max_seconds <= 0
        or max_seconds > 60
    ):
        raise ValueError("expansion019_train.json must declare 0 < max_seconds <= 60")
    if config.get("threads") != 1:
        raise ValueError("expansion019_train.json must declare threads=1")
    if config.get("checkpoint_interval", FULL_STEPS) > FULL_STEPS:
        raise ValueError("checkpoint_interval must not exceed the 8192-step budget")
    return config


def load_eval_config(root: Path) -> dict[str, Any]:
    config = read_json(root / EVAL_CONFIG)
    allowed = {"seeds", "opponent_path", "max_turns", "max_actions"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown expansion019_eval.json settings: {sorted(unknown)}")
    seeds = config.get("seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) != EVALUATION_GAMES
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("expansion019_eval.json must declare 20 unique nonnegative integer seeds")
    return {
        "seeds": seeds,
        "opponent_path": config.get("opponent_path", "configs/eval_opponents.json"),
        "max_turns": config.get("max_turns", 8),
        "max_actions": config.get("max_actions", 24),
    }


def train_model(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    from hearthstone_ai.training import train

    return train(config, output_dir)


def assert_checkpoint_loads(checkpoint: Path, *, max_turns: int, max_actions: int) -> None:
    from hearthstone_ai.env import BgEnv
    from hearthstone_ai.training import load_model

    env = BgEnv(max_turns=max_turns, max_actions=max_actions)
    try:
        load_model(checkpoint, env)
    finally:
        env.close()


def evaluate_policy(
    *,
    policy: str,
    seeds: list[int],
    opponent_path: Path,
    checkpoint: Path | None,
    max_turns: int,
    max_actions: int,
) -> dict[str, Any]:
    from hearthstone_ai.evaluation import evaluate

    return evaluate(
        policy=policy,
        seeds=seeds,
        opponent_path=opponent_path,
        checkpoint=checkpoint,
        max_turns=max_turns,
        max_actions=max_actions,
        trace=True,
    )


def phase_train(root: Path, output: Path, signature: dict[str, str]) -> dict[str, Any]:
    configure_torch_threads()
    phase_dir = output / "train"
    require_empty(phase_dir)
    config = load_training_config(root)
    started = time.monotonic()
    status = train_model(config, phase_dir / "run")
    train_seconds = time.monotonic() - started
    if status["status"] != "completed":
        raise RuntimeError(f"Training did not complete: {status}")
    achieved_steps = int(status["additional_steps"])
    if achieved_steps > FULL_STEPS:
        raise RuntimeError(f"Training exceeded 8192-step budget: {achieved_steps}")
    initial_checkpoint = phase_dir / "run" / status["initial_checkpoint"]
    final_checkpoint = phase_dir / "run" / status["checkpoint"]
    assert_checkpoint_loads(
        initial_checkpoint, max_turns=config.get("max_turns", 8), max_actions=config.get("max_actions", 24)
    )
    assert_checkpoint_loads(
        final_checkpoint, max_turns=config.get("max_turns", 8), max_actions=config.get("max_actions", 24)
    )
    result = {
        "campaign_id": CAMPAIGN_ID,
        "phase": "train",
        "compatibility": signature,
        "config_path": TRAIN_CONFIG.as_posix(),
        "config": config,
        "status": status,
        "initial_checkpoint": str(initial_checkpoint),
        "final_checkpoint": str(final_checkpoint),
        "initial_model_sha256": file_hash(initial_checkpoint / "model.zip"),
        "final_model_sha256": file_hash(final_checkpoint / "model.zip"),
        "achieved_steps": achieved_steps,
        "train_seconds": train_seconds,
        "training_steps_per_second": achieved_steps / max(train_seconds, 1e-9),
    }
    write_json(phase_dir / "summary.json", result)
    return result


def _model_case(stage: str, train_summary: dict[str, Any]) -> dict[str, str]:
    checkpoint = Path(train_summary[f"{stage}_checkpoint"])
    return {
        "stage": stage,
        "checkpoint": str(checkpoint),
        "model_sha256": train_summary[f"{stage}_model_sha256"],
    }


def _ensure_episode_count(label: str, report: dict[str, Any]) -> None:
    if report.get("episode_count") != EVALUATION_GAMES or len(report.get("episodes", [])) != EVALUATION_GAMES:
        raise RuntimeError(f"{label} did not produce exactly 20 evaluation games")


def phase_evaluate(root: Path, output: Path, signature: dict[str, str]) -> dict[str, Any]:
    configure_torch_threads()
    phase_dir = output / "evaluate"
    require_empty(phase_dir)
    train_summary = read_json(output / "train" / "summary.json")
    config = load_eval_config(root)
    opponent_path = (root / config["opponent_path"]).resolve()
    reports: dict[str, dict[str, Any]] = {}
    timings = {}
    model_cases = [_model_case("initial", train_summary), _model_case("final", train_summary)]
    for case in model_cases:
        checkpoint = Path(case["checkpoint"])
        before = file_hash(checkpoint / "model.zip")
        if before != case["model_sha256"]:
            raise RuntimeError(f"Checkpoint changed before evaluation: {checkpoint}")
        started = time.monotonic()
        report = evaluate_policy(
            policy="model",
            seeds=config["seeds"],
            opponent_path=opponent_path,
            checkpoint=checkpoint,
            max_turns=config["max_turns"],
            max_actions=config["max_actions"],
        )
        elapsed = time.monotonic() - started
        after = file_hash(checkpoint / "model.zip")
        if after != before:
            raise RuntimeError(f"Evaluation mutated checkpoint: {checkpoint}")
        label = f"model-{case['stage']}"
        _ensure_episode_count(label, report)
        report.update(
            campaign_id=CAMPAIGN_ID,
            phase="evaluate",
            stage=case["stage"],
            model_sha256=before,
            compatibility=signature,
        )
        write_json(phase_dir / f"{label}.json", report)
        reports[label] = {
            key: report[key] for key in ("mean_reward", "combat_win_rate", "survival_rate")
        }
        timings[label] = {
            "seconds": elapsed,
            "episodes_per_second": EVALUATION_GAMES / max(elapsed, 1e-9),
        }
    for policy in ("heuristic", "random"):
        started = time.monotonic()
        report = evaluate_policy(
            policy=policy,
            seeds=config["seeds"],
            opponent_path=opponent_path,
            checkpoint=None,
            max_turns=config["max_turns"],
            max_actions=config["max_actions"],
        )
        elapsed = time.monotonic() - started
        _ensure_episode_count(policy, report)
        report.update(campaign_id=CAMPAIGN_ID, phase="evaluate", compatibility=signature)
        write_json(phase_dir / f"{policy}.json", report)
        reports[policy] = {
            key: report[key] for key in ("mean_reward", "combat_win_rate", "survival_rate")
        }
        timings[policy] = {
            "seconds": elapsed,
            "episodes_per_second": EVALUATION_GAMES / max(elapsed, 1e-9),
        }
    result = {
        "campaign_id": CAMPAIGN_ID,
        "phase": "evaluate",
        "compatibility": signature,
        "config_path": EVAL_CONFIG.as_posix(),
        "config": config,
        "opponent_path": str(opponent_path),
        "model_cases": model_cases,
        "reports": reports,
        "timings": timings,
    }
    write_json(phase_dir / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--phase", choices=("train", "evaluate"), required=True)
    args = parser.parse_args()

    root, signature = load_project(args.checkout, args.expected_source_sha256)
    output = args.output.resolve()
    if args.phase == "train":
        result = phase_train(root, output, signature)
    else:
        result = phase_evaluate(root, output, signature)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
