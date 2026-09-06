"""Run one bounded 9/21-22 campaign train segment or development evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
import zipfile


CAMPAIGN_ID = "campaign-021"
SPEC_PATH = Path("experiments/expansion020_campaign.json")
TRAIN_CONFIG = Path("configs/expansion019_train.json")
DEV_EVAL_CONFIG = Path("configs/expansion020_eval.json")
DEV_OPPONENTS = Path("configs/expansion020_dev_opponents.json")
TRAINING_SEEDS = (7, 17, 27)
SEGMENT_STEPS = (8192,) * 12 + (1696,)
MILESTONES = (0, 49152, 100000)
N_STEPS = 32
N_EPOCHS = 4


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


def validate_campaign_spec(root: Path) -> dict[str, Any]:
    spec = read_json(root / SPEC_PATH)
    if spec.get("status") != "designed-not-executed":
        raise ValueError("Unexpected campaign status")
    if tuple(spec.get("training_seeds", ())) != TRAINING_SEEDS:
        raise ValueError("Training seeds must be exactly 7, 17, 27")
    if spec.get("steps_per_seed") != 100000:
        raise ValueError("steps_per_seed must be 100000")
    if tuple(spec.get("segment_additional_steps", ())) != SEGMENT_STEPS:
        raise ValueError("Segment schedule must be 8192*12 + 1696")
    if tuple(spec.get("evaluation_milestones", ())) != MILESTONES:
        raise ValueError("Evaluation milestones must be 0, 49152, 100000")
    if spec.get("expected_updates_per_seed") != 12500:
        raise ValueError("Expected final update count must be 12500")
    limits = spec.get("limits", {})
    required_limits = {
        "cpu_threads": 1,
        "train_seconds_per_segment": 60,
        "supervisor_seconds_per_phase": 180,
        "sampled_rss_bytes": 2 * 1024**3,
        "campaign_train_wall_seconds": 1200,
        "campaign_dev_eval_wall_seconds": 600,
        "campaign_total_wall_seconds": 1800,
        "all_outputs_and_return_zip_bytes": 1024**3,
        "automatic_retry": False,
    }
    for key, expected in required_limits.items():
        if limits.get(key) != expected:
            raise ValueError(f"Campaign limit mismatch: {key}")
    return spec


def load_train_config(root: Path, seed: int, additional_steps: int) -> dict[str, Any]:
    config = read_json(root / TRAIN_CONFIG)
    if config.get("max_steps") != 8192:
        raise ValueError("Base training config must declare max_steps=8192")
    if config.get("max_seconds") != 60 or config.get("threads") != 1:
        raise ValueError("Base training config must keep max_seconds=60 and threads=1")
    if config.get("n_steps") != N_STEPS or config.get("n_epochs") != N_EPOCHS:
        raise ValueError("Base training config must keep n_steps=32 and n_epochs=4")
    if config.get("opponent_mode") != "fixed-v1":
        raise ValueError("Training must use fixed-v1 opponents")
    config["seed"] = seed
    config["max_steps"] = additional_steps
    config["checkpoint_interval"] = additional_steps
    return config


def load_dev_eval_config(root: Path) -> dict[str, Any]:
    config = read_json(root / DEV_EVAL_CONFIG)
    allowed = {"seeds", "opponent_path", "max_turns", "max_actions"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown development eval settings: {sorted(unknown)}")
    if config.get("seeds") != list(range(20001, 20101)):
        raise ValueError("Development eval seeds must be exactly 20001..20100")
    if config.get("opponent_path") != DEV_OPPONENTS.as_posix():
        raise ValueError("Development eval must use configs/expansion020_dev_opponents.json")
    return {
        "seeds": config["seeds"],
        "opponent_path": config["opponent_path"],
        "max_turns": config.get("max_turns", 8),
        "max_actions": config.get("max_actions", 24),
    }


def model_counters(checkpoint: Path) -> dict[str, int]:
    model_zip = checkpoint / "model.zip" if checkpoint.is_dir() else checkpoint
    metadata = read_json(model_zip.with_name("metadata.json"))
    with zipfile.ZipFile(model_zip) as zf:
        data = json.loads(zf.read("data").decode("utf-8"))
    values: dict[str, int] = {}
    for key in ("num_timesteps", "_n_updates"):
        value = data.get(key)
        if type(value) is not int:
            raise RuntimeError(f"Model ZIP missing integer {key}: {model_zip}")
        values[key] = value
    if metadata.get("num_timesteps") != values["num_timesteps"]:
        raise RuntimeError("Checkpoint metadata disagrees with model ZIP num_timesteps")
    return {
        "num_timesteps": values["num_timesteps"],
        "_n_updates": values["_n_updates"],
        "metadata_additional_steps": int(metadata["additional_steps"]),
    }


def expected_updates(timesteps: int) -> int:
    if timesteps % N_STEPS:
        raise ValueError("Expected timesteps must align to n_steps")
    return timesteps // N_STEPS * N_EPOCHS


def checkpoint_for_milestone(output: Path, seed: int, milestone: int) -> Path:
    if milestone == 0:
        summary = read_json(output / f"seed-{seed:03d}" / "segment-01" / "summary.json")
        return Path(summary["initial_checkpoint"])
    cumulative = 0
    for index, steps in enumerate(SEGMENT_STEPS, start=1):
        cumulative += steps
        if cumulative == milestone:
            summary = read_json(output / f"seed-{seed:03d}" / f"segment-{index:02d}" / "summary.json")
            return Path(summary["final_checkpoint"])
    raise ValueError(f"Unsupported milestone: {milestone}")


def update_progress(output: Path, row: dict[str, Any]) -> dict[str, Any]:
    path = output / "progress.json"
    if path.exists():
        progress = read_json(path)
    else:
        progress = {"campaign_id": CAMPAIGN_ID, "events": []}
    progress["events"].append(row)
    progress["last_event"] = row
    write_json(path, progress)
    return progress


def train_model(config: dict[str, Any], output_dir: Path, resume: Path | None) -> dict[str, Any]:
    from hearthstone_ai.training import train

    return train(config, output_dir, resume=resume)


def phase_train(root: Path, output: Path, signature: dict[str, str], seed: int, segment: int) -> dict[str, Any]:
    validate_campaign_spec(root)
    if seed not in TRAINING_SEEDS:
        raise ValueError("Seed must be one of 7, 17, 27")
    if not 1 <= segment <= len(SEGMENT_STEPS):
        raise ValueError("Segment index out of range")
    configure_torch_threads()
    seed_dir = output / f"seed-{seed:03d}"
    phase_dir = seed_dir / f"segment-{segment:02d}"
    require_empty(phase_dir)
    previous_steps = sum(SEGMENT_STEPS[: segment - 1])
    additional_steps = SEGMENT_STEPS[segment - 1]
    expected_total = previous_steps + additional_steps
    resume = None
    if segment > 1:
        previous_summary = read_json(seed_dir / f"segment-{segment - 1:02d}" / "summary.json")
        if previous_summary["status"]["num_timesteps"] != previous_steps:
            raise RuntimeError("Previous segment counter mismatch")
        resume = Path(previous_summary["final_checkpoint"])
    config = load_train_config(root, seed, additional_steps)
    started = time.monotonic()
    status = train_model(config, phase_dir / "run", resume)
    elapsed = time.monotonic() - started
    if status.get("status") != "completed" or status.get("stop_reason") != "steps":
        raise RuntimeError(f"Training segment did not complete by steps: {status}")
    if status.get("additional_steps") != additional_steps or status.get("num_timesteps") != expected_total:
        raise RuntimeError(f"Partial or oversized training segment: {status}")
    initial_checkpoint = phase_dir / "run" / status["initial_checkpoint"]
    final_checkpoint = phase_dir / "run" / status["checkpoint"]
    initial_counters = model_counters(initial_checkpoint)
    final_counters = model_counters(final_checkpoint)
    if initial_counters["num_timesteps"] != previous_steps:
        raise RuntimeError("Initial checkpoint counter mismatch")
    if final_counters["num_timesteps"] != expected_total:
        raise RuntimeError("Final checkpoint counter mismatch")
    if final_counters["_n_updates"] != expected_updates(expected_total):
        raise RuntimeError("Final checkpoint update counter mismatch")
    result = {
        "campaign_id": CAMPAIGN_ID,
        "phase": "train",
        "seed": seed,
        "segment": segment,
        "additional_steps": additional_steps,
        "expected_total_timesteps": expected_total,
        "compatibility": signature,
        "config_path": TRAIN_CONFIG.as_posix(),
        "config": config,
        "resume_checkpoint": str(resume) if resume else None,
        "status": status,
        "initial_checkpoint": str(initial_checkpoint),
        "final_checkpoint": str(final_checkpoint),
        "initial_model_sha256": file_hash(initial_checkpoint / "model.zip"),
        "final_model_sha256": file_hash(final_checkpoint / "model.zip"),
        "initial_counters": initial_counters,
        "final_counters": final_counters,
        "train_seconds": elapsed,
    }
    write_json(phase_dir / "summary.json", result)
    update_progress(
        output,
        {
            "kind": "train",
            "seed": seed,
            "segment": segment,
            "num_timesteps": final_counters["num_timesteps"],
            "_n_updates": final_counters["_n_updates"],
            "model_sha256": result["final_model_sha256"],
            "summary": str((phase_dir / "summary.json").relative_to(output)),
        },
    )
    return result


def evaluate_policy(**kwargs: Any) -> dict[str, Any]:
    from hearthstone_ai.evaluation import evaluate

    return evaluate(**kwargs)


def phase_evaluate(
    root: Path, output: Path, signature: dict[str, str], seed: int, milestone: int
) -> dict[str, Any]:
    validate_campaign_spec(root)
    if seed not in TRAINING_SEEDS:
        raise ValueError("Seed must be one of 7, 17, 27")
    if milestone not in MILESTONES:
        raise ValueError("Milestone must be one of 0, 49152, 100000")
    configure_torch_threads()
    phase_dir = output / f"seed-{seed:03d}" / f"eval-{milestone:06d}"
    require_empty(phase_dir)
    config = load_dev_eval_config(root)
    checkpoint = checkpoint_for_milestone(output, seed, milestone)
    counters = model_counters(checkpoint)
    if counters["num_timesteps"] != milestone:
        raise RuntimeError("Evaluation checkpoint counter mismatch")
    before = file_hash(checkpoint / "model.zip")
    started = time.monotonic()
    report = evaluate_policy(
        policy="model",
        seeds=config["seeds"],
        opponent_path=(root / config["opponent_path"]).resolve(),
        checkpoint=checkpoint,
        max_turns=config["max_turns"],
        max_actions=config["max_actions"],
        trace=True,
    )
    elapsed = time.monotonic() - started
    after = file_hash(checkpoint / "model.zip")
    if after != before:
        raise RuntimeError("Evaluation mutated checkpoint")
    if report.get("episode_count") != 100 or len(report.get("episodes", [])) != 100:
        raise RuntimeError("Development evaluation did not produce exactly 100 games")
    from evaluation_evidence import audit_report

    trace_audit = audit_report(report, config["seeds"])
    report.update(
        campaign_id=CAMPAIGN_ID,
        phase="evaluate",
        training_seed=seed,
        milestone=milestone,
        model_sha256=before,
        model_counters=counters,
        compatibility=signature,
    )
    write_json(phase_dir / "model.json", report)
    result = {
        "campaign_id": CAMPAIGN_ID,
        "phase": "evaluate",
        "seed": seed,
        "milestone": milestone,
        "compatibility": signature,
        "config_path": DEV_EVAL_CONFIG.as_posix(),
        "config": config,
        "checkpoint": str(checkpoint),
        "model_sha256": before,
        "model_counters": counters,
        "trace_audit": trace_audit,
        "report": {
            key: report[key]
            for key in ("mean_reward", "combat_win_rate", "survival_rate", "episode_count")
        },
        "eval_seconds": elapsed,
    }
    write_json(phase_dir / "summary.json", result)
    update_progress(
        output,
        {
            "kind": "evaluate",
            "seed": seed,
            "milestone": milestone,
            "num_timesteps": counters["num_timesteps"],
            "_n_updates": counters["_n_updates"],
            "model_sha256": before,
            "summary": str((phase_dir / "summary.json").relative_to(output)),
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--seed", type=int, required=True)
    train_parser.add_argument("--segment", type=int, required=True)
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--seed", type=int, required=True)
    eval_parser.add_argument("--milestone", type=int, required=True)
    args = parser.parse_args()

    root, signature = load_project(args.checkout, args.expected_source_sha256)
    output = args.output.resolve()
    if args.phase == "train":
        result = phase_train(root, output, signature, args.seed, args.segment)
    else:
        result = phase_evaluate(root, output, signature, args.seed, args.milestone)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
