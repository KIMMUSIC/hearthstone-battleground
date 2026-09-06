"""Run 020 development baselines without touching reserved final evaluation inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from expansion_019 import configure_torch_threads, load_project


CAMPAIGN_ID = "expansion-020-baseline"
CONFIG_PATH = Path("configs/expansion020_eval.json")
DEV_OPPONENTS = Path("configs/expansion020_dev_opponents.json")
SEEDS = tuple(range(20001, 20101))
POLICIES = ("heuristic", "random")
REPRO_SEEDS = SEEDS[:3]


def content_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_baseline_config(root: Path) -> dict[str, Any]:
    path = root / CONFIG_PATH
    if path.exists():
        config = read_json(path)
    else:
        config = {
            "seeds": list(SEEDS),
            "opponent_path": DEV_OPPONENTS.as_posix(),
            "max_turns": 8,
            "max_actions": 24,
        }
    allowed = {"seeds", "opponent_path", "max_turns", "max_actions"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown expansion020_eval.json settings: {sorted(unknown)}")
    if tuple(config.get("seeds", ())) != SEEDS:
        raise ValueError("expansion020_eval.json must declare seeds 20001..20100")
    if Path(config.get("opponent_path", "")).as_posix() != DEV_OPPONENTS.as_posix():
        raise ValueError("020 baselines must use configs/expansion020_dev_opponents.json")
    max_turns = config.get("max_turns", 8)
    max_actions = config.get("max_actions", 24)
    for key, value in (("max_turns", max_turns), ("max_actions", max_actions)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    return {
        "seeds": list(SEEDS),
        "opponent_path": DEV_OPPONENTS.as_posix(),
        "max_turns": max_turns,
        "max_actions": max_actions,
    }


def require_empty(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {path}")


def evaluate_policy(
    *,
    policy: str,
    seeds: list[int],
    opponent_path: Path,
    max_turns: int,
    max_actions: int,
) -> dict[str, Any]:
    from hearthstone_ai.evaluation import evaluate

    return evaluate(
        policy=policy,
        seeds=seeds,
        opponent_path=opponent_path,
        max_turns=max_turns,
        max_actions=max_actions,
        trace=True,
    )


def episode_outcomes(report: dict[str, Any]) -> list[dict[str, Any]]:
    keys = (
        "seed",
        "reward",
        "wins",
        "draws",
        "losses",
        "damage_taken",
        "damage_dealt",
        "forced_end_turns",
        "rerolls",
        "swaps",
        "freezes",
        "actions",
        "hp",
        "last_turn",
        "survived",
        "termination_reason",
    )
    return [{key: episode.get(key) for key in keys} for episode in report["episodes"]]


def check_report(policy: str, report: dict[str, Any], expected_count: int) -> None:
    if report.get("policy") != policy:
        raise RuntimeError(f"{policy} baseline returned wrong policy")
    if report.get("episode_count") != expected_count or len(report.get("episodes", [])) != expected_count:
        raise RuntimeError(f"{policy} baseline did not produce {expected_count} episodes")


def phase_baseline(root: Path, output: Path, signature: dict[str, str]) -> dict[str, Any]:
    configure_torch_threads()
    phase_dir = output / "baseline"
    require_empty(phase_dir)
    config = load_baseline_config(root)
    opponent_path = (root / DEV_OPPONENTS).resolve()
    opponent_payload = read_json(opponent_path)
    if not isinstance(opponent_payload.get("opponents"), list) or not opponent_payload["opponents"]:
        raise ValueError("Development opponent config must contain nonempty opponents")
    opponent_hash = content_hash(opponent_payload)
    reports = {}
    reproducibility = {}
    timings = {}
    for policy in POLICIES:
        started = time.monotonic()
        report = evaluate_policy(
            policy=policy,
            seeds=config["seeds"],
            opponent_path=opponent_path,
            max_turns=config["max_turns"],
            max_actions=config["max_actions"],
        )
        elapsed = time.monotonic() - started
        check_report(policy, report, len(SEEDS))
        write_json(phase_dir / f"{policy}.json", report)
        repeat = evaluate_policy(
            policy=policy,
            seeds=list(REPRO_SEEDS),
            opponent_path=opponent_path,
            max_turns=config["max_turns"],
            max_actions=config["max_actions"],
        )
        check_report(policy, repeat, len(REPRO_SEEDS))
        expected = episode_outcomes(report)[: len(REPRO_SEEDS)]
        observed = episode_outcomes(repeat)
        if observed != expected:
            raise RuntimeError(f"{policy} baseline failed fixed-seed reproducibility check")
        write_json(phase_dir / f"{policy}-repro-first3.json", repeat)
        reports[policy] = {
            key: report[key] for key in ("mean_reward", "combat_win_rate", "survival_rate")
        }
        timings[policy] = {
            "seconds": elapsed,
            "episodes_per_second": len(SEEDS) / max(elapsed, 1e-9),
        }
        reproducibility[policy] = {
            "seeds": list(REPRO_SEEDS),
            "episode_outcomes_match": True,
        }
    result = {
        "campaign_id": CAMPAIGN_ID,
        "phase": "baseline",
        "compatibility": signature,
        "config_path": CONFIG_PATH.as_posix() if (root / CONFIG_PATH).exists() else None,
        "config": config,
        "opponent_path": str(opponent_path),
        "opponent_set_sha256": opponent_hash,
        "reports": reports,
        "timings": timings,
        "reproducibility": reproducibility,
        "reserved_final_inputs_used": False,
        "interpretation": "development baseline only; final reserved opponents are not read",
    }
    write_json(phase_dir / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    args = parser.parse_args()

    root, signature = load_project(args.checkout, args.expected_source_sha256)
    result = phase_baseline(root, args.output.resolve(), signature)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
