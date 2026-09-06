"""Explicit, bounded local commands suitable for a Grok Bot handoff."""

import argparse
import json
from pathlib import Path
import sys

from .artifacts import ROOT, atomic_json, environment_report


def main():
    parser = argparse.ArgumentParser(description="Restricted Battlegrounds development tools")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Report this process's runtime; does not contact Grok Bot")
    smoke = commands.add_parser("smoke", help="Complete a seeded episode without the RL stack")
    smoke.add_argument("--seed", type=int, default=42)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument(
        "--policy", choices=["random", "heuristic", "model"], default="heuristic"
    )
    evaluation.add_argument("--checkpoint", type=Path)
    evaluation.add_argument("--opponent-path", type=Path)
    evaluation.add_argument("--max-turns", type=int, default=8)
    evaluation.add_argument("--max-actions", type=int, default=24)
    evaluation.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--trace", action="store_true")
    evaluation.add_argument("--trace-output", type=Path)
    training = commands.add_parser("train")
    training.add_argument("--config", type=Path, default=ROOT / "configs/smoke_train.json")
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--resume", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            result = environment_report()
        elif args.command == "smoke":
            from .game import Game

            game = Game(max_turns=3)
            game.reset(seed=args.seed)
            battles = []
            while not game.done:
                result = game.step(0)
                battles.append(result.info)
            result = {"seed": args.seed, "battles": battles, "reason": game.termination_reason}
        elif args.command == "evaluate":
            from .evaluation import evaluate

            if args.output.exists():
                raise FileExistsError("Evaluation output exists; choose a new result path")
            if args.trace_output is not None and args.trace_output.exists():
                raise FileExistsError("Trace output exists; choose a new trace path")
            result = evaluate(
                policy=args.policy,
                seeds=args.seeds,
                opponent_path=args.opponent_path,
                checkpoint=args.checkpoint,
                max_turns=args.max_turns,
                max_actions=args.max_actions,
                trace=args.trace,
                trace_output=args.trace_output,
            )
            atomic_json(args.output, result)
        else:
            from .training import train

            config = json.loads(args.config.read_text(encoding="utf-8"))
            result = train(config, args.output, resume=args.resume)
        print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False))
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error
