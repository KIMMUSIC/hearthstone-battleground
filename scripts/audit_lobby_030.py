"""Audit saved lobby results by replaying each recorded legal action on frozen code."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

from run_diversity_009 import verify_files


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    started = time.monotonic()
    verify_files(root)
    sys.path.insert(0, str(root / "src"))
    from hearthstone_ai.lobby import LobbyGame
    from hearthstone_ai.lobby_observation import LobbyObservation

    require(Path(sys.modules[LobbyGame.__module__].__file__).resolve().is_relative_to(root), "Wrong source loaded")
    config = read(root / "configs/lobby030_verify.json")
    summary = read(root / "outputs/verification/summary.json")
    control = read(root / "outputs/control/result.json")
    require(control["status"] == "completed", "Supervisor incomplete")
    require(control["elapsed_seconds"] <= config["supervisor_seconds"], "Time limit")
    command, = control["commands"]
    require(command["returncode"] == 0 and command["remaining_pids"] == [], "Process exit")
    require(command["peak_sampled_rss"] < config["sampled_rss_bytes"], "RSS limit")
    require(summary["status"] == "passed", "Verification failed")
    expected = {(seed, mode) for seed in config["seeds"] for mode in config["policies"]}
    require(len(summary["games"]) == len(expected) == 100, "Game count")
    require({(g["seed"], g["policy"]) for g in summary["games"]} == expected, "Game conditions")
    replay_expected = {(seed, mode) for seed in config["replay_seeds"] for mode in config["policies"]}
    require(len(summary["replays"]) == 6 and
            {(r["seed"], r["policy"]) for r in summary["replays"] if r["exact_trace"]} == replay_expected,
            "Policy replay evidence")
    count = Counter()
    actions = 0
    for row in summary["games"]:
        path = root / "outputs/verification" / row["trace_file"]
        require(hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], "Trace hash")
        trace = read(path)
        game = LobbyGame(seed=row["seed"], max_rounds=config["max_rounds"])
        encoder = LobbyObservation()
        for event in trace:
            if event["event"] != "action":
                continue
            require(event["seat"] == game.current_seat and event["round"] == game.round, "Recruit order")
            view = game.view(game.current_seat)
            require(event["action"] in view.legal_actions, "Illegal action")
            encoder.encode(view)
            require(all(not hasattr(m, "entity_id") for collection in (view.shop, view.hand, view.board, view.pending_discover)
                        for m in collection if m is not None), "Private global ID exposed")
            game.step(event["seat"], event["action"])
            game.assert_conservation()
            require(all(0 <= available <= game.pool.initial[card]
                        for card, available in game.pool.available.items()), "Negative or excessive pool")
        require(game.done and game.terminated and not game.truncated, "Natural completion missing")
        require([p.rank for p in game.players] == row["ranks"] and sum(row["ranks"]) == 36, "Rank mismatch")
        require(game.round == row["rounds"] and row["terminated"] and not row["truncated"], "Terminal metadata")
        require(json.loads(json.dumps(game.trace)) == trace, "Full action replay differs")
        trace_actions = sum(e["event"] == "action" for e in trace)
        require(trace_actions == row["actions"], "Action count")
        count.update(e["event"] for e in trace)
        actions += trace_actions
    require(dict(count) == summary["events"] and actions == summary["actions"], "Aggregate mismatch")
    verify_files(root)
    report = {"status": "passed", "main_games": 100, "policy_replays": 6,
              "independent_action_trace_replays": 100, "actions": actions,
              "all_main_games_natural_termination": True, "events": dict(count),
              "peak_sampled_rss_bytes": command["peak_sampled_rss"],
              "supervised_wall_seconds": control["elapsed_seconds"],
              "audit_seconds": time.monotonic() - started,
              "frozen_inputs_unchanged": True, "private_global_ids_absent": True}
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
