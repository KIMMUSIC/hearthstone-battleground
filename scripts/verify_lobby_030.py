"""Rule-based lobby verification with preserved per-game traces and exact replays."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hearthstone_ai.lobby import LobbyGame  # noqa: E402
from hearthstone_ai.lobby_observation import LobbyObservation  # noqa: E402
from hearthstone_ai.lobby_policies import heuristic_policy, random_policy  # noqa: E402


def dump(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def play(seed, mode, max_rounds):
    game = LobbyGame(seed=seed, max_rounds=max_rounds)
    policies = [random.Random(f"policy:{seed}:{seat}") for seat in range(8)]
    encoder = LobbyObservation()
    actions = 0
    try:
        while not game.done:
            seat = game.current_seat
            view = game.view(seat)
            observation = encoder.encode(view)
            assert encoder.space.contains(observation)
            assert view.legal_actions == tuple(game.legal_actions(seat))
            action = (random_policy(view, policies[seat]) if mode == "mixed" and seat % 2
                      else heuristic_policy(view))
            assert action in view.legal_actions
            game.step(seat, action)
            actions += 1
            game.assert_conservation()
            assert all(not game.legal_actions(p.seat) for p in game.players if not p.alive)
        for seat in range(8):
            assert encoder.space.contains(encoder.encode(game.view(seat)))
        if game.terminated:
            assert not game.truncated and sum(p.rank for p in game.players) == 36
        else:
            assert game.truncated and all(p.rank is None for p in game.players if p.alive)
    except BaseException as error:
        error.lobby_trace = game.trace
        raise
    return game, actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / "configs/lobby030_verify.json").read_text())
    started = time.monotonic()
    result = {"status": "running", "games": [], "replays": [], "events": {}, "actions": 0}
    counts = Counter()
    try:
        for mode in config["policies"]:
            for seed in config["seeds"]:
                game, actions = play(seed, mode, config["max_rounds"])
                path = output / f"{mode}-{seed:03d}.json"
                dump(path, game.trace)
                counts.update(event["event"] for event in game.trace)
                result["actions"] += actions
                result["games"].append({"seed": seed, "policy": mode, "actions": actions,
                                        "rounds": game.round, "terminated": game.terminated,
                                        "truncated": game.truncated, "ranks": [p.rank for p in game.players],
                                        "trace_file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                if seed in config["replay_seeds"]:
                    replay, replay_actions = play(seed, mode, config["max_rounds"])
                    assert replay.trace == game.trace and replay_actions == actions
                    result["replays"].append({"seed": seed, "policy": mode, "exact_trace": True})
                if sum(p.stat().st_size for p in output.iterdir() if p.is_file()) >= config["output_bytes"]:
                    raise RuntimeError("Output limit")
                if len(result["games"]) % 10 == 0:
                    print(json.dumps({"games": len(result["games"]), "elapsed": time.monotonic() - started}), flush=True)
        result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        dump(output / "failure-trace.json", getattr(error, "lobby_trace", []))
        raise
    finally:
        result["events"] = dict(counts)
        result["elapsed_seconds"] = time.monotonic() - started
        dump(output / "summary.json", result)
    print(json.dumps({"status": result["status"], "games": len(result["games"]),
                      "replays": len(result["replays"]), "actions": result["actions"]}), flush=True)


if __name__ == "__main__":
    main()
