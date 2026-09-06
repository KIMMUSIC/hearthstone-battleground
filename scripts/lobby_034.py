"""Rule-only opponent-difficulty probe with preserved traces and supervised execution."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import zipfile

from lobby_032 import digest, dump, package, read
from run_diversity_009 import require_runtime, size, verify_files

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hearthstone_ai.lobby import LobbyGame  # noqa: E402
from hearthstone_ai.lobby_observation import LobbyObservation  # noqa: E402
from hearthstone_ai.lobby_policies import heuristic_policy, random_policy  # noqa: E402


def identity(h, policy, seed):
    return f"h{h}-{policy}-s{seed}"


def play(seed, heuristic_count, policy, max_rounds):
    game = LobbyGame(seed=seed, max_rounds=max_rounds)
    rngs = [random.Random(f"lobby-probe:{seed}:policy:{seat}") for seat in range(8)]
    encoder = LobbyObservation()
    learner_actions = []
    while not game.done and game.players[0].rank is None:
        seat = game.current_seat
        view = game.view(seat)
        encoder.encode(view)
        if seat == 0:
            if policy == "heuristic":
                action = heuristic_policy(view)
            elif policy == "random":
                action = random_policy(view, rngs[seat])
            elif policy == "freeze":
                action = 3 if 3 in view.legal_actions else 0
            elif policy == "end":
                action = 0
            else:
                raise ValueError("Unknown policy")
            learner_actions.append(action)
        else:
            action = heuristic_policy(view) if seat <= heuristic_count else random_policy(view, rngs[seat])
        game.step(seat, action)
        game.assert_conservation()
    rank = game.players[0].rank
    encoder.encode(game.view(0))
    return {"seed": seed, "heuristic_opponents": heuristic_count, "learner_policy": policy,
            "rank": rank, "reward": 0 if rank is None else (4.5 - rank) / 3.5,
            "truncated": bool(game.truncated and rank is None), "round": game.round,
            "learner_actions": learner_actions, "trace": game.trace}


def aggregate(episodes):
    ranks = [e["rank"] for e in episodes if e["rank"] is not None]
    mean = sum(ranks) / len(ranks) if ranks else None
    actions = Counter(a for e in episodes for a in e["learner_actions"])
    return {"episodes": len(episodes), "ranked": len(ranks), "mean_rank": mean,
            "rank_variance": sum((r - mean)**2 for r in ranks) / len(ranks) if ranks else None,
            "distinct_ranks": sorted(set(ranks)), "last_fraction": sum(r == 8 for r in ranks) / len(ranks) if ranks else None,
            "top4_fraction": sum(r <= 4 for r in ranks) / len(ranks) if ranks else None,
            "truncated": sum(e["truncated"] for e in episodes), "learner_actions": sum(actions.values()),
            "freezes": actions[3], "buys": sum(actions[a] for a in range(4, 11)),
            "plays": sum(actions[a] for a in range(18, 28)),
            "forced_ends": sum(e.get("event") == "recruit_end" and e.get("seat") == 0
                               and e.get("forced", False) for r in episodes for e in r["trace"])}


def decision(results):
    hard = results["h7-random"]
    mixed = results["h3-random"]
    competent = results["h3-heuristic"]
    supported = (hard["last_fraction"] is not None and hard["last_fraction"] >= .9
                 and len(mixed["distinct_ranks"]) >= 3 and mixed["last_fraction"] <= .7
                 and mixed["rank_variance"] > 0 and mixed["truncated"] == 0
                 and competent["truncated"] == 0
                 and mixed["mean_rank"] - competent["mean_rank"] >= 1.0)
    return {"hypothesis_supported": supported, "selected_heuristic_opponents": 3 if supported else None,
            "performance_improvement_claim": False}


def probe(root):
    verify_files(root)
    spec = read(root / "configs/lobby034_probe.json")
    output = root / "outputs/probe"
    output.mkdir(exist_ok=False)
    results = {}
    comparable = {}
    for h in spec["heuristic_opponents"]:
        for policy in spec["learner_policies"]:
            episodes = []
            for seed in spec["seeds"]:
                episode = play(seed, h, policy, spec["max_rounds"])
                dump(output / f"{identity(h, policy, seed)}.json", episode)
                episodes.append(episode)
            repeat = play(spec["seeds"][0], h, policy, spec["max_rounds"])
            if repeat != episodes[0]:
                raise AssertionError("Policy replay mismatch")
            results[f"h{h}-{policy}"] = aggregate(episodes)
            comparable[f"h{h}-{policy}"] = [(e["rank"], e["reward"]) for e in episodes]
    result = {"status": "completed", "conditions": results, "decision": decision(results),
              "episodes": len(spec["seeds"]) * len(results), "policy_replays": len(results),
              "end_freeze_equal_outcomes": {str(h): sum(a == b for a, b in zip(comparable[f"h{h}-end"],
                                               comparable[f"h{h}-freeze"], strict=True)) for h in spec["heuristic_opponents"]}}
    dump(output / "summary.json", result)
    verify_files(root)
    return result


def run(root):
    spec = read(root / "configs/lobby034_probe.json")
    python = str(require_runtime(spec))
    verify_files(root)
    output = root / "outputs"
    if output.exists() or (root / "return.zip").exists():
        raise FileExistsError("Already attempted")
    output.mkdir()
    control_spec = output / "supervision.json"
    dump(control_spec, {"commands": [[python, "-B", str(root / "scripts/lobby_034.py"), "probe"]],
                        "timeout_seconds": spec["limits"]["phase_seconds"], "rss_limit_bytes": spec["limits"]["rss_bytes"]})
    active = None
    started = time.monotonic()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with (output / "driver.log").open("wb") as log:
            active = subprocess.Popen([python, "-B", str(root / "scripts/supervise.py"),
                                       "--spec", str(control_spec), "--output", str(output / "control")],
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while active.poll() is None:
                if time.monotonic() - started > spec["limits"]["phase_seconds"] + 2 or size(output) >= spec["limits"]["disk_bytes"]:
                    raise RuntimeError("Outer probe budget exceeded")
                time.sleep(.2)
        if active.returncode != 0 or read(output / "control/result.json")["status"] != "completed":
            raise RuntimeError("Probe failed")
        verify_files(root)
        paths = [p for p in output.rglob("*") if p.is_file()]
        archive = root / "return.zip"
        with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as z:
            for p in paths:
                z.write(p, p.relative_to(root).as_posix())
            z.writestr("FILES.json", json.dumps({p.relative_to(root).as_posix(): digest(p) for p in paths}))
        if size(output) + archive.stat().st_size >= spec["limits"]["disk_bytes"]:
            raise RuntimeError("Return exceeds output budget")
        result = {"sha256": digest(archive), "bytes": archive.stat().st_size, "elapsed_seconds": time.monotonic() - started}
        dump(root / "return.manifest.json", result)
        return result
    finally:
        if active is not None and active.poll() is None:
            active.send_signal(signal.SIGTERM)
            active.wait(timeout=5)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["package", "probe", "run"])
    parser.add_argument("--name", default="lobby-034-local-r1")
    args = parser.parse_args()
    result = (package(ROOT, args.name, [ROOT / "experiments/LOBBY_034_DIAGNOSIS.md"]) if args.mode == "package"
              else probe(ROOT) if args.mode == "probe" else run(ROOT))
    print(json.dumps(result, allow_nan=False))
