"""Independent action-stream replay and artifact audit for probe034."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

from lobby_032 import digest, dump, read
from run_diversity_009 import size, verify_files


def audit(root):
    if (root / "audit.json").exists():
        raise FileExistsError("Preserve previous audit")
    verify_files(root)
    sys.path[:0] = [str(root / "scripts"), str(root / "src")]
    from lobby_034 import aggregate, decision, identity
    from hearthstone_ai.lobby import LobbyGame
    from hearthstone_ai.lobby_observation import LobbyObservation
    spec = read(root / "configs/lobby034_probe.json")
    if spec["heuristic_opponents"] != [7, 3, 0] or spec["learner_policies"] != ["heuristic", "random", "end", "freeze"]:
        raise ValueError("Precommitted condition mismatch")
    control = read(root / "outputs/control/result.json")
    command = control["commands"][0]
    if (control["status"] != "completed" or len(control["commands"]) != 1 or command["returncode"] != 0
            or command["remaining_pids"] or command["peak_sampled_rss"] >= spec["limits"]["rss_bytes"]
            or control["elapsed_seconds"] > spec["limits"]["phase_seconds"]):
        raise ValueError("Supervisor violation")
    archive = root / "return.zip"
    receipt = read(root / "return.manifest.json")
    if digest(archive) != receipt["sha256"] or archive.stat().st_size != receipt["bytes"]:
        raise ValueError("Return hash mismatch")
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("CRC failure")
        for name, sha in json.loads(z.read("FILES.json")).items():
            if digest(root / name) != sha or hashlib.sha256(z.read(name)).hexdigest() != sha:
                raise ValueError("Return member mismatch")
    if size(root / "outputs") + archive.stat().st_size >= spec["limits"]["disk_bytes"]:
        raise ValueError("Disk budget")
    encoder = LobbyObservation()
    conditions, outcomes = {}, {}
    action_count = 0
    for h in spec["heuristic_opponents"]:
        for policy in spec["learner_policies"]:
            records = []
            for seed in spec["seeds"]:
                record = read(root / f"outputs/probe/{identity(h, policy, seed)}.json")
                game = LobbyGame(seed=seed, max_rounds=spec["max_rounds"])
                learner_actions = []
                for event in record["trace"]:
                    if event["event"] != "action":
                        continue
                    if game.done or game.players[0].rank is not None:
                        raise ValueError("Action after learner termination")
                    seat, action = event["seat"], event["action"]
                    if seat != game.current_seat or event["round"] != game.round or action not in game.legal_actions(seat):
                        raise ValueError("Illegal trace")
                    encoder.encode(game.view(seat))
                    game.step(seat, action)
                    game.assert_conservation()
                    if seat == 0:
                        learner_actions.append(action)
                    action_count += 1
                rank = game.players[0].rank
                encoder.encode(game.view(0))
                if (rank is None and not game.truncated) or rank != record["rank"]:
                    raise ValueError("Missing or wrong final rank")
                if (record["seed"] != seed or record["heuristic_opponents"] != h or record["learner_policy"] != policy
                        or record["reward"] != (0 if rank is None else (4.5 - rank) / 3.5)
                        or record["truncated"] != (game.truncated and rank is None)
                        or learner_actions != record["learner_actions"]
                        or json.loads(json.dumps(game.trace)) != record["trace"]):
                    raise ValueError("Trace/result mismatch")
                records.append(record)
            key = f"h{h}-{policy}"
            conditions[key] = aggregate(records)
            outcomes[key] = [(r["rank"], r["reward"]) for r in records]
    summary = read(root / "outputs/probe/summary.json")
    equal = {str(h): sum(a == b for a, b in zip(outcomes[f"h{h}-end"], outcomes[f"h{h}-freeze"], strict=True))
             for h in spec["heuristic_opponents"]}
    if (summary["conditions"] != conditions or summary["decision"] != decision(conditions)
            or summary["episodes"] != 240 or summary["policy_replays"] != 12
            or summary["end_freeze_equal_outcomes"] != equal):
        raise ValueError("Aggregate/decision mismatch")
    verify_files(root)
    result = {"status": "passed", "episodes_replayed": 240, "actions_replayed": action_count,
              "decision": decision(conditions), "peak_sampled_rss_bytes": command["peak_sampled_rss"],
              "supervisor_seconds": control["elapsed_seconds"], "frozen_inputs_unchanged": True,
              "outputs_and_return_bytes": size(root / "outputs") + archive.stat().st_size}
    dump(root / "audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run.resolve())))
