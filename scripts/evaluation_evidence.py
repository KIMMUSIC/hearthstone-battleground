"""Validate episode summaries against full traces without running policies."""

import math


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_report(report, seeds):
    episodes = report["episodes"]
    traces = report["trace"]["episodes"]
    require(len(seeds) > 0 and len(set(seeds)) == len(seeds), "Invalid expected seeds")
    require(report["seeds"] == [e["seed"] for e in episodes] == seeds, "Seed mismatch")
    require(len(episodes) == len(traces) == report["episode_count"] == len(seeds), "Count mismatch")
    combats = sum(e[k] for e in episodes for k in ("wins", "draws", "losses"))
    require(combats > 0 and combats == report["combat_count"], "Combat count mismatch")
    for key, value in (
        ("mean_reward", sum(e["reward"] for e in episodes) / len(seeds)),
        ("survival_rate", sum(e["survived"] for e in episodes) / len(seeds)),
        ("combat_win_rate", sum(e["wins"] for e in episodes) / combats),
    ):
        require(math.isfinite(report[key]) and math.isclose(report[key], value, abs_tol=1e-10),
                f"Invalid aggregate: {key}")
    for episode, trace in zip(episodes, traces, strict=True):
        steps = trace["steps"]
        require(episode["seed"] == trace["seed"], "Trace seed mismatch")
        require(len(steps) > 0 and len(steps) == episode["actions"], "Action count mismatch")
        require(math.isclose(sum(s["reward"] for s in steps), episode["reward"], abs_tol=1e-10),
                "Trace reward mismatch")
        for step in steps:
            action = step["action"]
            require(type(action) is int and 0 <= action < len(step["action_mask"]), "Invalid action")
            require(action in step["legal_actions"] and step["action_mask"][action], "Illegal action")
        for key, predicate in (
            ("rerolls", lambda s: s["action"] == 1),
            ("freezes", lambda s: s["action"] == 3),
            ("swaps", lambda s: 28 <= s["action"] < 34),
            ("forced_end_turns", lambda s: s["forced_end_turn"]),
        ):
            require(episode[key] == sum(predicate(s) for s in steps), f"Trace mismatch: {key}")
        for key, outcome in (("wins", 1), ("draws", 0), ("losses", -1)):
            require(episode[key] == sum(s["combat_result"] == outcome for s in steps),
                    f"Trace mismatch: {key}")
        require(steps[-1]["terminated"] or steps[-1]["truncated"], "Missing terminal event")
        require(not any(s["terminated"] or s["truncated"] for s in steps[:-1]),
                "Actions after terminal event")
    return {"episodes": len(episodes), "combats": combats,
            "actions": sum(e["actions"] for e in episodes), "trace_verified": True}
