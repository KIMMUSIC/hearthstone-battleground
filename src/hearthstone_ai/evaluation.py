"""Fixed opponent evaluation without changing the training opponent distribution."""

import json
import random
from pathlib import Path

from .artifacts import ROOT, atomic_json, compatibility_signature, content_hash, file_hash
from .env import BgEnv


def strength(minion):
    return minion.attack + minion.health + 2 * minion.divine_shield + minion.taunt


def heuristic_action(game):
    legal = game.legal_actions()
    if game.pending_discover:
        return 34 + max(
            range(len(game.pending_discover)), key=lambda i: strength(game.pending_discover[i])
        )
    if len(game.board) < 7 and game.hand:
        return 18 + max(range(len(game.hand)), key=lambda i: strength(game.hand[i]))
    if len(game.board) == 7 and game.hand:
        weakest = min(range(len(game.board)), key=lambda i: strength(game.board[i]))
        if max(strength(m) for m in game.hand) > strength(game.board[weakest]):
            return 11 + weakest
    if 2 in legal and len(game.board) >= 3:
        return 2
    buys = [a for a in legal if 4 <= a < 11]
    if buys and len(game.hand) < 3:

        def value(action):
            minion = game.shop[action - 4]
            copies = sum(
                m.card_id == minion.card_id
                and not m.golden
                and game.catalog.by_id[m.card_id].triple_allowed
                for m in game.board + game.hand
            )
            return strength(minion) + 3 * copies

        best = max(buys, key=value)
        if len(game.board) < 7 or value(best) > min(strength(m) for m in game.board):
            return best
    if 1 in legal and game.gold >= 4:
        return 1
    return 0


def evaluate(
    *,
    policy="heuristic",
    seeds=(101, 102, 103, 104, 105),
    opponent_path: Path | None = None,
    checkpoint: Path | None = None,
    max_turns=8,
    max_actions=24,
    trace=False,
    trace_output: Path | None = None,
):
    seeds = list(seeds)
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Evaluation requires nonnegative integer seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation seeds must be unique")
    if policy not in {"random", "heuristic", "model"}:
        raise ValueError("Unknown evaluation policy")
    payload = json.loads(
        (opponent_path or ROOT / "configs/eval_opponents.json").read_text(encoding="utf-8")
    )
    opponents = payload["opponents"]
    original_hash = content_hash(payload)
    env = BgEnv(opponents=opponents, max_turns=max_turns, max_actions=max_actions)
    try:
        model = None
        if policy == "model":
            if checkpoint is None:
                raise ValueError("Model evaluation requires a checkpoint")
            from .training import load_model

            model = load_model(checkpoint, env)
        episodes = []
        trace_enabled = trace or trace_output is not None
        traces = []
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            policy_rng = random.Random(f"evaluation-policy:{seed}")
            episode_trace = [] if trace_enabled else None
            metrics = dict(
                seed=seed,
                reward=0.0,
                wins=0,
                draws=0,
                losses=0,
                damage_taken=0,
                damage_dealt=0,
                forced_end_turns=0,
                rerolls=0,
                freezes=0,
                swaps=0,
                actions=0,
            )
            # Each turn has bounded normal actions plus at most one final discover.
            for _ in range(max_turns * (max_actions + 2)):
                mask = env.action_masks()
                if episode_trace is not None:
                    game = env.game
                    pre_step = {
                        "step": metrics["actions"],
                        "turn": game.turn,
                        "gold": game.gold,
                        "hp": game.hp,
                        "actions_remaining": game.actions_remaining,
                        "board_count": len(game.board),
                        "hand_count": len(game.hand),
                        "shop_count": len(game.shop),
                        "discover_count": len(game.pending_discover),
                        "player_observation": obs["player"].tolist(),
                        "legal_actions": [index for index, allowed in enumerate(mask) if allowed],
                        "action_mask": [bool(value) for value in mask.tolist()],
                    }
                if model is not None:
                    action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                    action = int(action)
                elif policy == "random":
                    action = policy_rng.choice(env.game.legal_actions())
                else:
                    action = heuristic_action(env.game)
                obs, reward, terminated, truncated, info = env.step(action)
                if episode_trace is not None:
                    pre_step["action"] = action
                    pre_step["reward"] = reward
                    pre_step["terminated"] = bool(terminated)
                    pre_step["truncated"] = bool(truncated)
                    pre_step["combat_result"] = info.get("combat_result")
                    pre_step["forced_end_turn"] = bool(info.get("forced_end_turn", False))
                    episode_trace.append(pre_step)
                metrics["actions"] += 1
                metrics["reward"] += reward
                metrics["rerolls"] += action == 1
                metrics["freezes"] += action == 3
                metrics["swaps"] += 28 <= action < 34
                if "combat_result" in info:
                    metrics[{1: "wins", 0: "draws", -1: "losses"}[info["combat_result"]]] += 1
                    for key in ("damage_taken", "damage_dealt"):
                        metrics[key] += info[key]
                    metrics["forced_end_turns"] += bool(info["forced_end_turn"])
                if terminated or truncated:
                    metrics.update(
                        hp=env.game.hp,
                        last_turn=env.game.turn,
                        survived=env.game.hp > 0,
                        termination_reason=env.game.termination_reason,
                    )
                    episodes.append(metrics)
                    if episode_trace is not None:
                        traces.append({"seed": seed, "steps": episode_trace})
                    break
            else:
                raise RuntimeError("Evaluation exceeded the engine's action bound")
        assert original_hash == content_hash(payload), "Evaluation modified opponent inputs"
        combat_count = sum(row["wins"] + row["draws"] + row["losses"] for row in episodes)
        model_evidence = None
        if checkpoint is not None and policy == "model":
            model_path = checkpoint / "model.zip" if checkpoint.is_dir() else checkpoint
            model_evidence = {"path": str(model_path.resolve()), "sha256": file_hash(model_path)}
        result = {
            "policy": policy,
            "compatibility": compatibility_signature(),
            "model": model_evidence,
            "opponent_set_sha256": original_hash,
            "seeds": seeds,
            "max_turns": max_turns,
            "max_actions": max_actions,
            "episodes": episodes,
            "episode_count": len(episodes),
            "combat_count": combat_count,
            "combat_win_rate": sum(row["wins"] for row in episodes) / combat_count,
            "survival_rate": sum(row["survived"] for row in episodes) / len(episodes),
            "mean_reward": sum(row["reward"] for row in episodes) / len(episodes),
            "interpretation": "descriptive small-sample metrics; not live Battlegrounds skill",
        }
        if trace_enabled:
            trace_payload = {
                "policy": policy,
                "seeds": seeds,
                "max_turns": max_turns,
                "max_actions": max_actions,
                "episodes": traces,
            }
            if trace:
                result["trace"] = trace_payload
            if trace_output is not None:
                atomic_json(trace_output, trace_payload)
        return result
    finally:
        env.close()
