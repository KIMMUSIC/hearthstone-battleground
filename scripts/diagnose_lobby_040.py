"""Read-only information sufficiency check for fixed-v1 lobby observations."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np


PLAYER_SCALE = np.array([7, 100, 30, 10, 2, 5, 1, 24, 3], dtype=np.float32)
PUBLIC_SCALE = np.array([30, 2, 1, 8], dtype=np.float32)
PLAYER_NAMES = (
    "seat",
    "round",
    "hp",
    "gold",
    "tier",
    "upgrade_cost",
    "frozen",
    "actions_remaining",
    "swaps_remaining",
)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dump_fresh(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(f"Preserve previous diagnostic: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_files(root: Path) -> None:
    entries = read(root / "FILES.json")
    for name, expected in entries.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or (root / name).is_symlink():
            raise ValueError(f"Unsafe frozen input path: {name}")
        if digest(path) != expected:
            raise ValueError(f"Frozen input hash mismatch: {name}")


def source_hashes(root: Path) -> dict[str, str]:
    names = [
        "FILES.json",
        "src/hearthstone_ai/lobby.py",
        "src/hearthstone_ai/lobby_env.py",
        "src/hearthstone_ai/lobby_observation.py",
        "src/hearthstone_ai/lobby_policies.py",
        "src/hearthstone_ai/cards.py",
        "data/cards.json",
        "data/card_mapping.json",
        "configs/lobby039_comparison.json",
        "audit.json",
    ]
    return {name: digest(root / name) for name in names}


def final_trace_inputs(root: Path, spec: dict[str, Any]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for seed in spec["seeds"]:
        for arm in spec["arms"]:
            evaluation = root / "outputs" / f"evaluate-s{seed:03d}-{arm}" / "final.json"
            report = read(evaluation)
            result[f"s{seed:03d}-{arm}"] = {
                "final_report": evaluation.relative_to(root).as_posix(),
                "final_report_sha256": digest(evaluation),
                "model_sha256": report["model_sha256"],
            }
    return result


def _near_int(value: float, *, field: str, minimum: int | None = None, maximum: int | None = None) -> int:
    rounded = int(round(float(value)))
    if abs(float(value) - rounded) > 1e-4:
        raise ValueError(f"{field} is not integer-recoverable: {value!r}")
    if minimum is not None and rounded < minimum:
        raise ValueError(f"{field} below minimum: {rounded}")
    if maximum is not None and rounded > maximum:
        raise ValueError(f"{field} above maximum: {rounded}")
    return rounded


def _near_bool(value: float, *, field: str) -> bool:
    rounded = _near_int(value, field=field, minimum=0, maximum=1)
    return bool(rounded)


def _near_half_rank(value: float, *, field: str) -> float | None:
    doubled = _near_int(float(value) * 2, field=f"{field}*2", minimum=0, maximum=16)
    return None if doubled == 0 else doubled / 2


def _decode_minion_row(row: np.ndarray, catalog, inverse_mapping: dict[int, int], *, field: str):
    from hearthstone_ai.lobby import MinionView

    if row.ndim != 1:
        raise ValueError(f"{field} row must be one-dimensional")
    if not np.isfinite(row).all():
        raise ValueError(f"{field} row contains non-finite values")
    feature_start = len(row) - 6
    if feature_start <= 1:
        raise ValueError(f"{field} row is too narrow")
    indicators = row[:feature_start]
    active = np.flatnonzero(np.abs(indicators - 1.0) <= 1e-6)
    inactive_bad = np.flatnonzero((np.abs(indicators) > 1e-6) & (np.abs(indicators - 1.0) > 1e-6))
    if len(inactive_bad):
        raise ValueError(f"{field} row has non-binary one-hot values")
    if len(active) != 1:
        raise ValueError(f"{field} row must have exactly one active one-hot slot")
    active_index = int(active[0])
    tail = row[feature_start:]
    if active_index == 0:
        if np.any(np.abs(tail) > 1e-6):
            raise ValueError(f"{field} padding row has nonzero minion features")
        return None
    if active_index not in inverse_mapping:
        raise ValueError(f"{field} row uses an unmapped card slot")
    card_id = inverse_mapping[active_index]
    if card_id not in catalog.by_id:
        raise ValueError(f"{field} row uses a reserved or unknown card id: {card_id}")
    attack = _near_int(tail[0] * 100, field=f"{field}.attack", minimum=0)
    health = _near_int(tail[1] * 100, field=f"{field}.health", minimum=1)
    golden = _near_bool(tail[2], field=f"{field}.golden")
    taunt = _near_bool(tail[3], field=f"{field}.taunt")
    divine_shield = _near_bool(tail[4], field=f"{field}.divine_shield")
    tier = _near_int(tail[5] * 2, field=f"{field}.tier", minimum=1, maximum=2)
    if tier != catalog.by_id[card_id].tier:
        raise ValueError(f"{field} tier does not match catalog")
    return MinionView(card_id, attack, health, golden, taunt, divine_shield)


def _decode_collection(rows: np.ndarray, catalog, inverse_mapping: dict[int, int], *, field: str, keep_none: bool):
    decoded = [
        _decode_minion_row(np.asarray(row, dtype=np.float32), catalog, inverse_mapping, field=f"{field}[{index}]")
        for index, row in enumerate(rows)
    ]
    if keep_none:
        return tuple(decoded)
    compact = [item for item in decoded if item is not None]
    if any(item is not None for item in decoded[len(compact):]):
        raise ValueError(f"{field} contains a gap before a minion")
    return tuple(compact)


def decode_observation(obs: dict[str, np.ndarray], catalog):
    from hearthstone_ai.lobby import PolicyView, PublicPlayerView

    required = {"player", "public_players", "shop", "board", "hand", "discover", "action_mask"}
    if set(obs) != required:
        raise ValueError(f"Observation keys changed: {sorted(obs)}")
    player = np.asarray(obs["player"], dtype=np.float32) * PLAYER_SCALE
    public = np.asarray(obs["public_players"], dtype=np.float32) * PUBLIC_SCALE
    mask = np.asarray(obs["action_mask"])
    if player.shape != (9,) or public.shape != (8, 4) or mask.shape != (37,):
        raise ValueError("Observation has an unexpected shape")
    if not all(np.isfinite(np.asarray(obs[key])).all() for key in required):
        raise ValueError("Observation contains non-finite values")
    if not np.isin(mask, [0, 1]).all():
        raise ValueError("action_mask must be binary")

    inverse_mapping = {slot: card_id for card_id, slot in catalog.mapping.items()}
    player_values = {
        name: _near_int(value, field=f"player.{name}") for name, value in zip(PLAYER_NAMES, player, strict=True)
    }
    player_values["frozen"] = bool(player_values["frozen"])
    public_players = []
    for seat, row in enumerate(public):
        public_players.append(
            PublicPlayerView(
                seat=seat,
                hp=_near_int(row[0], field=f"public_players[{seat}].hp"),
                tier=_near_int(row[1], field=f"public_players[{seat}].tier", minimum=1, maximum=2),
                alive=_near_bool(row[2], field=f"public_players[{seat}].alive"),
                rank=_near_half_rank(row[3], field=f"public_players[{seat}].rank"),
            )
        )
    return PolicyView(
        seat=player_values["seat"],
        round=player_values["round"],
        hp=player_values["hp"],
        gold=player_values["gold"],
        tier=player_values["tier"],
        upgrade_cost=player_values["upgrade_cost"],
        frozen=player_values["frozen"],
        actions_remaining=player_values["actions_remaining"],
        swaps_remaining=player_values["swaps_remaining"],
        hand=_decode_collection(np.asarray(obs["hand"]), catalog, inverse_mapping, field="hand", keep_none=False),
        board=_decode_collection(np.asarray(obs["board"]), catalog, inverse_mapping, field="board", keep_none=False),
        shop=_decode_collection(np.asarray(obs["shop"]), catalog, inverse_mapping, field="shop", keep_none=True),
        pending_discover=_decode_collection(
            np.asarray(obs["discover"]), catalog, inverse_mapping, field="discover", keep_none=False
        ),
        public_players=tuple(public_players),
        legal_actions=tuple(int(i) for i in np.flatnonzero(mask)),
        cards=tuple(catalog.cards),
    )


def hash_state(hasher, label: str, seed: int, step: int, obs: dict[str, np.ndarray], action: int) -> None:
    hasher.update(label.encode("utf-8"))
    hasher.update(str(seed).encode("ascii"))
    hasher.update(str(step).encode("ascii"))
    hasher.update(str(action).encode("ascii"))
    for key in sorted(obs):
        value = np.ascontiguousarray(obs[key])
        hasher.update(key.encode("ascii"))
        hasher.update(str(value.shape).encode("ascii"))
        hasher.update(str(value.dtype).encode("ascii"))
        hasher.update(value.tobytes())


def record_state(
    *,
    label: str,
    seed: int,
    step: int,
    obs: dict[str, np.ndarray],
    original_view,
    catalog,
    heuristic_policy,
    counters: dict[str, Any],
    hasher,
) -> None:
    decoded = decode_observation(obs, catalog)
    original_action = heuristic_policy(original_view)
    decoded_action = heuristic_policy(decoded)
    counters["states"] += 1
    counters["original_heuristic_actions"][str(original_action)] += 1
    counters["decoded_heuristic_actions"][str(decoded_action)] += 1
    if decoded_action != original_action:
        counters["mismatches"].append(
            {
                "label": label,
                "seed": seed,
                "step": step,
                "original_heuristic": original_action,
                "decoded_heuristic": decoded_action,
            }
        )
    hash_state(hasher, label, seed, step, obs, original_action)


def run_policy_episodes(spec: dict[str, Any], catalog, heuristic_policy, random_policy, counters, hasher):
    from hearthstone_ai.lobby_env import LobbyEnv

    details = {}
    for policy in ("heuristic", "random"):
        details[policy] = {"episodes": 0, "states": 0, "actions": Counter()}
        for seed in spec["eval_seeds"]:
            env = LobbyEnv(
                learner_seat=spec["train"]["learner_seat"],
                max_rounds=spec["train"]["max_rounds"],
                heuristic_opponents=7,
                observation_scale=spec["train"]["observation_scale"],
            )
            obs, _ = env.reset(seed=seed)
            rng = random.Random(f"lobby-probe:{seed}:policy:0")
            done = False
            step = 0
            while not done:
                view = env.game.view(env.learner_seat)
                record_state(
                    label=f"policy-{policy}",
                    seed=seed,
                    step=step,
                    obs=obs,
                    original_view=view,
                    catalog=catalog,
                    heuristic_policy=heuristic_policy,
                    counters=counters,
                    hasher=hasher,
                )
                action = heuristic_policy(view) if policy == "heuristic" else random_policy(view, rng)
                if not env.action_masks()[action] or not env.observation_space.contains(obs):
                    raise ValueError("Generated policy episode reached an invalid state/action")
                obs, _, terminated, truncated, _ = env.step(action)
                env.game.assert_conservation()
                details[policy]["states"] += 1
                details[policy]["actions"][str(action)] += 1
                step += 1
                done = terminated or truncated
            details[policy]["episodes"] += 1
            env.close()
    for detail in details.values():
        detail["actions"] = dict(detail["actions"])
    return details


def replay_final_traces(root: Path, spec: dict[str, Any], catalog, heuristic_policy, counters, hasher):
    from hearthstone_ai.lobby_env import LobbyEnv

    traces = {}
    for train_seed in spec["seeds"]:
        for arm in spec["arms"]:
            key = f"s{train_seed:03d}-{arm}"
            report = read(root / "outputs" / f"evaluate-s{train_seed:03d}-{arm}" / "final.json")
            traces[key] = {"episodes": 0, "states": 0, "actions": Counter(), "model_sha256": report["model_sha256"]}
            if [episode["seed"] for episode in report["episodes"]] != spec["eval_seeds"]:
                raise ValueError(f"Unexpected saved eval seeds: {key}")
            for episode in report["episodes"]:
                env = LobbyEnv(
                    learner_seat=spec["train"]["learner_seat"],
                    max_rounds=spec["train"]["max_rounds"],
                    heuristic_opponents=7,
                    observation_scale=spec["train"]["observation_scale"],
                )
                obs, _ = env.reset(seed=episode["seed"])
                rewards = []
                done = False
                for step, action in enumerate(episode["decisions"]):
                    view = env.game.view(env.learner_seat)
                    if done or not env.action_masks()[action] or not env.observation_space.contains(obs):
                        raise ValueError(f"Invalid saved trace action: {key} seed {episode['seed']} step {step}")
                    record_state(
                        label=f"trace-{key}",
                        seed=episode["seed"],
                        step=step,
                        obs=obs,
                        original_view=view,
                        catalog=catalog,
                        heuristic_policy=heuristic_policy,
                        counters=counters,
                        hasher=hasher,
                    )
                    obs, reward, terminated, truncated, info = env.step(action)
                    env.game.assert_conservation()
                    rewards.append(reward)
                    traces[key]["states"] += 1
                    traces[key]["actions"][str(action)] += 1
                    done = terminated or truncated
                rank = env.game.players[env.learner_seat].rank
                if (
                    not done
                    or rank != episode["rank"]
                    or sum(rewards) != episode["reward"]
                    or terminated != episode["terminated"]
                    or truncated != episode["truncated"]
                    or info != episode["info"]
                    or json.loads(json.dumps(env.game.trace)) != episode["trace"]
                    or not env.observation_space.contains(obs)
                ):
                    raise ValueError(f"Saved final trace replay mismatch: {key} seed {episode['seed']}")
                traces[key]["episodes"] += 1
                env.close()
    for detail in traces.values():
        detail["actions"] = dict(detail["actions"])
    return traces


def diagnose(run_root: Path, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    if output.exists():
        raise FileExistsError(f"Preserve previous diagnostic: {output}")
    run_root = run_root.resolve()
    verify_files(run_root)
    spec = read(run_root / "configs/lobby039_comparison.json")
    if spec["eval_seeds"] != list(range(20001, 20021)):
        raise ValueError("040 only uses dev eval seeds 20001..20020")
    if spec["train"].get("observation_scale") != "fixed-v1":
        raise ValueError("040 is scoped to fixed-v1 observations")
    sys.path.insert(0, str(run_root / "src"))

    import torch
    from hearthstone_ai import lobby_env
    from hearthstone_ai.cards import Catalog
    from hearthstone_ai.lobby_policies import heuristic_policy, random_policy

    if not Path(lobby_env.__file__).resolve().is_relative_to(run_root / "src"):
        raise ValueError("Frozen run source was not imported")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    catalog = Catalog(run_root / "data")
    counters = {"states": 0, "mismatches": [], "original_heuristic_actions": Counter(), "decoded_heuristic_actions": Counter()}
    hasher = hashlib.sha256()
    generated = run_policy_episodes(spec, catalog, heuristic_policy, random_policy, counters, hasher)
    saved_traces = replay_final_traces(run_root, spec, catalog, heuristic_policy, counters, hasher)
    mismatch_count = len(counters["mismatches"])
    verify_files(run_root)
    result = {
        "status": "passed",
        "diagnostic": "lobby040-fixed-v1-heuristic-information-sufficiency",
        "run": str(run_root),
        "frozen_inputs_unchanged": True,
        "cpu_threads": {"torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads()},
        "source_hashes": source_hashes(run_root),
        "trace_inputs": final_trace_inputs(run_root, spec),
        "episodes": {
            "heuristic_policy": generated["heuristic"]["episodes"],
            "random_policy": generated["random"]["episodes"],
            "saved_039_final_traces": sum(trace["episodes"] for trace in saved_traces.values()),
            "total": generated["heuristic"]["episodes"]
            + generated["random"]["episodes"]
            + sum(trace["episodes"] for trace in saved_traces.values()),
        },
        "states": {
            "total": counters["states"],
            "heuristic_policy": generated["heuristic"]["states"],
            "random_policy": generated["random"]["states"],
            "saved_039_final_traces": sum(trace["states"] for trace in saved_traces.values()),
            "sha256": hasher.hexdigest(),
        },
        "decode_vs_original_heuristic": {
            "mismatch_count": mismatch_count,
            "match_count": counters["states"] - mismatch_count,
            "mismatch_fraction": mismatch_count / counters["states"] if counters["states"] else None,
            "mismatches": counters["mismatches"][:20],
            "original_heuristic_actions": dict(counters["original_heuristic_actions"]),
            "decoded_heuristic_actions": dict(counters["decoded_heuristic_actions"]),
        },
        "generated_policy_episodes": generated,
        "saved_final_traces": saved_traces,
        "claim_boundary": [
            "This verifies whether fixed-v1 numeric observations plus frozen public Catalog reconstruct the current heuristic decision.",
            "No hidden game/view input is used by the decoded policy path.",
            "This is not a causal claim about PPO learning, neural learnability, or policy performance.",
            "No training, dependency change, model inference repeat, source edit, or run mutation is performed.",
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    if result["episodes"]["total"] != 160:
        raise ValueError(f"Unexpected episode count: {result['episodes']['total']}")
    dump_fresh(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/lobby-039-local-r1"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/diagnostics/lobby040_information_sufficiency.json"),
    )
    args = parser.parse_args()
    print(json.dumps(diagnose(args.run, args.output), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
