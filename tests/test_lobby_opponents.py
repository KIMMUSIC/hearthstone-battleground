import random

import pytest

from hearthstone_ai.lobby import LobbyGame
from hearthstone_ai.lobby_env import LobbyEnv
from hearthstone_ai.lobby_policies import heuristic_policy, random_policy


def play_env(seed, *, heuristic_opponents=7, max_rounds=20):
    env = LobbyEnv(max_rounds=max_rounds, heuristic_opponents=heuristic_opponents)
    env.reset(seed=seed)
    return drain_env(env)


def drain_env(env):
    while True:
        action = heuristic_policy(env.game.view(env.learner_seat))
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return env.game.trace


def play_direct(seed, *, heuristic_opponents=7, max_rounds=20):
    game = LobbyGame(seed=seed, max_rounds=max_rounds)
    rngs = [random.Random(f"lobby-probe:{seed}:policy:{seat}") for seat in range(8)]
    while not game.done and game.players[0].rank is None:
        seat = game.current_seat
        view = game.view(seat)
        if seat == 0 or seat <= heuristic_opponents:
            action = heuristic_policy(view)
        else:
            action = random_policy(view, rngs[seat])
        game.step(seat, action)
    return game.trace


def test_default_seven_heuristic_opponents_preserves_direct_trace():
    assert play_env(17, heuristic_opponents=7) == play_direct(17, heuristic_opponents=7)


def test_mixed_opponents_match_probe_rng_and_interleaved_resets():
    expected = [
        play_direct(seed, heuristic_opponents=3, max_rounds=4)
        for seed in (7, 17)
    ]
    envs = [
        LobbyEnv(max_rounds=4, heuristic_opponents=3),
        LobbyEnv(max_rounds=4, heuristic_opponents=3),
    ]
    for env, seed in zip(envs, (7, 17), strict=True):
        env.reset(seed=seed)
    done = [False, False]
    while not all(done):
        for index, env in enumerate(envs):
            if done[index]:
                continue
            action = heuristic_policy(env.game.view(env.learner_seat))
            _, _, terminated, truncated, _ = env.step(action)
            done[index] = terminated or truncated
    assert [env.game.trace for env in envs] == expected

    env = LobbyEnv(max_rounds=4, heuristic_opponents=3)
    env.reset(seed=31)
    first = drain_env(env)
    env.reset(seed=31)
    second = drain_env(env)
    assert first == second == play_direct(31, heuristic_opponents=3, max_rounds=4)


@pytest.mark.parametrize("count", [-1, True, 8])
def test_constructor_rejects_invalid_heuristic_opponent_counts(count):
    with pytest.raises(ValueError, match="heuristic_opponents"):
        LobbyEnv(heuristic_opponents=count)


def test_opponent_spec_describes_mixed_policy_order():
    env = LobbyEnv(learner_seat=3, heuristic_opponents=2)
    assert env.opponent_spec == {
        "version": "lobby-opponent-mix-v1",
        "learner_seat": 3,
        "heuristic_opponents": 2,
        "opponent_count": 7,
        "ordering": "ascending-seat-excluding-learner",
        "heuristic_policy": "heuristic_policy",
        "random_policy": "random_policy",
        "random_seed_protocol": "lobby-probe:{seed}:policy:{seat}",
        "opponent_seats": [0, 1, 2, 4, 5, 6, 7],
        "heuristic_seats": [0, 1],
        "random_seats": [2, 4, 5, 6, 7],
    }
