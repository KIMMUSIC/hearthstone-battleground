"""Privacy and RNG boundaries of the actual learner adapter."""

import numpy as np

from hearthstone_ai.lobby_env import LobbyEnv
from hearthstone_ai.lobby_policies import heuristic_policy


def advance(env):
    action = heuristic_policy(env.game.view(env.learner_seat))
    return env.step(action)


def test_opponent_private_board_cannot_change_learner_observation():
    env = LobbyEnv(learner_seat=3)
    before, info = env.reset(seed=17)
    hidden = env.game.players[0].board[0]
    hidden.attack += 123
    after = env.observation()
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    assert set(info) == {"seed", "round", "rank", "learner_alive", "termination_reason", "adapter_event"}
    assert all(not hasattr(card, "entity_id") for card in env.game.view(3).shop if card)
    after["player"][2] = -999
    assert env.game.players[3].hp != -999


def test_interleaved_adapter_games_match_independent_traces_and_seed_stream():
    expected = []
    for seed in (7, 17):
        env = LobbyEnv(max_rounds=3)
        env.reset(seed=seed)
        while True:
            _, _, term, trunc, _ = advance(env)
            if term or trunc:
                break
        expected.append(env.game.trace)
    envs = [LobbyEnv(max_rounds=3), LobbyEnv(max_rounds=3)]
    for env, seed in zip(envs, (7, 17), strict=True):
        env.reset(seed=seed)
    done = [False, False]
    while not all(done):
        for index, env in enumerate(envs):
            if not done[index]:
                _, _, term, trunc, _ = advance(env)
                done[index] = term or trunc
    assert [env.game.trace for env in envs] == expected
    first, second = LobbyEnv(), LobbyEnv()
    first.reset(seed=7)
    second.reset(seed=7)
    for _ in range(12):
        _, a = first.reset()
        envs[0].reset()
        _, b = second.reset()
        assert a["seed"] == b["seed"] and 0 <= a["seed"] <= 9999
