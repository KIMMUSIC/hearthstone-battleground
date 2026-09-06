"""Gymnasium learner-seat contract for the restricted eight-player lobby."""

import numpy as np
import pytest

from hearthstone_ai.lobby_env import LobbyEnv
from hearthstone_ai.lobby_policies import heuristic_policy


def legal(mask):
    return np.flatnonzero(mask).tolist()


def test_reset_returns_public_observation_and_mask_for_learner():
    env = LobbyEnv(learner_seat=0, max_rounds=20)
    obs, info = env.reset(seed=20001)
    assert info == {
        "adapter_event": "reset",
        "learner_alive": True,
        "rank": None,
        "round": 1,
        "seed": 20001,
        "termination_reason": None,
    }
    assert env.observation_space.contains(obs)
    assert legal(env.action_masks()) == env.game.legal_actions(0)
    assert env.game.current_seat == 0
    assert "trace" not in info and "players" not in info and "board" not in info


def test_auto_play_advances_back_to_learner_deterministically():
    actions = []
    env = LobbyEnv(max_rounds=50)
    obs, _ = env.reset(seed=7)
    for _ in range(6):
        action = heuristic_policy(env.game.view(env.learner_seat))
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            break
        assert env.game.current_seat == env.learner_seat
        assert reward == 0.0
        assert info["rank"] is None

    replay = LobbyEnv(max_rounds=50)
    replay.reset(seed=7)
    replay_actions = []
    for _ in range(len(actions)):
        action = heuristic_policy(replay.game.view(replay.learner_seat))
        replay_actions.append(action)
        obs, reward, terminated, truncated, info = replay.step(action)
        if terminated or truncated:
            break
    assert replay_actions == actions
    assert replay.game.trace == env.game.trace


def test_episode_ends_when_learner_rank_is_known_and_reward_is_rank_scaled():
    env = LobbyEnv(max_rounds=100)
    obs, _ = env.reset(seed=0)
    for _ in range(1000):
        action = heuristic_policy(env.game.view(env.learner_seat))
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            break
    else:
        pytest.fail("lobby learner episode did not end")
    assert terminated and not truncated
    assert info["rank"] is not None
    assert reward == pytest.approx((4.5 - info["rank"]) / 3.5)
    assert not env.action_masks().any()
    with pytest.raises(RuntimeError):
        env.step(0)


def test_round_limit_truncates_without_fake_rank_or_reward():
    env = LobbyEnv(max_rounds=1)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert env.observation_space.contains(obs)
    assert not terminated and truncated
    assert reward == 0.0
    assert info["rank"] is None
    assert info["termination_reason"] == "max_rounds"


def test_learner_seat_can_start_after_heuristic_opponents():
    env = LobbyEnv(learner_seat=3, max_rounds=20)
    obs, info = env.reset(seed=17)
    assert env.game.current_seat == 3
    assert obs["player"][0] == 3
    assert info["rank"] is None
    assert legal(env.action_masks()) == env.game.legal_actions(3)


def test_invalid_actions_and_reset_seeds_are_rejected():
    with pytest.raises(ValueError):
        LobbyEnv(learner_seat=8)
    with pytest.raises(ValueError):
        LobbyEnv(train_seed_min=10, train_seed_max=9)
    env = LobbyEnv()
    with pytest.raises(RuntimeError):
        env.step(0)
    with pytest.raises(ValueError):
        env.reset(seed=-1)
    env.reset(seed=1)
    illegal = next(action for action in range(env.action_space.n) if action not in env.game.legal_actions(0))
    with pytest.raises(ValueError):
        env.step(illegal)
