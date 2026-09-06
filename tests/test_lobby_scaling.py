from types import SimpleNamespace

import numpy as np
import pytest

from hearthstone_ai.cards import Catalog
from hearthstone_ai.lobby_env import LobbyEnv
from hearthstone_ai.lobby_observation import LobbyObservation
from hearthstone_ai.lobby_policies import heuristic_policy


def view():
    return SimpleNamespace(
        seat=7,
        round=100,
        hp=-15,
        gold=10,
        tier=2,
        upgrade_cost=5,
        frozen=True,
        actions_remaining=24,
        swaps_remaining=3,
        shop=(None,) * 7,
        board=(),
        hand=(),
        pending_discover=(),
        public_players=tuple(
            SimpleNamespace(seat=i, hp=-15 if i == 0 else 30, tier=2, alive=i != 0, rank=8 if i == 0 else None)
            for i in range(8)
        ),
        legal_actions=(0, 3),
        cards=Catalog().cards,
    )


def test_raw_observation_scale_preserves_existing_numeric_contract():
    state = view()
    obs = LobbyObservation().encode(state)
    np.testing.assert_array_equal(
        obs["player"],
        np.array([7, 100, -15, 10, 2, 5, 1, 24, 3], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        obs["public_players"][0],
        np.array([-15, 2, 0, 8], dtype=np.float32),
    )
    assert np.flatnonzero(obs["action_mask"]).tolist() == [0, 3]


def test_fixed_observation_scale_divides_player_and_public_without_clipping():
    state = view()
    raw = LobbyObservation(observation_scale="raw").encode(state)
    fixed = LobbyObservation(observation_scale="fixed-v1").encode(state)
    np.testing.assert_allclose(
        fixed["player"],
        np.array([1, 1, -0.5, 1, 1, 1, 1, 1, 1], dtype=np.float32),
    )
    np.testing.assert_allclose(
        fixed["public_players"][0],
        np.array([-0.5, 1, 0, 1], dtype=np.float32),
    )
    for key in ("shop", "board", "hand", "discover", "action_mask"):
        np.testing.assert_array_equal(fixed[key], raw[key])
    assert np.isfinite(fixed["player"]).all()
    assert np.isfinite(fixed["public_players"]).all()
    assert fixed["shop"].shape == raw["shop"].shape == (7, raw["shop"].shape[1])


def test_raw_and_fixed_envs_share_actions_trace_reward_and_masks():
    raw = LobbyEnv(max_rounds=20, heuristic_opponents=3, observation_scale="raw")
    fixed = LobbyEnv(max_rounds=20, heuristic_opponents=3, observation_scale="fixed-v1")
    raw_obs, _ = raw.reset(seed=17)
    fixed_obs, _ = fixed.reset(seed=17)
    np.testing.assert_array_equal(raw.action_masks(), fixed.action_masks())
    np.testing.assert_array_equal(raw_obs["action_mask"], fixed_obs["action_mask"])

    while True:
        raw_action = heuristic_policy(raw.game.view(raw.learner_seat))
        fixed_action = heuristic_policy(fixed.game.view(fixed.learner_seat))
        assert raw_action == fixed_action
        raw_step = raw.step(raw_action)
        fixed_step = fixed.step(fixed_action)
        np.testing.assert_array_equal(raw.action_masks(), fixed.action_masks())
        np.testing.assert_array_equal(raw_step[0]["action_mask"], fixed_step[0]["action_mask"])
        assert raw_step[1:] == fixed_step[1:]
        if raw_step[2] or raw_step[3]:
            break

    assert raw.game.trace == fixed.game.trace


@pytest.mark.parametrize("scale", ["bad-scale", "", None])
def test_invalid_observation_scale_is_rejected(scale):
    with pytest.raises(ValueError, match="observation_scale"):
        LobbyObservation(observation_scale=scale)
    with pytest.raises(ValueError, match="observation_scale"):
        LobbyEnv(observation_scale=scale)
