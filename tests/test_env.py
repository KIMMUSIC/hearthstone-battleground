import numpy as np
import pytest

from hearthstone_ai.env import BgEnv


def test_registered_golem_and_token_have_distinct_observation_slots():
    env = BgEnv()
    env.reset(seed=16)
    env.game.hand = [env.game.catalog.spawn(96763, 100)]
    env.step(18)
    # Tokens normally live only in combat copies; encoding still must recognize the ID.
    env.game.board.append(env.game.catalog.spawn(96764, 101))
    observation = env.observation()
    assert observation["board"].shape == (7, 18)
    assert observation["board"][0, :7].tolist() == [0, 0, 0, 0, 0, 1, 0]
    assert observation["board"][1, :7].tolist() == [0, 0, 0, 0, 0, 0, 1]
    assert observation["board"][2, :7].tolist() == [1, 0, 0, 0, 0, 0, 0]
    env.game.board[0].card_id = 999999
    with pytest.raises(KeyError):
        env.observation()


def test_second_bundle_cards_are_observable_without_reserved_slot_aliasing():
    env = BgEnv()
    env.reset(seed=17)
    new_ids = [60628, 96796]
    env.game.board = [env.game.catalog.spawn(card_id, index + 1) for index, card_id in enumerate(new_ids)]
    observation = env.observation()
    assert observation["board"].shape == (7, 18)
    for row, card_id in enumerate(new_ids):
        assert observation["board"][row, env.game.catalog.mapping[card_id]] == 1
    reserved_indices = [env.game.catalog.mapping[card_id] for card_id in (49278, 103652, 108715)]
    assert observation["board"][:, reserved_indices].sum() == 0
    vocab = env.slot_width - 6
    assert observation["board"][0, vocab + 2] == 0
    assert observation["board"][0, vocab + 3] == 1
    assert observation["board"][0, vocab + 4] == 0


def test_signed_terminal_hp_is_valid():
    env = BgEnv()
    env.reset(seed=1)
    env.game.hp = -9
    assert env.observation_space.contains(env.observation())


def test_overflow_never_silently_truncates():
    env = BgEnv()
    env.reset(seed=1)
    card_id = next(iter(env.game.catalog.by_id))
    env.game.hand = [env.game.catalog.spawn(card_id, i) for i in range(11)]
    with pytest.raises(ValueError, match="capacity"):
        env.observation()


def test_slot_position_and_golden_are_observable():
    env = BgEnv()
    env.reset(seed=1)
    env.game.board = [env.game.catalog.spawn(42467, 1), env.game.catalog.spawn(72387, 2)]
    before = env.observation()["board"].copy()
    env.game.board.reverse()
    assert not np.array_equal(before, env.observation()["board"])
    env.game.board[0].golden = True
    assert env.observation()["board"][0, -4] == 1


def test_masked_rollouts_reproducible_and_in_space():
    envs = [BgEnv(), BgEnv()]
    observations = [env.reset(seed=42)[0] for env in envs]
    rng = np.random.default_rng(900)
    for _ in range(500):
        assert all(np.array_equal(observations[0][k], observations[1][k]) for k in observations[0])
        masks = [env.action_masks() for env in envs]
        assert np.array_equal(*masks)
        action = int(rng.choice(np.flatnonzero(masks[0])))
        results = [env.step(action) for env in envs]
        observations = [result[0] for result in results]
        assert results[0][1:] == results[1][1:]
        assert all(
            env.observation_space.contains(obs) for env, obs in zip(envs, observations, strict=True)
        )
        if results[0][2] or results[0][3]:
            break
    else:
        pytest.fail("Episode failed to terminate")


def test_invalid_action_is_rejected():
    env = BgEnv()
    env.reset(seed=1)
    for action in (-1, 37, 1.5):
        with pytest.raises(ValueError):
            env.step(action)


def test_opponent_mode_is_forwarded_without_observation_leakage():
    env = BgEnv(opponent_mode="diverse-v1")
    obs, _ = env.reset(seed=5)
    assert env.opponent_mode == "diverse-v1"
    assert env.game.opponent_mode == "diverse-v1"
    assert "opponent" not in obs
    result = env.step(0)
    assert result[4]["opponent_mode"] == "diverse-v1"
