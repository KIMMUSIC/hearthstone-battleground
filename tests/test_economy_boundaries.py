import copy

import numpy as np
import pytest

from hearthstone_ai.env import BgEnv
from hearthstone_ai.game import Game


LEGACY_TRIPLE_CARD_ID = 42467
FILLER_CARD_ID = 72387


def test_reachable_buy_sell_upgrade_gold_flow_and_cap():
    game = Game(opponents=[{"tier": 1, "board": []}])
    game.reset(18)
    game.shop[0] = game._spawn(LEGACY_TRIPLE_CARD_ID)

    assert game.gold == 3
    result = game.step(4)
    assert result.reward == 0.0
    assert game.gold == 0
    assert len(game.hand) == 1
    assert game.shop[0] is None

    game.step(18)
    assert len(game.board) == 1
    assert len(game.hand) == 0

    game.step(11)
    assert game.gold == 1
    assert game.board == []

    result = game.step(0)
    assert result.info["turn"] == 1
    assert game.turn == 2
    assert game.gold == 4
    assert game.upgrade_cost == 4

    result = game.step(2)
    assert result.reward == 0.0
    assert game.tier == 2
    assert game.gold == 0
    assert game.upgrade_cost == 0
    assert 2 not in game.legal_actions()

    game.board = [game._spawn(LEGACY_TRIPLE_CARD_ID)]
    game.gold = 10
    game.step(11)
    assert game.gold == 10


def test_full_hand_and_full_board_boundaries_reject_without_mutation():
    game = Game()
    game.reset(18)
    game.hand = [game._spawn(LEGACY_TRIPLE_CARD_ID) for _ in range(10)]
    game.shop[0] = game._spawn(LEGACY_TRIPLE_CARD_ID)
    before = copy.deepcopy((game.gold, game.shop, game.hand, game.board, game.actions_remaining))

    assert all(action not in game.legal_actions() for action in range(4, 11))
    with pytest.raises(ValueError, match="Illegal action 4"):
        game.step(4)
    assert (game.gold, game.shop, game.hand, game.board, game.actions_remaining) == before

    game.hand = [game._spawn(LEGACY_TRIPLE_CARD_ID)]
    game.board = [game._spawn(FILLER_CARD_ID) for _ in range(7)]
    before = copy.deepcopy((game.gold, game.hand, game.board, game.actions_remaining))

    assert all(action not in game.legal_actions() for action in range(18, 28))
    with pytest.raises(ValueError, match="Illegal action 18"):
        game.step(18)
    assert (game.gold, game.hand, game.board, game.actions_remaining) == before


def test_triple_golden_preserves_buffs_and_frees_capacity():
    game = Game()
    game.reset(18)
    card = game.catalog.by_id[LEGACY_TRIPLE_CARD_ID]

    board_copy = game._spawn(LEGACY_TRIPLE_CARD_ID)
    board_copy.attack += 3
    board_copy.health += 4
    hand_copy = game._spawn(LEGACY_TRIPLE_CARD_ID)
    hand_copy.attack += 2
    non_triple_filler_id = next(
        card.dbf_id
        for card in game.catalog.cards
        if card.dbf_id != LEGACY_TRIPLE_CARD_ID and not card.triple_allowed
    )
    filler = [game._spawn(non_triple_filler_id) for _ in range(8)]

    game.board = [board_copy]
    game.hand = [hand_copy, *filler]
    game.shop[0] = game._spawn(LEGACY_TRIPLE_CARD_ID)
    game.gold = 3

    game.step(4)

    assert game.board == []
    assert len(game.hand) == 9
    golden = game.hand[-1]
    assert golden.card_id == LEGACY_TRIPLE_CARD_ID
    assert golden.golden
    assert golden.attack == (2 * card.attack) + 5
    assert golden.health == (2 * card.health) + 4
    assert golden.taunt
    assert golden.divine_shield


def test_modal_discover_mask_and_last_action_forced_combat():
    env = BgEnv(max_actions=1, opponents=[{"tier": 1, "board": []}])
    env.reset(seed=18)
    env.game.hand = [env.game.catalog.spawn(LEGACY_TRIPLE_CARD_ID, 100, golden=True)]

    obs, reward, terminated, truncated, info = env.step(18)

    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert info == {"action": 18}
    assert env.game.actions_remaining == 0
    assert obs["player"][-1] == 1
    assert [card.tier for card in env.game.pending_discover] == [2, 2, 2]
    assert np.flatnonzero(env.action_masks()).tolist() == [34, 35, 36]

    pending_before = list(env.game.pending_discover)
    with pytest.raises(ValueError, match="Illegal action 3"):
        env.step(3)
    assert env.game.pending_discover == pending_before
    assert env.game.actions_remaining == 0

    obs, reward, terminated, truncated, info = env.step(34)

    assert not terminated
    assert not truncated
    assert info["forced_end_turn"]
    assert info["turn"] == 1
    assert env.game.turn == 2
    assert env.game.actions_remaining == 1
    assert len(env.game.hand) == 1
    assert obs["player"][-1] == 0


def test_public_golden_observation_and_opponent_state_no_observation_leakage():
    visible = BgEnv(opponents=[{"tier": 1, "board": [LEGACY_TRIPLE_CARD_ID]}])
    hidden_variant = BgEnv(opponents=[{"tier": 1, "board": [FILLER_CARD_ID, FILLER_CARD_ID]}])
    visible_obs, _ = visible.reset(seed=18)
    hidden_obs, _ = hidden_variant.reset(seed=18)

    assert visible_obs.keys() == hidden_obs.keys()
    assert "opponent" not in visible_obs
    assert "opponent_board" not in visible_obs
    assert all(np.array_equal(visible_obs[key], hidden_obs[key]) for key in visible_obs)

    visible.game.board = [visible.game.catalog.spawn(LEGACY_TRIPLE_CARD_ID, 200, golden=True)]
    obs = visible.observation()
    vocab = visible.slot_width - 6
    assert obs["board"][0, visible.game.catalog.mapping[LEGACY_TRIPLE_CARD_ID]] == 1
    assert obs["board"][0, vocab + 2] == 1

    obs, _, _, _, info = visible.step(0)
    assert "opponent_board" in info
    assert "opponent_board" not in obs
