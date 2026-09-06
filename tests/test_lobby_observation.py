"""Numeric contract is independent of the engine's private mutable state."""

from types import SimpleNamespace

import numpy as np
import pytest

from hearthstone_ai.cards import Catalog
from hearthstone_ai.lobby_observation import LobbyObservation
from hearthstone_ai.lobby_policies import heuristic_policy


def view():
    return SimpleNamespace(seat=0, round=1, hp=30, gold=3, tier=1, upgrade_cost=5,
                           frozen=False, actions_remaining=24, swaps_remaining=3,
                           shop=(None,) * 7, board=(), hand=(), pending_discover=(),
                           public_players=tuple(SimpleNamespace(seat=i, hp=30, tier=1, alive=True, rank=None)
                                                for i in range(8)), legal_actions=(0, 3), cards=Catalog().cards)


def test_signed_hp_legal_mask_and_all_space_fields():
    encoder = LobbyObservation()
    state = view()
    state.hp = -123
    state.public_players[0].hp = -123
    state.public_players[0].alive = False
    state.public_players[0].rank = 7.5
    obs = encoder.encode(state)
    assert encoder.space.contains(obs)
    assert obs["player"][2] == -123 and obs["public_players"][0, 3] == 7.5
    assert np.flatnonzero(obs["action_mask"]).tolist() == [0, 3]
    assert obs["shop"].shape == (7, 18)
    assert np.all(obs["shop"][:, 0] == 1)


def test_golden_stats_are_distinct_unclipped_and_return_arrays_are_copies():
    encoder = LobbyObservation()
    state = view()
    card = Catalog().spawn(42467, 1, golden=True)
    card.attack = 100000
    state.board = (card,)
    obs = encoder.encode(state)
    assert obs["board"][0, -6] == 1000 and obs["board"][0, -4] == 1
    obs["board"][:] = 0
    assert encoder.encode(state)["board"][0, -6] == 1000
    assert card.attack == 100000


@pytest.mark.parametrize("fault", ["unknown", "capacity", "nonfinite", "action", "seats"])
def test_invalid_views_are_rejected(fault):
    state = view()
    if fault == "unknown":
        card = Catalog().spawn(42467, 1)
        card.card_id = 49278  # Tombstoned identity is not an active definition.
        state.board = (card,)
    elif fault == "capacity":
        state.hand = (None,) * 11
    elif fault == "nonfinite":
        state.hp = float("nan")
    elif fault == "action":
        state.legal_actions = (37,)
    else:
        state.public_players = state.public_players[:-1]
    with pytest.raises((KeyError, ValueError)):
        LobbyObservation().encode(state)


def test_heuristic_only_uses_supplied_legal_view():
    state = view()
    assert heuristic_policy(state) == 0
    state.pending_discover = (Catalog().spawn(42467, 1), Catalog().spawn(2053, 2))
    state.legal_actions = (34, 35)
    assert heuristic_policy(state) in state.legal_actions
