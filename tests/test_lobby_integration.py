"""Cross-layer eight-policy games, observation privacy, and RNG isolation."""

from dataclasses import FrozenInstanceError
import random

import numpy as np
import pytest

from hearthstone_ai.lobby import LobbyGame
from hearthstone_ai.lobby_observation import LobbyObservation
from hearthstone_ai.lobby_policies import heuristic_policy, random_policy


def advance(game, rngs, encoder):
    seat = game.current_seat
    view = game.view(seat)
    observation = encoder.encode(view)
    assert encoder.space.contains(observation)
    assert np.flatnonzero(observation["action_mask"]).tolist() == game.legal_actions(seat)
    action = heuristic_policy(view) if seat % 2 == 0 else random_policy(view, rngs[seat])
    assert action in view.legal_actions
    game.step(seat, action)
    game.assert_conservation()


def play(seed):
    game = LobbyGame(seed=seed, max_rounds=50)
    rngs = [random.Random(f"policy:{seed}:{seat}") for seat in range(8)]
    encoder = LobbyObservation()
    while not game.done:
        advance(game, rngs, encoder)
    for seat in range(8):
        assert encoder.space.contains(encoder.encode(game.view(seat)))
        assert not game.legal_actions(seat)
    if game.terminated:
        assert sum(p.rank for p in game.players) == 36
    else:
        assert game.truncated and all(p.rank is None for p in game.players if p.alive)
    return game


@pytest.mark.parametrize("seed", [0, 1, 7, 17, 27])
def test_real_eight_policy_games_finish_with_conservation(seed):
    game = play(seed)
    assert game.terminated and not game.truncated
    with pytest.raises(RuntimeError):
        game.step(0, 0)


def test_interleaved_sessions_match_separate_full_traces():
    expected = [play(seed).trace for seed in (4, 8)]
    games = [LobbyGame(seed=seed, max_rounds=50) for seed in (4, 8)]
    rngs = [[random.Random(f"policy:{seed}:{seat}") for seat in range(8)] for seed in (4, 8)]
    encoder = LobbyObservation()
    while not all(game.done for game in games):
        for game, policies in zip(games, rngs, strict=True):
            if not game.done:
                advance(game, policies, encoder)
    assert [game.trace for game in games] == expected


def test_private_opponent_state_cannot_change_policy_view_or_observation():
    game = LobbyGame(seed=7)
    game.step(0, 0)
    assert game.current_seat == 1
    before = game.view(1)
    encoder = LobbyObservation()
    encoded = encoder.encode(before)
    hidden = next(card for card in game.players[0].shop if card is not None)
    hidden.attack += 100
    after = game.view(1)
    assert after == before
    assert all(np.array_equal(encoded[key], encoder.encode(after)[key]) for key in encoded)
    assert not hasattr(after, "pool") and not hasattr(after, "players")
    own = next(card for card in after.shop if card is not None)
    with pytest.raises(FrozenInstanceError):
        own.attack = 999
    with pytest.raises(FrozenInstanceError):
        after.gold = 999
    encoded["shop"][:] = 99
    assert game.view(1) == after


def test_round_limit_has_no_fabricated_winner():
    game = LobbyGame(seed=0, max_rounds=1)
    for seat in range(8):
        assert game.current_seat == seat
        game.step(seat, 0)
    assert game.truncated and not game.terminated
    assert all(p.rank is None for p in game.players)
