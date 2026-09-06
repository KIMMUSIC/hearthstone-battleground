"""Combat snapshots, ghost reuse, and token/pool conservation across a round."""

from copy import deepcopy

import hearthstone_ai.lobby as module
from hearthstone_ai.combat import CombatResult
from hearthstone_ai.lobby import LobbyGame


def reserve(game, seat, card_id):
    game.pool.available[card_id] -= 1
    card = game._spawn(card_id)
    game.players[seat].board.append(card)
    return card


def test_real_deathrattle_is_transient_and_recruit_boards_survive_combat(monkeypatch):
    monkeypatch.setattr(module, "match_players", lambda *args: (((0, 1), (2, 3), (4, 5), (6, 7)), (0, 0, 0)))
    game = LobbyGame(seed=7, max_rounds=1)
    reserve(game, 0, 96763)
    strong = reserve(game, 1, 42467)
    strong.attack = strong.health = 100
    boards = [deepcopy(p.board) for p in game.players]
    for seat in range(7):
        game.step(seat, 0)
    before = dict(game.pool.available)
    game.step(7, 0)
    assert game.truncated
    assert game.pool.available == before
    assert [p.board for p in game.players] == boards
    assert any(event.get("summons") for event in game.trace if event["event"] == "combat")
    assert all(m.card_id != 96764 for p in game.players for m in p.board + p.hand)
    game.assert_conservation()


def test_odd_round_uses_eliminated_precombat_board_and_only_alive_side_takes_damage(monkeypatch):
    monkeypatch.setattr(module, "match_players", lambda alive, *args: (
        ((0, 1), (2, 3), (4, 5), (6, 7)) if len(alive) == 8
        else ((0, None), (2, 3), (4, 5), (6, 7)), (0, 0, 0)))
    observed = []

    def combat(player, enemy, rng, catalog, player_tier=1, enemy_tier=1):
        observed.append(deepcopy(enemy))
        if len(observed) == 1:
            return CombatResult(1, 0, 40, (), "scripted death")
        if len(observed) == 5:
            return CombatResult(-1, 2, 999, (), "scripted ghost")
        return CombatResult(0, 0, 0, (), "scripted draw")

    monkeypatch.setattr(module, "simulate", combat)
    game = LobbyGame(seed=17, max_rounds=2)
    ghost_card = reserve(game, 1, 42467)
    ghost_card.attack += 7
    expected = deepcopy(game.players[1].board)
    while not game.done:
        game.step(game.current_seat, 0)
    assert observed[4] == expected
    assert not game.players[1].alive and game.players[1].rank == 8
    assert game.players[1].hp == -10
    assert game.players[0].hp == 28
    assert game.players[1].board == []
    assert game.legal_actions(1) == []
    game.assert_conservation()
