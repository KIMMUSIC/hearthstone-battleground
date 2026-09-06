from dataclasses import FrozenInstanceError

import pytest

import hearthstone_ai.lobby as lobby_module
from hearthstone_ai.actions import Action, encode_action
from hearthstone_ai.combat import CombatResult
from hearthstone_ai.lobby import LobbyGame


TIER_ONE_CARD = 42467
FILLER_CARD = 72387
TIER_TWO_CARD = 2053


def drain_recruit(game: LobbyGame) -> None:
    while game.current_seat is not None and not game.done:
        game.step(game.current_seat, encode_action(Action("end")))


def reserve_spawn(game: LobbyGame, card_id: int, golden: bool = False):
    copies = 3 if golden else 1
    assert game.pool.available[card_id] >= copies
    game.pool.available[card_id] -= copies
    return game._spawn(card_id, golden=golden)


def test_initial_recruit_and_wrong_seat_rejection_are_atomic():
    game = LobbyGame(seed=11, pool_copies=4)
    assert game.round == 1
    assert game.current_seat == 0
    assert game.players[0].gold == 3
    assert len([card for card in game.players[0].shop if card is not None]) == 3
    assert game.legal_actions(1) == []

    before = (
        game.current_seat,
        game.players[0].gold,
        list(game.players[0].shop),
        dict(game.pool.available),
        len(game.trace),
    )
    with pytest.raises(ValueError, match="current seat"):
        game.step(1, encode_action(Action("end")))
    after = (
        game.current_seat,
        game.players[0].gold,
        list(game.players[0].shop),
        dict(game.pool.available),
        len(game.trace),
    )
    assert after == before
    game.assert_conservation()


def test_buy_play_sell_and_pool_conservation():
    game = LobbyGame(seed=12, pool_copies={TIER_ONE_CARD: 4, FILLER_CARD: 4, TIER_TWO_CARD: 4})
    player = game.players[0]
    bought = player.shop[0].card_id

    game.step(0, 4)
    assert player.gold == 0
    assert len(player.hand) == 1
    assert player.shop[0] is None
    game.assert_conservation()

    game.step(0, 18)
    assert len(player.board) == 1
    assert len(player.hand) == 0
    game.assert_conservation()

    game.step(0, 11)
    assert player.gold == 1
    assert player.board == []
    assert game.pool.available[bought] >= 1
    game.assert_conservation()


def test_freeze_preserves_shop_holes_and_upgrade_discount():
    game = LobbyGame(seed=13, pool_copies=5)
    first_shop = [(card.card_id, card.entity_id) if card else None for card in game.players[0].shop]

    game.step(0, encode_action(Action("freeze")))
    game.step(0, 4)
    game.step(0, encode_action(Action("end")))
    while game.current_seat != 0:
        game.step(game.current_seat, encode_action(Action("end")))

    next_shop = [(card.card_id, card.entity_id) if card else None for card in game.players[0].shop]
    assert next_shop[0] is None
    assert next_shop[1:] == first_shop[1:]
    assert game.players[0].gold == 4
    assert game.players[0].upgrade_cost == 4

    game.players[0].gold = 4
    game.step(0, encode_action(Action("upgrade")))
    assert game.players[0].tier == 2
    assert game.players[0].upgrade_cost == 0
    assert encode_action(Action("upgrade")) not in game.legal_actions(0)
    game.assert_conservation()


def test_triple_golden_discover_modal_reserves_and_returns_candidates():
    game = LobbyGame(
        seed=14,
        pool_copies={TIER_ONE_CARD: 6, FILLER_CARD: 6, TIER_TWO_CARD: 3},
    )
    player = game.players[0]
    for minion in player.shop:
        if minion is not None:
            game.pool.return_cards([minion.card_id])
    player.shop = [reserve_spawn(game, TIER_ONE_CARD), None, None, None, None, None, None]
    player.hand = [reserve_spawn(game, TIER_ONE_CARD), reserve_spawn(game, TIER_ONE_CARD)]
    player.hand[0].attack += 2
    player.gold = 3

    game.step(0, 4)
    assert len(player.hand) == 1
    assert player.hand[0].golden
    assert player.hand[0].attack == game.catalog.by_id[TIER_ONE_CARD].attack * 2 + 2
    game.step(0, 18)
    assert player.pending_discover
    assert game.legal_actions(0) == [34, 35, 36]
    before_choice = dict(game.pool.available)

    chosen = player.pending_discover[1].card_id
    returned = [card.card_id for index, card in enumerate(player.pending_discover) if index != 1]
    game.step(0, 35)

    assert [card.card_id for card in player.hand] == [chosen]
    for card_id in returned:
        assert game.pool.available[card_id] == before_choice[card_id] + returned.count(card_id)
    assert player.pending_discover == []
    game.assert_conservation()


def test_zero_candidate_discover_does_not_trap_player():
    game = LobbyGame(seed=141, pool_copies={TIER_ONE_CARD: 6, FILLER_CARD: 6})
    player = game.players[0]
    for minion in player.shop:
        if minion is not None:
            game.pool.return_cards([minion.card_id])
    player.shop = [None] * 7
    player.hand = [reserve_spawn(game, TIER_ONE_CARD, golden=True)]

    result = game.step(0, 18)

    assert any(event["event"] == "discover_exhausted" for event in result["events"])
    assert player.pending_discover == []
    assert game.legal_actions(0) != [34, 35, 36]
    game.assert_conservation()


def test_last_action_discover_modal_resolves_before_forced_recruit_end():
    game = LobbyGame(seed=142, pool_copies={TIER_ONE_CARD: 6, FILLER_CARD: 6, TIER_TWO_CARD: 3})
    player = game.players[0]
    for minion in player.shop:
        if minion is not None:
            game.pool.return_cards([minion.card_id])
    player.shop = [None] * 7
    player.hand = [reserve_spawn(game, TIER_ONE_CARD, golden=True)]
    player.actions_remaining = 1

    result = game.step(0, 18)

    assert not result["terminated"]
    assert game.current_seat == 0
    assert player.actions_remaining == 0
    assert game.legal_actions(0) == [34, 35, 36]
    result = game.step(0, 34)
    assert game.current_seat == 1
    assert any(event["event"] == "recruit_end" and event["forced"] for event in result["events"])
    assert any(event["event"] == "recruit_start" and event["seat"] == 1 for event in result["events"])
    assert all("round" in event for event in result["events"])
    game.assert_conservation()


def test_selling_golden_returns_three_base_copies():
    game = LobbyGame(seed=143, pool_copies={TIER_ONE_CARD: 6, FILLER_CARD: 6, TIER_TWO_CARD: 3})
    player = game.players[0]
    player.board = [reserve_spawn(game, TIER_ONE_CARD, golden=True)]
    before = game.pool.available[TIER_ONE_CARD]

    game.step(0, 11)

    assert game.pool.available[TIER_ONE_CARD] == before + 3
    assert player.gold == 4
    game.assert_conservation()


def test_full_hand_and_full_board_rejections_are_atomic():
    game = LobbyGame(seed=144, pool_copies={TIER_ONE_CARD: 30, FILLER_CARD: 30, TIER_TWO_CARD: 3})
    player = game.players[0]
    for minion in player.shop:
        if minion is not None:
            game.pool.return_cards([minion.card_id])
    player.shop = [reserve_spawn(game, TIER_ONE_CARD), None, None, None, None, None, None]
    player.hand = [reserve_spawn(game, FILLER_CARD) for _ in range(10)]
    player.gold = 3
    before = (player.gold, list(player.shop), list(player.hand), dict(game.pool.available))

    assert 4 not in game.legal_actions(0)
    with pytest.raises(ValueError, match="Illegal action 4"):
        game.step(0, 4)
    assert (player.gold, list(player.shop), list(player.hand), dict(game.pool.available)) == before

    game = LobbyGame(seed=145, pool_copies={TIER_ONE_CARD: 30, FILLER_CARD: 30, TIER_TWO_CARD: 3})
    player = game.players[0]
    for minion in player.shop:
        if minion is not None:
            game.pool.return_cards([minion.card_id])
    player.shop = [None] * 7
    player.hand = [reserve_spawn(game, TIER_ONE_CARD)]
    player.board = [reserve_spawn(game, FILLER_CARD) for _ in range(7)]
    before = (list(player.hand), list(player.board), dict(game.pool.available))
    assert 18 not in game.legal_actions(0)
    with pytest.raises(ValueError, match="Illegal action 18"):
        game.step(0, 18)
    assert (list(player.hand), list(player.board), dict(game.pool.available)) == before
    game.assert_conservation()


def test_policy_view_is_immutable_and_does_not_expose_hidden_state():
    game = LobbyGame(seed=15)
    view = game.view(0)

    assert view.seat == 0
    assert len(view.shop) == 7
    assert len(view.public_players) == 8
    assert view.cards == tuple(game.catalog.cards)
    assert not hasattr(view, "pool")
    assert not hasattr(view, "trace")
    assert not hasattr(next(card for card in view.shop if card is not None), "entity_id")
    assert view.public_players[1].alive
    with pytest.raises(FrozenInstanceError):
        view.gold = 99

    first_shop_card = next(card for card in view.shop if card is not None)
    game.players[0].shop[0].attack += 100
    assert first_shop_card.attack != game.players[0].shop[0].attack


def test_policy_view_does_not_leak_prior_seat_spawn_count():
    end = encode_action(Action("end"))
    reroll = encode_action(Action("reroll"))
    quiet = LobbyGame(seed=101, pool_copies=100)
    noisy = LobbyGame(seed=101, pool_copies=100)

    quiet.step(0, end)
    noisy.step(0, reroll)
    noisy.step(0, end)

    assert [card.card_id if card else None for card in quiet.players[1].shop] == [
        card.card_id if card else None for card in noisy.players[1].shop
    ]
    assert [card.entity_id if card else None for card in quiet.players[1].shop] != [
        card.entity_id if card else None for card in noisy.players[1].shop
    ]
    assert quiet.view(1) == noisy.view(1)


def test_full_lobby_terminates_and_records_matching_trace(monkeypatch):
    def decisive_simulate(player, enemy, rng, catalog, player_tier=1, enemy_tier=1):
        del player, enemy, rng, catalog, player_tier, enemy_tier
        return CombatResult(
            result=1,
            damage_taken=0,
            damage_dealt=40,
            attacks=(),
            reason="test",
        )

    monkeypatch.setattr(lobby_module, "simulate", decisive_simulate)
    game = LobbyGame(seed=16, max_rounds=5)

    drain_recruit(game)

    assert game.terminated
    assert not game.truncated
    assert game.done
    assert [player.rank for player in game.players].count(1.0) == 1
    assert any(event["event"] == "matching" for event in game.trace)
    assert any(event["event"] == "combat" for event in game.trace)
    assert any(event["event"] == "eliminations" for event in game.trace)
    game.assert_conservation()


def test_tied_elimination_and_ghost_assignment(monkeypatch):
    calls = []

    def scripted_simulate(player, enemy, rng, catalog, player_tier=1, enemy_tier=1):
        del player, enemy, rng, catalog, player_tier, enemy_tier
        calls.append(1)
        if len(calls) <= 4:
            return CombatResult(
                result=0,
                damage_taken=40,
                damage_dealt=40,
                attacks=(),
                reason="tie",
            )
        return CombatResult(
            result=1,
            damage_taken=0,
            damage_dealt=0,
            attacks=(),
            reason="ghost-control",
        )

    monkeypatch.setattr(lobby_module, "simulate", scripted_simulate)
    game = LobbyGame(seed=17, max_rounds=2)
    for seat in range(PLAYER_COUNT := 8):
        game.players[seat].board = [reserve_spawn(game, TIER_ONE_CARD)]

    drain_recruit(game)

    assert game.terminated
    assert all(player.rank == 4.5 for player in game.players)
    assert game._ghost_source_seat == 0
    assert game.current_seat is None
    assert PLAYER_COUNT == 8
    game.assert_conservation()


def test_odd_survivor_round_uses_latest_ghost_snapshot(monkeypatch):
    round_matchings = [
        (((0, 1), (2, 3), (4, 5), (6, 7)), (0, 0, 0)),
        (((1, None), (2, 3), (4, 5), (6, 7)), (0, 0, 0)),
    ]
    combat_calls = []

    def scripted_matching(alive, history, ghost_counts, last_ghost, rng):
        del alive, history, ghost_counts, last_ghost, rng
        return round_matchings[len(combat_calls) // 4]

    def scripted_simulate(player, enemy, rng, catalog, player_tier=1, enemy_tier=1):
        del player, rng, catalog, player_tier, enemy_tier
        call_index = len(combat_calls)
        combat_calls.append([minion.card_id for minion in enemy])
        if call_index == 0:
            return CombatResult(
                result=-1,
                damage_taken=40,
                damage_dealt=0,
                attacks=(),
                reason="single-elimination",
            )
        return CombatResult(result=0, damage_taken=0, damage_dealt=0, attacks=(), reason="tie")

    monkeypatch.setattr(lobby_module, "match_players", scripted_matching)
    monkeypatch.setattr(lobby_module, "simulate", scripted_simulate)
    game = LobbyGame(seed=171, max_rounds=2)
    game.players[0].board = [reserve_spawn(game, TIER_ONE_CARD)]

    drain_recruit(game)

    assert game.truncated
    assert game._ghost_source_seat == 0
    assert game._ghost_counts[1] == 1
    assert combat_calls[4] == [TIER_ONE_CARD]
    ghost_combat = [event for event in game.trace if event["event"] == "combat" and event["right"] is None]
    assert ghost_combat[-1]["enemy_seat"] == 0
    game.assert_conservation()


def test_even_round_clears_consecutive_ghost_state(monkeypatch):
    def tie_simulate(player, enemy, rng, catalog, player_tier=1, enemy_tier=1):
        del player, enemy, rng, catalog, player_tier, enemy_tier
        return CombatResult(result=0, damage_taken=0, damage_dealt=0, attacks=(), reason="tie")

    monkeypatch.setattr(lobby_module, "simulate", tie_simulate)
    game = LobbyGame(seed=172, max_rounds=2)
    game._last_ghost = 3

    for _ in range(8):
        game.step(game.current_seat, encode_action(Action("end")))

    assert game.current_seat == 1
    assert game._last_ghost is None
    assert game._ghost_counts == {}
    game.assert_conservation()


def test_combat_failure_does_not_commit_matching_or_damage(monkeypatch):
    def broken_simulate(player, enemy, rng, catalog, player_tier=1, enemy_tier=1):
        del player, enemy, rng, catalog, player_tier, enemy_tier
        raise RuntimeError("combat exploded")

    def tie_simulate(player, enemy, rng, catalog, player_tier=1, enemy_tier=1):
        del player, enemy, rng, catalog, player_tier, enemy_tier
        return CombatResult(result=0, damage_taken=0, damage_dealt=0, attacks=(), reason="tie")

    monkeypatch.setattr(lobby_module, "simulate", broken_simulate)
    game = LobbyGame(seed=173, max_rounds=2)
    end = encode_action(Action("end"))
    for _ in range(7):
        game.step(game.current_seat, end)

    before = {
        "current_seat": game.current_seat,
        "ready": [player.ready for player in game.players],
        "hp": [player.hp for player in game.players],
        "rank": [player.rank for player in game.players],
        "history": list(game._matching_history),
        "ghost_counts": dict(game._ghost_counts),
        "last_ghost": game._last_ghost,
        "trace": list(game.trace),
        "match_rng": game._match_rng.getstate(),
        "combat_rng": game._combat_rng.getstate(),
    }

    with pytest.raises(RuntimeError, match="combat exploded"):
        game.step(game.current_seat, end)

    assert game.current_seat == before["current_seat"]
    assert [player.ready for player in game.players] == before["ready"]
    assert [player.hp for player in game.players] == before["hp"]
    assert [player.rank for player in game.players] == before["rank"]
    assert game._matching_history == before["history"]
    assert game._ghost_counts == before["ghost_counts"]
    assert game._last_ghost == before["last_ghost"]
    assert game.trace == before["trace"]
    assert game._match_rng.getstate() == before["match_rng"]
    assert game._combat_rng.getstate() == before["combat_rng"]

    monkeypatch.setattr(lobby_module, "simulate", tie_simulate)
    result = game.step(game.current_seat, end)
    assert any(event["event"] == "matching" for event in result["events"])
    assert game.round == 2
    game.assert_conservation()


def test_forced_combat_failure_keeps_action_and_allows_end_retry(monkeypatch):
    def broken_simulate(player, enemy, rng, catalog, player_tier=1, enemy_tier=1):
        del player, enemy, rng, catalog, player_tier, enemy_tier
        raise RuntimeError("combat exploded")

    def tie_simulate(player, enemy, rng, catalog, player_tier=1, enemy_tier=1):
        del player, enemy, rng, catalog, player_tier, enemy_tier
        return CombatResult(result=0, damage_taken=0, damage_dealt=0, attacks=(), reason="tie")

    monkeypatch.setattr(lobby_module, "simulate", broken_simulate)
    game = LobbyGame(seed=174, max_rounds=2)
    end = encode_action(Action("end"))
    for _ in range(7):
        game.step(game.current_seat, end)
    game.players[7].actions_remaining = 1
    before_trace = list(game.trace)

    with pytest.raises(RuntimeError, match="combat exploded"):
        game.step(7, encode_action(Action("freeze")))

    assert game.current_seat == 7
    assert game.players[7].frozen
    assert game.players[7].actions_remaining == 0
    assert game.legal_actions(7) == [end]
    assert game.trace == before_trace

    monkeypatch.setattr(lobby_module, "simulate", tie_simulate)
    game.step(7, end)
    assert game.round == 2
    game.assert_conservation()


def test_round_limit_truncates_without_assigning_survivor_ranks(monkeypatch):
    def tie_simulate(player, enemy, rng, catalog, player_tier=1, enemy_tier=1):
        del player, enemy, rng, catalog, player_tier, enemy_tier
        return CombatResult(result=0, damage_taken=0, damage_dealt=0, attacks=(), reason="tie")

    monkeypatch.setattr(lobby_module, "simulate", tie_simulate)
    game = LobbyGame(seed=18, max_rounds=1)

    drain_recruit(game)

    assert not game.terminated
    assert game.truncated
    assert game.done
    assert all(player.rank is None for player in game.players)
    with pytest.raises(RuntimeError, match="complete"):
        game.step(0, 0)
    game.assert_conservation()
