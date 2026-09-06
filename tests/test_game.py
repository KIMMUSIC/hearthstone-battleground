import copy
import json
import os
import subprocess
import sys
import zipfile
import unittest

import pytest

from hearthstone_ai.artifacts import ROOT
from hearthstone_ai.cards import Card
import hearthstone_ai.game as game_module
from hearthstone_ai.game import DIVERSE_OPPONENT_CARD_IDS, Game, opponent_distribution_spec

SNAPSHOT_003 = ROOT / "handoff/snapshot-003/HearthStoneAI-Rebuild-003.zip"

TRACE_PROGRAM = r"""
import json
import os
from pathlib import Path
import random
from hearthstone_ai.cards import Catalog
from hearthstone_ai.game import Game

DATA_DIR = os.environ.get("HSAI_TRACE_DATA_DIR")

def snap(game):
    return {
        "hp": game.hp,
        "gold": game.gold,
        "tier": game.tier,
        "turn": game.turn,
        "upgrade_cost": game.upgrade_cost,
        "frozen": game.frozen,
        "actions_remaining": game.actions_remaining,
        "swaps_remaining": game.swaps_remaining,
        "shop": [None if card is None else [card.card_id, card.entity_id] for card in game.shop],
        "board": [[m.card_id, m.entity_id, m.attack, m.health, m.golden] for m in game.board],
        "hand": [[m.card_id, m.entity_id, m.attack, m.health, m.golden] for m in game.hand],
        "legal": game.legal_actions() if not game.done else [],
        "done": game.done,
        "termination_reason": game.termination_reason,
    }

def common_info(info):
    keys = [
        "combat_result",
        "damage_taken",
        "damage_dealt",
        "attacks",
        "combat_reason",
        "forced_end_turn",
        "turn",
        "termination_reason",
    ]
    return {key: info[key] for key in keys if key in info}

def episode(seed):
    catalog = Catalog(Path(DATA_DIR)) if DATA_DIR else None
    game = Game(catalog=catalog, max_turns=8, max_actions=24)
    game.reset(seed)
    rng = random.Random(f"old-trace-actions:{seed}")
    trace = [{"state": snap(game)}]
    for _ in range(8 * 26):
        action = rng.choice(game.legal_actions())
        result = game.step(action)
        trace.append(
            {
                "action": action,
                "reward": result.reward,
                "terminated": result.terminated,
                "truncated": result.truncated,
                "info": common_info(result.info),
                "state": snap(game),
            }
        )
        if result.terminated or result.truncated:
            break
    else:
        raise RuntimeError("episode did not finish")
    return trace

print(json.dumps({str(seed): episode(seed) for seed in [7, 17, 27]}, sort_keys=True))
"""


class CatalogLike:
    def __init__(self, cards):
        self.cards = tuple(cards)
        self.by_id = {card.dbf_id: card for card in cards}
        self.mapping = {card.dbf_id: index + 1 for index, card in enumerate(cards)}


class GameTests(unittest.TestCase):
    def game(self, **kwargs):
        game = Game(**kwargs)
        game.reset(17)
        return game

    def test_invalid_padding_and_terminal_are_explicit(self):
        game = self.game(max_turns=1)
        snapshot = copy.deepcopy(game.__dict__)
        with self.assertRaises(ValueError):
            game.step(10)
        self.assertEqual(game.shop, snapshot["shop"])
        self.assertEqual(game.gold, snapshot["gold"])
        self.assertEqual(game.actions_remaining, snapshot["actions_remaining"])
        self.assertEqual(game._shop_rng.getstate(), snapshot["_shop_rng"].getstate())
        self.assertEqual(game._combat_rng.getstate(), snapshot["_combat_rng"].getstate())
        result = game.step(0)
        self.assertTrue(result.terminated)
        self.assertFalse(result.truncated)
        self.assertEqual(game.turn, 1)
        with self.assertRaises(RuntimeError):
            game.step(0)

    def test_hand_capacity_and_modal_discover(self):
        game = self.game()
        card_id = game.catalog.cards[0].dbf_id
        game.hand = [game.catalog.spawn(card_id, i, golden=True) for i in range(10)]
        self.assertNotIn(4, game.legal_actions())
        game.step(18)
        self.assertEqual(len(game.pending_discover), 3)
        self.assertEqual(game.legal_actions(), [34, 35, 36])
        self.assertTrue(all(card.tier == 2 for card in game.pending_discover))
        with self.assertRaises(ValueError):
            game.step(28)
        game.step(34)
        self.assertEqual(len(game.hand), 10)

    def test_triple_preserves_buffs_and_frees_capacity(self):
        game = self.game()
        card = game.catalog.cards[0]
        game.board = [game.catalog.spawn(card.dbf_id, 100)]
        game.hand = [game.catalog.spawn(card.dbf_id, 101)]
        game.board[0].attack += 3
        game.shop[0] = game.catalog.spawn(card.dbf_id, 102)
        game.step(4)
        self.assertEqual(len(game.board), 0)
        self.assertEqual(len(game.hand), 1)
        self.assertTrue(game.hand[0].golden)
        self.assertEqual(game.hand[0].attack, 2 * card.attack + 3)

    def test_golem_is_not_triple_converted_until_golden_effect_is_supported(self):
        game = self.game()
        game.board = [game.catalog.spawn(96763, 100)]
        game.hand = [game.catalog.spawn(96763, 101), game.catalog.spawn(96763, 102)]
        game._triples()
        self.assertEqual([m.card_id for m in game.board + game.hand], [96763, 96763, 96763])
        self.assertTrue(all(not m.golden for m in game.board + game.hand))

    def test_same_seed_and_session_isolation(self):
        first, second = self.game(), self.game()
        unrelated = self.game()
        for action in [4, 18, 0, 1, 0]:
            unrelated.reset(999)
            self.assertEqual(first.step(action), second.step(action))
            self.assertEqual(first.shop, second.shop)
            self.assertEqual(first.board, second.board)

    def test_action_limit_forces_combat_and_resets(self):
        game = self.game(max_actions=1)
        result = game.step(3)
        self.assertTrue(result.info["forced_end_turn"])
        self.assertEqual(game.turn, 2)
        self.assertEqual(game.actions_remaining, 1)

    def test_last_action_discover_resolves_before_forced_combat(self):
        game = self.game(max_actions=1)
        game.hand = [game.catalog.spawn(game.catalog.cards[0].dbf_id, 100, golden=True)]
        result = game.step(18)
        self.assertFalse(result.terminated)
        self.assertEqual(game.actions_remaining, 0)
        self.assertEqual(game.legal_actions(), [34, 35, 36])
        result = game.step(34)
        self.assertTrue(result.info["forced_end_turn"])
        self.assertEqual(game.turn, 2)
        self.assertEqual(len(game.hand), 1)

    def test_final_turn_is_fought_and_death_is_separate(self):
        game = self.game(max_turns=3, opponents=[{"tier": 1, "board": []}])
        for turn in range(1, 4):
            result = game.step(0)
            self.assertEqual(result.info["turn"], turn)
            self.assertEqual(result.terminated, turn == 3)
        self.assertEqual(game.termination_reason, "horizon")
        game = self.game()
        game.hp = 1
        result = game.step(0)
        self.assertTrue(result.terminated)
        self.assertEqual(game.termination_reason, "death")

    def test_freeze_preserves_shop_once(self):
        game = self.game()
        shop = copy.deepcopy(game.shop)
        game.step(3)
        game.step(0)
        self.assertEqual(game.shop, shop)
        self.assertFalse(game.frozen)
        game.step(0)
        self.assertNotEqual(game.shop, shop)

    def test_upgrade_discount_and_no_shaping_reward(self):
        game = self.game()
        game.step(0)
        self.assertEqual(game.upgrade_cost, 4)
        result = game.step(2)
        self.assertEqual(result.reward, 0.0)
        self.assertEqual(game.tier, 2)
        self.assertNotIn(2, game.legal_actions())

    def test_opponent_templates_are_not_mutated(self):
        game = self.game()
        templates = [{"tier": 1, "board": [game.catalog.cards[0].dbf_id]}]
        original = copy.deepcopy(templates)
        game = self.game(opponents=templates)
        game.step(0)
        self.assertEqual(templates, original)

    def test_swap_budget_and_board_capacity(self):
        game = self.game()
        card_id = game.catalog.cards[0].dbf_id
        game.board = [game.catalog.spawn(card_id, i, golden=True) for i in range(7)]
        game.hand = [game.catalog.spawn(card_id, 100)]
        self.assertNotIn(18, game.legal_actions())
        for _ in range(3):
            game.step(28)
        self.assertNotIn(28, game.legal_actions())


def trace_fixed_game():
    game = Game(max_turns=3, max_actions=4)
    game.reset(17)
    trace = [
        (
            game.hp,
            game.gold,
            game.turn,
            [(card.card_id, card.entity_id) if card is not None else None for card in game.shop],
            list(game.legal_actions()),
        )
    ]
    for action in [4, 18, 0, 1, 0]:
        result = game.step(action)
        trace.append(
            (
                action,
                result.reward,
                result.terminated,
                result.info.get("combat_result"),
                result.info.get("opponent_board"),
                game.hp,
                game.gold,
                game.turn,
                [(card.card_id, card.entity_id) if card is not None else None for card in game.shop],
                [(minion.card_id, minion.entity_id) for minion in game.board],
                list(game.legal_actions()) if not game.done else [],
            )
        )
    return trace


def test_fixed_opponent_mode_preserves_existing_trace():
    assert trace_fixed_game() == [
        (
            30,
            3,
            1,
            [(60628, 1), (60628, 2), (72387, 3), None, None, None, None],
            [0, 1, 3, 4, 5, 6],
        ),
        (
            4,
            0.0,
            False,
            None,
            None,
            30,
            0,
            1,
            [None, (60628, 2), (72387, 3), None, None, None, None],
            [],
            [0, 3, 18],
        ),
        (
            18,
            0.0,
            False,
            None,
            None,
            30,
            0,
            1,
            [None, (60628, 2), (72387, 3), None, None, None, None],
            [(60628, 1)],
            [0, 3, 11],
        ),
        (
            0,
            0.0,
            False,
            0,
            [72387],
            30,
            4,
            2,
            [(60628, 4), (96796, 5), (42467, 6), None, None, None, None],
            [(60628, 1)],
            [0, 1, 2, 3, 4, 5, 6, 11],
        ),
        (
            1,
            0.0,
            False,
            None,
            None,
            30,
            3,
            2,
            [(42467, 7), (60628, 8), (96796, 9), None, None, None, None],
            [(60628, 1)],
            [0, 1, 3, 4, 5, 6, 11],
        ),
        (
            0,
            0.0,
            False,
            0,
            [72387],
            30,
            5,
            3,
            [(72387, 10), (60628, 11), (72387, 12), None, None, None, None],
            [(60628, 1)],
            [0, 1, 2, 3, 4, 5, 6, 11],
        ),
    ]


def test_fixed_mode_matches_snapshot003_for_three_full_seeded_episodes(tmp_path):
    extract_dir = tmp_path / "snapshot003"
    with zipfile.ZipFile(SNAPSHOT_003) as archive:
        archive.extractall(extract_dir)
    old_root = extract_dir
    if not (old_root / "src").exists():
        old_root = next(path for path in extract_dir.iterdir() if (path / "src").exists())
    env = os.environ.copy()
    env["PYTHONPATH"] = str(old_root / "src")
    old = subprocess.run(
        [sys.executable, "-B", "-c", TRACE_PROGRAM],
        check=True,
        capture_output=True,
        text=True,
        cwd=old_root,
        env=env,
        timeout=30,
    )
    current_env = os.environ.copy()
    current_env["PYTHONPATH"] = str(ROOT / "src")
    current_env["HSAI_TRACE_DATA_DIR"] = str(old_root / "data")
    current = subprocess.run(
        [sys.executable, "-B", "-c", TRACE_PROGRAM],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=current_env,
        timeout=30,
    )
    assert json.loads(current.stdout) == json.loads(old.stdout)


def test_diverse_opponent_mode_is_seeded_and_session_independent():
    first = Game(opponent_mode="diverse-v1")
    second = Game(opponent_mode="diverse-v1")
    first.reset(123)
    second.reset(123)
    assert first._opponents == second._opponents
    assert first._opponents != Game()._opponents
    for _, board in first._opponents:
        assert set(board) <= set(DIVERSE_OPPONENT_CARD_IDS)
    unrelated = Game(opponent_mode="diverse-v1")
    unrelated.reset(999)
    assert first._opponents == second._opponents
    assert first._opponents != unrelated._opponents


def test_diverse_opponent_schedule_has_exact_turn_sizes_and_order():
    import random

    seed = 2468
    game = Game(opponent_mode="diverse-v1")
    game.reset(seed)
    rng = random.Random(f"opponent:{seed}")
    expected = tuple(
        (1, tuple(rng.choice(DIVERSE_OPPONENT_CARD_IDS) for _ in range(count)))
        for count in (1, 1, 2, 2, 3, 3, 4, 4)
    )
    assert game._opponents == expected
    assert [len(board) for _, board in game._opponents] == [1, 1, 2, 2, 3, 3, 4, 4]
    for expected_turn, (_, board) in enumerate(expected, start=1):
        result = game.step(0)
        assert result.info["turn"] == expected_turn
        assert result.info["opponent_board"] == list(board)


def test_diverse_sessions_remain_independent_across_interleaved_steps_and_resets():
    first = Game(opponent_mode="diverse-v1")
    second = Game(opponent_mode="diverse-v1")
    first.reset(55)
    second.reset(55)
    for action in [4, 18, 0, 3, 0]:
        assert first.step(action) == second.step(action)
        unrelated = Game(opponent_mode="diverse-v1")
        unrelated.reset(56)
        if not unrelated.done:
            unrelated.step(0)
        assert first._opponents == second._opponents
        assert [(card.card_id, card.entity_id) if card else None for card in first.shop] == [
            (card.card_id, card.entity_id) if card else None for card in second.shop
        ]
    first.reset(55)
    second.reset(55)
    assert first._opponents == second._opponents
    assert [(card.card_id, card.entity_id) if card else None for card in first.shop] == [
        (card.card_id, card.entity_id) if card else None for card in second.shop
    ]
    first.reset(57)
    second.reset(57)
    for action in [0, 1, 0]:
        assert first.step(action) == second.step(action)
        first.reset(55)
        second.reset(55)
        assert first._opponents == second._opponents
        assert [(card.card_id, card.entity_id) if card else None for card in first.shop] == [
            (card.card_id, card.entity_id) if card else None for card in second.shop
        ]


def test_diverse_opponent_generation_does_not_consume_shop_or_combat_rng():
    fixed = Game()
    diverse = Game(opponent_mode="diverse-v1")
    fixed.reset(77)
    diverse.reset(77)
    assert fixed._shop_rng.getstate() == diverse._shop_rng.getstate()
    assert fixed._combat_rng.getstate() == diverse._combat_rng.getstate()
    assert [(card.card_id, card.entity_id) if card else None for card in fixed.shop] == [
        (card.card_id, card.entity_id) if card else None for card in diverse.shop
    ]
    shop_state = diverse._shop_rng.getstate()
    combat_state = diverse._combat_rng.getstate()
    entity_id = diverse._entity_id
    diverse._opponents_for_seed(78)
    assert diverse._shop_rng.getstate() == shop_state
    assert diverse._combat_rng.getstate() == combat_state
    assert diverse._entity_id == entity_id


def test_opponent_mode_validation_and_spec():
    with pytest.raises(ValueError, match="opponent_mode"):
        Game(opponent_mode="unknown")
    with pytest.raises(ValueError, match="opponent_mode"):
        Game(opponent_mode=["diverse-v1"])
    with pytest.raises(ValueError, match="Explicit opponents"):
        Game(opponents=[{"tier": 1, "board": [72387]}], opponent_mode="diverse-v1")
    spec = opponent_distribution_spec(mode="diverse-v1")
    assert spec["candidate_card_ids"] == [42467, 72387]
    assert spec["turn_counts"] == [1, 1, 2, 2, 3, 3, 4, 4]

    with pytest.raises(ValueError, match="not in the supported pool"):
        Game(opponents=[{"tier": 1, "board": [96764]}])


def test_generated_opponent_candidate_validation_errors_are_explicit(monkeypatch):
    missing = CatalogLike([Card(42467, "protector", 1, 1, 1, True, True)])
    with pytest.raises(ValueError, match="72387"):
        opponent_distribution_spec(missing, mode="fixed-v1")

    wrong_tier = CatalogLike(
        [
            Card(42467, "protector", 1, 1, 1, True, True),
            Card(72387, "pupbot", 2, 2, 1, False, True),
        ]
    )
    with pytest.raises(ValueError, match="tier one"):
        opponent_distribution_spec(wrong_tier, mode="fixed-v1")

    monkeypatch.setattr(game_module, "DIVERSE_OPPONENT_CARD_IDS", (42467, 999999))
    with pytest.raises(ValueError, match="999999"):
        Game(opponent_mode="diverse-v1")


def test_explicit_opponents_do_not_require_default_fixed_card():
    catalog = CatalogLike([Card(42467, "protector", 1, 1, 1, True, True)])

    def spawn(card_id, entity_id, golden=False):
        return catalog.by_id[card_id]

    catalog.spawn = spawn
    catalog.sample = lambda rng, max_tier, count, exact=False: [catalog.by_id[42467]] * count
    game = Game(catalog=catalog, opponents=[{"tier": 1, "board": [42467]}])
    assert game.opponent_spec["version"] == "explicit-v1"
