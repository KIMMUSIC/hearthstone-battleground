import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

from hearthstone_ai.cards import Catalog, Minion
from hearthstone_ai.combat import simulate


BASE_CARD = {
    "dbf_id": 99159,
    "name": "plain",
    "tier": 2,
    "attack": 1,
    "health": 5,
    "keywords": [],
}
GOLEM_CARD = {
    "dbf_id": 96763,
    "name": "Harvest Golem",
    "tier": 2,
    "attack": 2,
    "health": 3,
    "keywords": ["DEATHRATTLE"],
    "deathrattle": {"type": "summon", "token_id": 96764, "count": 1},
}
TOKEN_CARD = {
    "dbf_id": 96764,
    "name": "Damaged Golem",
    "tier": 1,
    "attack": 2,
    "health": 1,
    "keywords": [],
}


def plain(entity_id, attack=1, health=5, **kwargs):
    return Minion(entity_id, 99159, attack, health, **kwargs)


def golem(entity_id, attack=2, health=3, **kwargs):
    return Minion(entity_id, 96763, attack, health, **kwargs)


def test_registered_golem_summons_registered_token_in_default_catalog():
    catalog = Catalog()
    player = [catalog.spawn(96763, 1)]
    enemy = [catalog.spawn(96763, 2)]
    result = simulate(player, enemy, random.Random(16), catalog)
    assert len(result.summons) == 2
    assert result.result == 0
    assert len(result.deaths) == 4
    assert player[0].health == enemy[0].health == 3


class DeathSummonTests(unittest.TestCase):
    def make_catalog(self, *cards):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name)
        records_by_id = {card["dbf_id"]: card for card in [BASE_CARD, GOLEM_CARD, TOKEN_CARD]}
        for card in cards:
            records_by_id[card["dbf_id"]] = card
        records = list(records_by_id.values())
        (path / "cards.json").write_text(
            json.dumps({"schema_version": 1, "ruleset": "test", "cards": records}),
            encoding="utf-8",
        )
        (path / "card_mapping.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "padding_index": 0,
                    "mapping": {str(card["dbf_id"]): i + 1 for i, card in enumerate(records)},
                }
            ),
            encoding="utf-8",
        )
        return Catalog(path)

    def test_explicit_golem_deathrattle_catalog_is_supported(self):
        catalog = self.make_catalog()
        card = catalog.by_id[96763]
        self.assertEqual((card.deathrattle_token_id, card.deathrattle_count), (96764, 1))

    def test_bad_deathrattle_and_missing_token_are_rejected(self):
        bad_effect = dict(GOLEM_CARD, dbf_id=1, deathrattle={"type": "buff", "attack": 1})
        with self.assertRaisesRegex(ValueError, "Unsupported deathrattle"):
            self.make_catalog(bad_effect)

        bad_shape = dict(GOLEM_CARD, dbf_id=2, deathrattle=["summon"])
        with self.assertRaisesRegex(ValueError, "Unsupported deathrattle"):
            self.make_catalog(bad_shape)

        missing_token = dict(GOLEM_CARD, dbf_id=3, deathrattle={"type": "summon", "token_id": 4, "count": 1})
        with self.assertRaisesRegex(ValueError, "Unknown deathrattle token"):
            self.make_catalog(missing_token)

    def test_unsupported_summon_modifier_is_not_silently_ignored(self):
        effect = dict(GOLEM_CARD["deathrattle"], immediate_attack=True)
        with self.assertRaisesRegex(ValueError, "Unsupported deathrattle"):
            self.make_catalog(dict(GOLEM_CARD, deathrattle=effect))

    def test_simultaneous_golem_deaths_remove_then_summon_tokens_with_new_ids(self):
        catalog = self.make_catalog()
        result = simulate([golem(1, health=2)], [golem(2, health=2)], random.Random(1), catalog)

        first_side = result.attacks[0]["side"]
        other_side = "enemy" if first_side == "player" else "player"
        self.assertEqual([(event["side"], event["entity_id"]) for event in result.deaths[:2]], [(first_side, 1 if first_side == "player" else 2), (other_side, 2 if first_side == "player" else 1)])
        self.assertEqual([(event["side"], event["source"], event["card_id"]) for event in result.summons[:2]], [(first_side, 1 if first_side == "player" else 2, 96764), (other_side, 2 if first_side == "player" else 1, 96764)])
        self.assertTrue({event["entity_id"] for event in result.summons[:2]}.isdisjoint({1, 2}))

    def test_attacker_slot_summon_does_not_steal_next_friendly_attack(self):
        catalog = self.make_catalog()
        result = simulate(
            [golem(1, health=1), plain(2, attack=1, health=20)],
            [plain(3, attack=2, health=50)],
            random.Random(2),
            catalog,
        )

        player_attackers = [event["attacker"] for event in result.attacks if event["side"] == "player"]
        self.assertEqual(player_attackers[:2], [1, 2])

    def test_defender_slot_summon_can_take_that_sides_next_attack(self):
        catalog = self.make_catalog()
        result = simulate(
            [plain(1, attack=3, health=20), plain(2, attack=0, health=20)],
            [golem(3, attack=1, health=3)],
            random.Random(2),
            catalog,
        )

        token_id = result.summons[0]["entity_id"]
        self.assertEqual(result.attacks[1]["attacker"], token_id)
        self.assertEqual(result.attacks[1]["side"], "enemy")

    def test_token_can_die_later_in_the_same_combat(self):
        catalog = self.make_catalog()
        result = simulate([golem(1, health=2)], [golem(2, health=2)], random.Random(1), catalog)

        summoned_ids = {event["entity_id"] for event in result.summons}
        later_deaths = {event["entity_id"] for event in result.deaths[2:]}
        self.assertTrue(summoned_ids & later_deaths)

    def test_middle_slot_summon_preserves_position_and_next_attacker(self):
        result = simulate(
            [plain(1, attack=0, health=20), golem(2, health=1), plain(3, health=20)],
            [plain(4, attack=1, health=50)], random.Random(2), self.make_catalog(),
        )
        self.assertEqual(result.summons[0]["position"], 1)
        attackers = [event["attacker"] for event in result.attacks if event["side"] == "player"]
        self.assertEqual(attackers[:2], [2, 3])

    def test_recursive_summon_chain_reports_combat_limit(self):
        looping_token = dict(
            TOKEN_CARD, keywords=["DEATHRATTLE"],
            deathrattle={"type": "summon", "token_id": 96764, "count": 1},
        )
        catalog = self.make_catalog(looping_token)
        result = simulate([golem(1, health=2)], [golem(2, health=2)], random.Random(1), catalog)
        self.assertEqual((result.reason, len(result.attacks)), ("combat_limit", 10000))
        self.assertEqual(len(result.summons), 20000)

    def test_board_limit_records_omitted_summons(self):
        double_golem = dict(GOLEM_CARD, deathrattle={"type": "summon", "token_id": 96764, "count": 2})
        catalog = self.make_catalog(double_golem)
        player = [plain(i, attack=0, health=5) for i in range(1, 7)] + [golem(7, health=1)]
        enemy = [plain(8, attack=1, health=20)]

        result = simulate(player, enemy, random.Random(1), catalog)

        self.assertEqual([event["source"] for event in result.summons if event["event"] == "summon"], [7])
        self.assertEqual([event["omitted"] for event in result.summons if event["event"] == "summon_omitted"], [1])

    def test_combat_rejects_oversized_input_board(self):
        catalog = self.make_catalog()

        with self.assertRaisesRegex(ValueError, "seven minions"):
            simulate([plain(i) for i in range(1, 9)], [plain(9)], random.Random(1), catalog)

    def test_multiple_summons_before_cursor_preserve_next_living_attacker(self):
        double_golem = dict(GOLEM_CARD, deathrattle={"type": "summon", "token_id": 96764, "count": 2})
        catalog = self.make_catalog(double_golem)
        result = simulate(
            [golem(1, attack=1, health=1), plain(2, attack=1, health=20)],
            [plain(3, attack=1, health=50)],
            random.Random(2),
            catalog,
        )

        player_attackers = [event["attacker"] for event in result.attacks if event["side"] == "player"]
        self.assertEqual(player_attackers[:2], [1, 2])

    def test_inputs_and_seed_remain_stable_with_summons(self):
        catalog = self.make_catalog()
        player = [golem(1, health=2)]
        enemy = [golem(2, health=2)]
        before = copy.deepcopy((player, enemy))

        first = simulate(player, enemy, random.Random(4), catalog)
        second = simulate(player, enemy, random.Random(4), catalog)

        self.assertEqual((player, enemy), before)
        self.assertEqual(first, second)

    def test_golden_deathrattle_is_not_silently_downgraded(self):
        catalog = self.make_catalog()

        with self.assertRaisesRegex(ValueError, "Golden deathrattle"):
            simulate([golem(1, golden=True)], [plain(2)], random.Random(1), catalog)
