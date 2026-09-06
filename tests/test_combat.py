import copy
import random
import unittest

from hearthstone_ai.cards import Catalog, Minion
from hearthstone_ai.combat import simulate


def unit(entity_id, attack, health, **kwargs):
    return Minion(entity_id, 99159, attack, health, **kwargs)


class CombatTests(unittest.TestCase):
    def setUp(self):
        self.catalog = Catalog()

    def run_combat(self, player, enemy, seed=1):
        return simulate(player, enemy, random.Random(seed), self.catalog)

    def test_attacker_death_does_not_skip_next(self):
        result = self.run_combat([unit(1, 1, 1), unit(2, 1, 20), unit(3, 1, 20)], [unit(4, 1, 100)])
        attacks = [event["attacker"] for event in result.attacks if event["side"] == "player"]
        self.assertEqual(attacks[:3], [1, 2, 3])

    def test_defensive_death_preserves_ring(self):
        result = self.run_combat(
            [unit(1, 0, 1, taunt=True), unit(2, 1, 20), unit(3, 1, 20)], [unit(4, 1, 100)]
        )
        attacks = [event["attacker"] for event in result.attacks if event["side"] == "player"]
        self.assertEqual(attacks[:3], [2, 3, 2])

    def test_taunt_and_shield(self):
        result = self.run_combat(
            [unit(1, 2, 10), unit(2, 1, 10)],
            [unit(3, 1, 1), unit(4, 1, 1, taunt=True, divine_shield=True)],
        )
        self.assertEqual(result.attacks[0]["defender"], 4)
        self.assertTrue(result.attacks[0]["defender_shield_broken"])

    def test_simultaneous_damage_and_immutability(self):
        player, enemy = [unit(1, 2, 2)], [unit(2, 2, 2)]
        before = copy.deepcopy((player, enemy))
        result = self.run_combat(player, enemy)
        self.assertEqual(result.result, 0)
        self.assertEqual((player, enemy), before)
        self.assertEqual(result, self.run_combat(player, enemy))

    def test_zero_attack_stalemate(self):
        result = self.run_combat([unit(1, 0, 2)], [unit(2, 0, 2)])
        self.assertEqual((result.result, result.reason, result.attacks), (0, "stalemate", ()))

    def test_zero_retaliation_does_not_break_shield(self):
        result = self.run_combat([unit(1, 2, 2, divine_shield=True)], [unit(2, 0, 1)])
        self.assertFalse(result.attacks[0]["attacker_shield_broken"])
        self.assertEqual((result.result, result.damage_dealt), (1, 3))

    def test_empty_boards(self):
        self.assertEqual(self.run_combat([], []).result, 0)
        self.assertEqual(self.run_combat([], [unit(1, 1, 1)]).damage_taken, 3)

    def test_survivor_damage_uses_tier_not_golden_multiplier(self):
        result = simulate(
            [unit(1, 10, 10, golden=True)], [], random.Random(1), self.catalog, player_tier=2
        )
        self.assertEqual(result.damage_dealt, 4)

    def test_combat_limit_is_explicit(self):
        result = self.run_combat([unit(1, 1, 100000)], [unit(2, 1, 100000)])
        self.assertEqual(
            (result.result, result.reason, len(result.attacks)), (0, "combat_limit", 10000)
        )

    def test_negative_attack_rejected(self):
        with self.assertRaises(ValueError):
            self.run_combat([unit(1, -1, 1)], [])
