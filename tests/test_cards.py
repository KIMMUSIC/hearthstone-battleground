import random
import unittest
import json
from pathlib import Path
import tempfile

from hearthstone_ai.artifacts import compatibility_signature
from hearthstone_ai.cards import Catalog


class CatalogTests(unittest.TestCase):
    def test_fixed_mapping_and_golden(self):
        catalog = Catalog()
        self.assertEqual(
            catalog.mapping,
            {
                42467: 1,
                72387: 2,
                2053: 3,
                99159: 4,
                96763: 5,
                96764: 6,
                49278: 7,
                60628: 8,
                96796: 9,
                103652: 10,
                108715: 11,
            },
        )
        card = catalog.spawn(42467, 7, golden=True)
        self.assertEqual((card.attack, card.health, card.entity_id), (2, 2, 7))
        self.assertTrue(card.golden and card.taunt and card.divine_shield)

    def test_golem_bundle_contract(self):
        catalog = Catalog()
        golem = catalog.by_id[96763]
        token = catalog.by_id[96764]
        self.assertEqual((golem.tier, golem.attack, golem.health), (2, 2, 3))
        self.assertEqual((golem.deathrattle_token_id, golem.deathrattle_count), (96764, 1))
        self.assertTrue(golem.pool)
        self.assertFalse(golem.triple_allowed)
        self.assertFalse(golem.golden_allowed)
        self.assertEqual((token.tier, token.attack, token.health), (1, 2, 1))
        self.assertFalse(token.pool)
        self.assertFalse(token.triple_allowed)
        self.assertFalse(token.golden_allowed)
        self.assertEqual((catalog.mapping[96763], catalog.mapping[96764]), (5, 6))
        with self.assertRaisesRegex(ValueError, "Golden card is unsupported: 96763"):
            catalog.spawn(96763, 1, golden=True)
        with self.assertRaisesRegex(ValueError, "Golden card is unsupported: 96764"):
            catalog.spawn(96764, 1, golden=True)

    def test_reserved_mapping_indices_are_not_active_cards(self):
        catalog = Catalog()
        self.assertEqual(catalog.active_card_ids, frozenset(catalog.by_id))
        self.assertTrue({49278, 103652, 108715}.isdisjoint(catalog.active_card_ids))
        for reserved_id in (49278, 103652, 108715):
            with self.assertRaisesRegex(KeyError, str(reserved_id)):
                catalog.spawn(reserved_id, reserved_id)
        self.assertEqual(
            {card.dbf_id for card in catalog.sample(random.Random(17), 2, 300)},
            {42467, 72387, 2053, 99159, 96763, 60628, 96796},
        )

    def test_second_bundle_contract_and_golden_encoding(self):
        catalog = Catalog()
        expected = {
            60628: ("용혈족 부관", 1, 2, 3, True, False),
            96796: ("공허방랑자", 1, 1, 3, True, False),
        }
        for card_id, (name, tier, attack, health, taunt, shield) in expected.items():
            card = catalog.by_id[card_id]
            self.assertEqual((card.name, card.tier, card.attack, card.health), (name, tier, attack, health))
            self.assertEqual((card.taunt, card.divine_shield, card.pool), (taunt, shield, True))
            golden = catalog.spawn(card_id, card_id, golden=True)
            self.assertEqual((golden.attack, golden.health), (attack * 2, health * 2))
            self.assertTrue(golden.golden)
            self.assertEqual((golden.taunt, golden.divine_shield), (taunt, shield))

    def test_sampling_tier_and_reproducibility(self):
        catalog = Catalog()
        sample = catalog.sample(random.Random(10), 2, 20, exact=True)
        self.assertTrue(all(c.tier == 2 for c in sample))
        self.assertIn(96763, {c.dbf_id for c in catalog.sample(random.Random(3), 2, 200)})
        self.assertNotIn(96764, {c.dbf_id for c in catalog.sample(random.Random(4), 2, 200)})
        self.assertEqual(sample, catalog.sample(random.Random(10), 2, 20, exact=True))
        with self.assertRaises(ValueError):
            catalog.sample(random.Random(1), 3, 1, exact=True)

    def test_unknown_card_is_error(self):
        with self.assertRaises(KeyError):
            Catalog().spawn(999999, 1)

    def test_unsupported_effect_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "cards.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "cards": [
                            {
                                "dbf_id": 1,
                                "name": "unsupported",
                                "tier": 1,
                                "attack": 1,
                                "health": 1,
                                "keywords": ["DEATHRATTLE"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (path / "card_mapping.json").write_text(
                json.dumps({"schema_version": 1, "padding_index": 0, "mapping": {"1": 1}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Unsupported card keywords"):
                Catalog(path)

    def test_triple_requires_golden_support(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "cards.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "cards": [
                            {
                                "dbf_id": 1,
                                "name": "bad triple",
                                "tier": 1,
                                "attack": 1,
                                "health": 1,
                                "keywords": [],
                                "golden_allowed": False,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (path / "card_mapping.json").write_text(
                json.dumps({"schema_version": 1, "padding_index": 0, "mapping": {"1": 1}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Triple requires golden support"):
                Catalog(path)

    def test_raw_snapshot_provenance_is_present(self):
        path = Path(__file__).resolve().parents[1] / "data" / "cards.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["source"]["sha256"],
            "38eaff2e32f13de5bfe684eda8ad75c1ac792f1257eb459ad711d6903372b097",
        )
        self.assertEqual(payload["ruleset"], "archival-subset-v4")

    def test_mapping_and_observation_versions_mark_old_models_incompatible(self):
        path = Path(__file__).resolve().parents[1] / "data" / "card_mapping.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["content_version"], "card-mapping-v4")
        self.assertEqual(
            {int(card_id): payload["mapping"][card_id] for card_id in ("42467", "72387", "2053", "99159")},
            {42467: 1, 72387: 2, 2053: 3, 99159: 4},
        )
        self.assertEqual(
            {int(card_id): payload["mapping"][card_id] for card_id in ("96763", "96764", "49278", "60628", "96796", "103652", "108715")},
            {96763: 5, 96764: 6, 49278: 7, 60628: 8, 96796: 9, 103652: 10, 108715: 11},
        )
        self.assertEqual(payload["reserved_tombstones"], ["49278", "103652", "108715"])
        self.assertEqual(compatibility_signature()["observation_version"], "onehot-slots-v4")
