"""Versioned, intentionally limited archival card catalog."""

from dataclasses import dataclass
import json
from pathlib import Path
import random


@dataclass(frozen=True)
class Card:
    dbf_id: int
    name: str
    tier: int
    attack: int
    health: int
    taunt: bool = False
    divine_shield: bool = False
    deathrattle_token_id: int | None = None
    deathrattle_count: int = 0
    pool: bool = True
    triple_allowed: bool = True
    golden_allowed: bool = True


@dataclass
class Minion:
    entity_id: int
    card_id: int
    attack: int
    health: int
    golden: bool = False
    taunt: bool = False
    divine_shield: bool = False


class Catalog:
    def __init__(self, data_dir: Path | None = None):
        directory = (
            data_dir if data_dir is not None else Path(__file__).resolve().parents[2] / "data"
        )
        payload = json.loads((directory / "cards.json").read_text(encoding="utf-8"))
        mapping_payload = json.loads((directory / "card_mapping.json").read_text(encoding="utf-8"))
        if payload["schema_version"] != 1 or mapping_payload["schema_version"] != 1:
            raise ValueError("Unsupported catalog or mapping schema version")
        cards = []
        for record in payload["cards"]:
            keywords = set(record.get("keywords", []))
            deathrattle = record.get("deathrattle")
            allowed_keywords = {"TAUNT", "DIVINE_SHIELD"}
            if deathrattle is not None:
                allowed_keywords.add("DEATHRATTLE")
            unsupported = keywords - allowed_keywords
            if unsupported:
                raise ValueError(f"Unsupported card keywords: {sorted(unsupported)}")
            deathrattle_token_id = None
            deathrattle_count = 0
            if deathrattle is not None:
                if (
                    not isinstance(deathrattle, dict)
                    or set(deathrattle) != {"type", "token_id", "count"}
                    or deathrattle.get("type") != "summon"
                    or type(deathrattle.get("token_id")) is not int
                    or type(deathrattle.get("count")) is not int
                    or deathrattle["count"] <= 0
                    or deathrattle["count"] > 7
                ):
                    raise ValueError(f"Unsupported deathrattle: {record['dbf_id']}")
                deathrattle_token_id = deathrattle["token_id"]
                deathrattle_count = deathrattle["count"]
            card = Card(
                record["dbf_id"],
                record["name"],
                record["tier"],
                record["attack"],
                record["health"],
                "TAUNT" in keywords,
                "DIVINE_SHIELD" in keywords,
                deathrattle_token_id,
                deathrattle_count,
                _optional_bool(record, "pool", True),
                _optional_bool(record, "triple_allowed", True),
                _optional_bool(record, "golden_allowed", True),
            )
            if card.dbf_id <= 0 or card.tier < 1 or card.attack < 0 or card.health <= 0:
                raise ValueError(f"Invalid card stats: {card.dbf_id}")
            if card.triple_allowed and not card.golden_allowed:
                raise ValueError(f"Triple requires golden support: {card.dbf_id}")
            cards.append(card)
        self.cards = tuple(cards)
        self.by_id = {card.dbf_id: card for card in self.cards}
        for card in self.cards:
            if card.deathrattle_token_id is not None and card.deathrattle_token_id not in self.by_id:
                raise ValueError(f"Unknown deathrattle token: {card.deathrattle_token_id}")
        self.mapping = {int(key): value for key, value in mapping_payload["mapping"].items()}
        if not cards or len(self.by_id) != len(cards):
            raise ValueError("Catalog must be nonempty and card IDs unique")
        indices = list(self.mapping.values())
        if (
            mapping_payload["padding_index"] != 0
            or any(type(i) is not int or i <= 0 for i in indices)
            or len(indices) != len(set(indices))
        ):
            raise ValueError("Mapping indices must be unique positive integers; padding is zero")
        reserved_tombstones = mapping_payload.get("reserved_tombstones", [])
        if (
            type(reserved_tombstones) is not list
            or any(type(card_id) is not str for card_id in reserved_tombstones)
            or len(reserved_tombstones) != len(set(reserved_tombstones))
        ):
            raise ValueError("Reserved tombstones must be unique string card IDs")
        self.reserved_tombstones = frozenset(int(card_id) for card_id in reserved_tombstones)
        if not self.reserved_tombstones <= self.mapping.keys():
            raise ValueError("Reserved tombstones must keep mapping slots")
        if self.reserved_tombstones & self.by_id.keys():
            raise ValueError("Reserved tombstones cannot be active cards")
        if not self.by_id.keys() <= self.mapping.keys():
            raise ValueError("Catalog contains unmapped card IDs")
        self.active_card_ids = frozenset(self.by_id)

    def sample(
        self, rng: random.Random, max_tier: int, count: int, exact: bool = False
    ) -> list[Card]:
        if count < 0:
            raise ValueError("Sample count must be nonnegative")
        pool = [
            card
            for card in self.cards
            if card.pool and (card.tier == max_tier if exact else card.tier <= max_tier)
        ]
        if not pool:
            raise ValueError(f"No supported cards for tier {max_tier} (exact={exact})")
        return [rng.choice(pool) for _ in range(count)]

    def spawn(self, card_id: int, entity_id: int, golden: bool = False) -> Minion:
        card = self.by_id[card_id]
        if golden and not card.golden_allowed:
            raise ValueError(f"Golden card is unsupported: {card_id}")
        multiplier = 2 if golden else 1
        return Minion(
            entity_id,
            card_id,
            card.attack * multiplier,
            card.health * multiplier,
            golden,
            card.taunt,
            card.divine_shield,
        )


def _optional_bool(record: dict, key: str, default: bool) -> bool:
    value = record.get(key, default)
    if type(value) is not bool:
        raise ValueError(f"{key} must be boolean: {record['dbf_id']}")
    return value
