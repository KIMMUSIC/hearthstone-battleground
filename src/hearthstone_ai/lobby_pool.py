"""Finite shared card copy pool for the restricted eight-player lobby."""

from __future__ import annotations

from collections import Counter
import random
from typing import Mapping

from .cards import Catalog


class SharedPool:
    """Track only initial and currently available base card copies."""

    def __init__(self, catalog: Catalog | None = None, copies: int | Mapping[int, int] = 15):
        self.catalog = catalog if catalog is not None else Catalog()
        self.initial = self._initial_counts(copies)
        self.available = dict(self.initial)

    def _initial_counts(self, copies: int | Mapping[int, int]) -> dict[int, int]:
        if isinstance(copies, Mapping):
            counts = {}
            for card_id, count in copies.items():
                card_id = self._require_card_id(card_id)
                self._require_supported_pool_card(card_id)
                counts[card_id] = self._require_count(count)
            return dict(sorted(counts.items()))
        count = self._require_count(copies)
        return {
            card.dbf_id: count
            for card in sorted(self.catalog.cards, key=lambda card: card.dbf_id)
            if card.pool and card.tier in (1, 2)
        }

    def draw(
        self,
        rng: random.Random,
        tier: int,
        count: int,
        exact: bool = False,
    ) -> list[int]:
        tier = self._require_tier(tier)
        count = self._require_count(count)
        if type(exact) is not bool:
            raise ValueError("exact must be boolean")
        drawn = []
        for _ in range(count):
            candidates = [
                card_id
                for card_id, available in self.available.items()
                if available > 0 and self._tier_matches(card_id, tier, exact)
            ]
            total = sum(self.available[card_id] for card_id in candidates)
            if total == 0:
                break
            ticket = rng.randrange(total)
            running = 0
            for card_id in candidates:
                running += self.available[card_id]
                if ticket < running:
                    self.available[card_id] -= 1
                    drawn.append(card_id)
                    break
        return drawn

    def return_cards(self, card_ids: list[int]) -> None:
        if type(card_ids) is not list:
            raise ValueError("card_ids must be a list")
        returned = Counter(self._require_card_id(card_id) for card_id in card_ids)
        for card_id, amount in returned.items():
            self._require_supported_pool_card(card_id)
            if card_id not in self.initial:
                raise ValueError(f"Card was not part of this shared pool: {card_id}")
            if self.available[card_id] + amount > self.initial[card_id]:
                raise ValueError(f"Returning cards would exceed the initial pool: {card_id}")
        for card_id, amount in returned.items():
            self.available[card_id] += amount

    def _require_supported_pool_card(self, card_id: int) -> None:
        try:
            card = self.catalog.by_id[card_id]
        except KeyError as error:
            raise ValueError(f"Unknown card: {card_id}") from error
        if not card.pool or card.tier not in (1, 2):
            raise ValueError(f"Card is not in the supported shared pool: {card_id}")

    def _tier_matches(self, card_id: int, tier: int, exact: bool) -> bool:
        card_tier = self.catalog.by_id[card_id].tier
        return card_tier == tier if exact else card_tier <= tier

    @staticmethod
    def _require_card_id(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("card IDs must be integers")
        return value

    @staticmethod
    def _require_count(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("copy counts must be nonnegative integers")
        return value

    @staticmethod
    def _require_tier(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value not in (1, 2):
            raise ValueError("tier must be 1 or 2")
        return value
