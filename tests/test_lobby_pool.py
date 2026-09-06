import random
from collections import Counter

import pytest

from hearthstone_ai.cards import Catalog
from hearthstone_ai.lobby_pool import SharedPool


def card_ids_by_tier(catalog):
    return {
        tier: [card.dbf_id for card in catalog.cards if card.pool and card.tier == tier]
        for tier in (1, 2)
    }


def test_default_pool_contains_supported_shop_cards_only_with_fifteen_copies():
    catalog = Catalog()
    pool = SharedPool(catalog)
    expected = {
        card.dbf_id: 15
        for card in catalog.cards
        if card.pool and card.tier in (1, 2)
    }

    assert pool.initial == expected
    assert pool.available == expected
    assert 96764 not in pool.initial


def test_draw_uses_weighted_available_copies_without_replacement_and_is_deterministic():
    catalog = Catalog()
    by_tier = card_ids_by_tier(catalog)
    copies = {by_tier[1][0]: 2, by_tier[1][1]: 1, by_tier[2][0]: 2}
    first = SharedPool(catalog, copies=copies)
    second = SharedPool(catalog, copies=copies)

    first_draw = first.draw(random.Random(17), tier=2, count=5)
    second_draw = second.draw(random.Random(17), tier=2, count=5)

    assert first_draw == second_draw
    assert Counter(first_draw) == copies
    assert first.available == {card_id: 0 for card_id in copies}


def test_draw_supports_zero_one_and_two_copy_supply_and_returns_up_to_requested():
    catalog = Catalog()
    by_tier = card_ids_by_tier(catalog)
    zero, one, two = by_tier[1][:3]
    pool = SharedPool(catalog, copies={zero: 0, one: 1, two: 2})

    drawn = pool.draw(random.Random(3), tier=1, count=10, exact=True)

    assert len(drawn) == 3
    assert zero not in drawn
    assert Counter(drawn) == {one: 1, two: 2}
    assert pool.available == {zero: 0, one: 0, two: 0}


def test_exact_draw_excludes_lower_tiers_and_inclusive_draw_allows_them():
    catalog = Catalog()
    by_tier = card_ids_by_tier(catalog)
    tier_one, tier_two = by_tier[1][0], by_tier[2][0]

    exact_pool = SharedPool(catalog, copies={tier_one: 3, tier_two: 3})
    assert set(exact_pool.draw(random.Random(1), tier=2, count=10, exact=True)) == {tier_two}
    assert exact_pool.available[tier_one] == 3

    inclusive_pool = SharedPool(catalog, copies={tier_one: 3, tier_two: 3})
    assert set(inclusive_pool.draw(random.Random(1), tier=2, count=10)) == {tier_one, tier_two}


def test_return_cards_restores_available_copies_after_validating_entire_batch():
    catalog = Catalog()
    card_id = card_ids_by_tier(catalog)[1][0]
    pool = SharedPool(catalog, copies={card_id: 2})
    assert pool.draw(random.Random(1), tier=1, count=1) == [card_id]

    pool.return_cards([card_id])

    assert pool.available == pool.initial == {card_id: 2}


def test_return_cards_rejects_overreturn_atomically():
    catalog = Catalog()
    card_id = card_ids_by_tier(catalog)[1][0]
    pool = SharedPool(catalog, copies={card_id: 2})
    pool.draw(random.Random(1), tier=1, count=1)
    before = dict(pool.available)

    with pytest.raises(ValueError, match="exceed"):
        pool.return_cards([card_id, card_id])

    assert pool.available == before


@pytest.mark.parametrize("copies", [{999999: 1}, {96764: 1}, {42467: -1}, {42467: True}])
def test_injected_copies_reject_unknown_nonpool_and_invalid_counts(copies):
    with pytest.raises(ValueError):
        SharedPool(Catalog(), copies=copies)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tier": 0, "count": 1},
        {"tier": 3, "count": 1},
        {"tier": True, "count": 1},
        {"tier": 1, "count": -1},
        {"tier": 1, "count": True},
        {"tier": 1, "count": 1, "exact": 1},
    ],
)
def test_draw_rejects_invalid_arguments_without_mutating(kwargs):
    pool = SharedPool(Catalog(), copies={42467: 1})
    before = dict(pool.available)

    with pytest.raises(ValueError):
        pool.draw(random.Random(1), **kwargs)

    assert pool.available == before


def test_return_cards_rejects_nonpool_unknown_and_non_integer_ids_atomically():
    pool = SharedPool(Catalog(), copies={42467: 1})
    before = dict(pool.available)

    for card_ids in ([96764], [999999], [True], (42467,)):
        with pytest.raises(ValueError):
            pool.return_cards(card_ids)
        assert pool.available == before
