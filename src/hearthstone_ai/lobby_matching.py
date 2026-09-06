"""Lobby opponent matching for the restricted 8-player contract."""

from __future__ import annotations

import random
from collections.abc import Iterable

Pair = tuple[int, int | None]
Matching = tuple[Pair, ...]
Cost = tuple[int, int, int]


def match_players(
    alive: list[int],
    history: list[tuple[tuple[int, int | None], ...]],
    ghost_counts: dict[int, int],
    last_ghost: int | None,
    rng: random.Random,
) -> tuple[Matching, Cost]:
    """Return a deterministic-cost lobby matching, using ``rng`` only for final ties."""
    seats = _validate_alive(alive)
    recent_real_pairs = _recent_real_pairs(history)
    candidates = sorted(
        {_canonical_matching(candidate) for candidate in _candidate_matchings(seats)},
        key=_matching_sort_key,
    )
    scored = [(matching, _cost(matching, recent_real_pairs, ghost_counts, last_ghost)) for matching in candidates]
    best_cost = min(cost for _, cost in scored)
    best_matchings = [matching for matching, cost in scored if cost == best_cost]
    return rng.choice(best_matchings), best_cost


def _validate_alive(alive: list[int]) -> tuple[int, ...]:
    if not isinstance(alive, list):
        raise ValueError("alive must be a list of seat integers")
    if not 2 <= len(alive) <= 8:
        raise ValueError("alive must contain 2 to 8 seats")
    if any(type(seat) is not int for seat in alive):
        raise ValueError("alive seats must be integers")
    if any(seat < 0 or seat > 7 for seat in alive):
        raise ValueError("alive seats must be in the range 0..7")
    if len(set(alive)) != len(alive):
        raise ValueError("alive seats must be distinct")
    return tuple(alive)


def _candidate_matchings(seats: tuple[int, ...]) -> Iterable[Matching]:
    if len(seats) % 2:
        for ghost in seats:
            remaining = tuple(seat for seat in seats if seat != ghost)
            for matching in _perfect_matchings(remaining):
                yield ((ghost, None), *matching)
        return
    yield from _perfect_matchings(seats)


def _perfect_matchings(seats: tuple[int, ...]) -> Iterable[Matching]:
    if not seats:
        yield ()
        return
    first = seats[0]
    for index in range(1, len(seats)):
        second = seats[index]
        rest = seats[1:index] + seats[index + 1 :]
        for matching in _perfect_matchings(rest):
            yield ((first, second), *matching)


def _recent_real_pairs(
    history: list[tuple[tuple[int, int | None], ...]],
) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for round_pairs in history[-2:]:
        for left, right in round_pairs:
            if right is None:
                continue
            pair = _canonical_real_pair(left, right)
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def _cost(
    matching: Matching,
    recent_real_pairs: dict[tuple[int, int], int],
    ghost_counts: dict[int, int],
    last_ghost: int | None,
) -> Cost:
    repeat_cost = 0
    repeated_ghost_cost = 0
    ghost_count_cost = 0
    for left, right in matching:
        if right is None:
            repeated_ghost_cost = int(left == last_ghost)
            ghost_count_cost = ghost_counts.get(left, 0)
            continue
        repeat_cost += recent_real_pairs.get(_canonical_real_pair(left, right), 0)
    return repeat_cost, repeated_ghost_cost, ghost_count_cost


def _canonical_matching(matching: Matching) -> Matching:
    pairs = tuple(_canonical_pair(left, right) for left, right in matching)
    return tuple(sorted(pairs, key=_pair_sort_key))


def _canonical_pair(left: int, right: int | None) -> Pair:
    if right is None:
        return left, None
    return _canonical_real_pair(left, right)


def _canonical_real_pair(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def _pair_sort_key(pair: Pair) -> tuple[int, int]:
    left, right = pair
    return left, -1 if right is None else right


def _matching_sort_key(matching: Matching) -> tuple[tuple[int, int], ...]:
    return tuple(_pair_sort_key(pair) for pair in matching)
