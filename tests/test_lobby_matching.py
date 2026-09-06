import random

import pytest

from hearthstone_ai.lobby_matching import match_players


def flatten_real_seats(pairs):
    return [seat for pair in pairs for seat in pair if seat is not None]


def test_alive_counts_cover_each_seat_once_and_use_expected_parity():
    for count in range(2, 9):
        alive = list(range(count))
        pairs, cost = match_players(alive, [], {}, None, random.Random(100 + count))

        assert sorted(flatten_real_seats(pairs)) == alive
        assert cost == (0, 0, 0)
        if count % 2 == 0:
            assert len(pairs) == count // 2
            assert all(right is not None for _, right in pairs)
        else:
            ghost_pairs = [pair for pair in pairs if pair[1] is None]
            assert len(pairs) == (count + 1) // 2
            assert len(ghost_pairs) == 1
            assert ghost_pairs[0][0] in alive


def test_even_count_avoids_recent_repeat_when_possible():
    pairs, cost = match_players(
        [0, 1, 2, 3],
        [((0, 1), (2, 3))],
        {},
        None,
        random.Random(4),
    )

    assert cost == (0, 0, 0)
    assert (0, 1) not in pairs
    assert (2, 3) not in pairs


def test_only_last_two_rounds_count_as_repeats():
    pairs, cost = match_players(
        [0, 1],
        [((0, 1),), ((0, None),), ((1, None),)],
        {},
        None,
        random.Random(1),
    )

    assert pairs == ((0, 1),)
    assert cost == (0, 0, 0)


def test_two_player_recent_repeat_is_unavoidable_and_does_not_hang():
    pairs, cost = match_players(
        [1, 0],
        [((0, 1),), ((1, 0),)],
        {},
        None,
        random.Random(1),
    )

    assert pairs == ((0, 1),)
    assert cost == (2, 0, 0)


def test_odd_count_minimizes_last_ghost_then_ghost_counts():
    pairs, cost = match_players(
        [0, 1, 2],
        [],
        {0: 4, 1: 0, 2: 0},
        1,
        random.Random(0),
    )

    assert (2, None) in pairs
    assert cost == (0, 0, 0)


def test_ghost_count_is_reported_when_all_remaining_ghost_choices_are_tied():
    pairs, cost = match_players(
        [0, 1, 2],
        [((0, 1),), ((0, 2),)],
        {0: 2, 1: 5, 2: 7},
        None,
        random.Random(0),
    )

    assert pairs == ((0, None), (1, 2))
    assert cost == (0, 0, 2)


def test_ties_are_canonical_before_rng_choice_and_reproducible():
    args = ([0, 1, 2, 3], [], {}, None)
    first = match_players(*args, random.Random(33))
    second = match_players(*args, random.Random(33))

    assert first == second


def test_rng_is_injected_and_global_rng_is_not_used(monkeypatch):
    def fail_choice(_items):
        raise AssertionError("global random.choice must not be used")

    monkeypatch.setattr(random, "choice", fail_choice)
    pairs, cost = match_players([0, 1, 2, 3], [], {}, None, random.Random(12))

    assert len(pairs) == 2
    assert cost == (0, 0, 0)


@pytest.mark.parametrize(
    "alive",
    [
        [],
        [0],
        list(range(9)),
        [0, 0],
        [-1, 0],
        [0, 8],
        [False, 1],
    ],
)
def test_invalid_alive_lists_are_rejected(alive):
    with pytest.raises(ValueError):
        match_players(alive, [], {}, None, random.Random(1))


def test_alive_must_be_a_list():
    with pytest.raises(ValueError):
        match_players((0, 1), [], {}, None, random.Random(1))
