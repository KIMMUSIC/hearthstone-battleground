from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from diagnose_lobby_040 import decode_observation  # noqa: E402
from hearthstone_ai.cards import Catalog  # noqa: E402
from hearthstone_ai.lobby import MinionView, PolicyView, PublicPlayerView  # noqa: E402
from hearthstone_ai.lobby_observation import OBSERVATION_SCALE_FIXED_V1, LobbyObservation  # noqa: E402
from hearthstone_ai.lobby_policies import heuristic_policy  # noqa: E402


def minion(catalog, card_id):
    card = catalog.by_id[card_id]
    return MinionView(card.dbf_id, card.attack, card.health, False, card.taunt, card.divine_shield)


def public_players():
    return tuple(PublicPlayerView(seat, 30, 1, True, None) for seat in range(8))


def view(**overrides):
    catalog = overrides.pop("catalog")
    cards = sorted(catalog.active_card_ids)
    defaults = dict(
        seat=0,
        round=4,
        hp=28,
        gold=6,
        tier=2,
        upgrade_cost=0,
        frozen=False,
        actions_remaining=17,
        swaps_remaining=3,
        hand=(minion(catalog, cards[-1]),),
        board=tuple(minion(catalog, card_id) for card_id in cards[:7]),
        shop=tuple(minion(catalog, card_id) for card_id in cards[:3]) + (None, None, None, None),
        pending_discover=(),
        public_players=public_players(),
        legal_actions=tuple(range(11, 18)),
        cards=tuple(catalog.cards),
    )
    defaults.update(overrides)
    return PolicyView(**defaults)


def encode(policy_view):
    return LobbyObservation(observation_scale=OBSERVATION_SCALE_FIXED_V1).encode(policy_view)


def test_decode_preserves_board_full_sell_heuristic():
    catalog = Catalog()
    policy_view = view(catalog=catalog)
    decoded = decode_observation(encode(policy_view), catalog)
    assert len(decoded.board) == 7
    assert decoded.shop.count(None) == 4
    assert heuristic_policy(decoded) == heuristic_policy(policy_view)


def test_decode_preserves_discover_heuristic():
    catalog = Catalog()
    cards = sorted(catalog.active_card_ids)
    public = list(public_players())
    public[3] = PublicPlayerView(3, 0, 2, False, 7.5)
    policy_view = view(
        catalog=catalog,
        pending_discover=tuple(minion(catalog, card_id) for card_id in cards[:3]),
        public_players=tuple(public),
        legal_actions=(34, 35, 36),
    )
    decoded = decode_observation(encode(policy_view), catalog)
    assert len(decoded.pending_discover) == 3
    assert decoded.public_players[3].rank == 7.5
    assert heuristic_policy(decoded) == heuristic_policy(policy_view)


def test_decode_rejects_illegal_slot_encoding():
    catalog = Catalog()
    obs = encode(view(catalog=catalog))
    first_card_slot = next(iter(catalog.mapping.values()))
    second_card_slot = next(slot for slot in catalog.mapping.values() if slot != first_card_slot)
    obs["shop"][0][first_card_slot] = 1
    obs["shop"][0][second_card_slot] = 1
    with pytest.raises(ValueError, match="exactly one active"):
        decode_observation(obs, catalog)
