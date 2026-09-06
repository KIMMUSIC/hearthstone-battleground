"""Restricted eight-player Battlegrounds lobby engine."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import random
from types import MappingProxyType

from .actions import decode_action
from .cards import Card, Catalog, Minion
from .combat import CombatResult, simulate
from .lobby_matching import match_players
from .lobby_pool import SharedPool

PLAYER_COUNT = 8
STARTING_HP = 30
MAX_ACTIONS = 24
MAX_SWAPS = 3
HAND_SIZE = 10
BOARD_SIZE = 7
SHOP_SIZE = 7
MAX_TIER = 2


@dataclass(frozen=True)
class MinionView:
    card_id: int
    attack: int
    health: int
    golden: bool
    taunt: bool
    divine_shield: bool


@dataclass(frozen=True)
class PublicPlayerView:
    seat: int
    hp: int
    tier: int
    alive: bool
    rank: float | None


@dataclass(frozen=True)
class PolicyView:
    seat: int
    round: int
    hp: int
    gold: int
    tier: int
    upgrade_cost: int
    frozen: bool
    actions_remaining: int
    swaps_remaining: int
    hand: tuple[MinionView, ...]
    board: tuple[MinionView, ...]
    shop: tuple[MinionView | None, ...]
    pending_discover: tuple[MinionView, ...]
    public_players: tuple[PublicPlayerView, ...]
    legal_actions: tuple[int, ...]
    cards: tuple[Card, ...]


@dataclass
class PlayerState:
    seat: int
    hp: int = STARTING_HP
    gold: int = 0
    tier: int = 1
    upgrade_cost: int = 5
    frozen: bool = False
    actions_remaining: int = MAX_ACTIONS
    swaps_remaining: int = MAX_SWAPS
    hand: list[Minion] = field(default_factory=list)
    board: list[Minion] = field(default_factory=list)
    shop: list[Minion | None] = field(default_factory=lambda: [None] * SHOP_SIZE)
    pending_discover: list[Minion] = field(default_factory=list)
    alive: bool = True
    ready: bool = False
    rank: float | None = None


class LobbyGame:
    """Finite-pool, sequential-recruit, simultaneous-combat eight-player lobby."""

    def __init__(
        self,
        seed: int = 7,
        max_rounds: int = 50,
        catalog: Catalog | None = None,
        pool_copies: int | dict[int, int] = 15,
    ):
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        if type(max_rounds) is not int or max_rounds < 1:
            raise ValueError("max_rounds must be a positive integer")
        self.seed = seed
        self.max_rounds = max_rounds
        self.catalog = catalog if catalog is not None else Catalog()
        self.cards = tuple(self.catalog.cards)
        self.card_mapping = MappingProxyType(dict(self.catalog.mapping))
        self.pool = SharedPool(self.catalog, pool_copies)
        self._shop_rngs = [random.Random(f"lobby:{seed}:shop:{seat}") for seat in range(PLAYER_COUNT)]
        self._match_rng = random.Random(f"lobby:{seed}:matching")
        self._combat_rng = random.Random(f"lobby:{seed}:combat")
        self._entity_id = 0
        self.round = 1
        self.players = [PlayerState(seat) for seat in range(PLAYER_COUNT)]
        self.current_seat: int | None = None
        self.terminated = False
        self.truncated = False
        self.done = False
        self.trace: list[dict] = []
        self._round_order: list[int] = []
        self._round_index = 0
        self._matching_history: list[tuple[tuple[int, int | None], ...]] = []
        self._ghost_counts: dict[int, int] = {}
        self._last_ghost: int | None = None
        self._ghost_board: list[Minion] = []
        self._ghost_tier = 1
        self._ghost_source_seat: int | None = None
        self._ghost_death_round: int | None = None
        self._begin_round()
        self.assert_conservation()

    def legal_actions(self, seat: int) -> list[int]:
        if self.done or seat != self.current_seat or not self._valid_seat(seat):
            return []
        player = self.players[seat]
        if not player.alive or player.ready:
            return []
        if player.pending_discover:
            if len(player.hand) >= HAND_SIZE:
                return []
            return [34 + index for index in range(len(player.pending_discover))]
        if player.actions_remaining <= 0:
            return [0]
        legal = [0, 3]
        if player.gold >= 1:
            legal.append(1)
        if player.tier < MAX_TIER and player.gold >= player.upgrade_cost:
            legal.append(2)
        if player.gold >= 3 and len(player.hand) < HAND_SIZE:
            legal.extend(4 + index for index, card in enumerate(player.shop) if card is not None)
        legal.extend(11 + index for index in range(len(player.board)))
        if len(player.board) < BOARD_SIZE:
            legal.extend(18 + index for index in range(len(player.hand)))
        if player.swaps_remaining > 0:
            legal.extend(28 + index for index in range(max(0, len(player.board) - 1)))
        return sorted(legal)

    def step(self, seat: int, action: int) -> dict:
        if self.done:
            raise RuntimeError("Lobby is complete")
        if seat != self.current_seat:
            raise ValueError(f"Seat {seat} cannot act; current seat is {self.current_seat}")
        action_round = self.round
        decoded = decode_action(action)
        legal = self.legal_actions(seat)
        if action not in legal:
            raise ValueError(f"Illegal action {action}: {decoded}")
        player = self.players[seat]
        events = [
            {
                "event": "action",
                "round": self.round,
                "seat": seat,
                "action": action,
                "kind": decoded.kind,
                "index": decoded.index,
            }
        ]
        if decoded.kind == "end":
            events.extend(self._finish_recruit(seat, forced=False))
        else:
            events.extend(self._apply_action(player, decoded.kind, decoded.index))
            player.actions_remaining = max(0, player.actions_remaining - 1)
            if player.actions_remaining == 0 and not player.pending_discover:
                events.extend(self._finish_recruit(seat, forced=True))
        for event in events:
            event.setdefault("round", action_round)
        for event in events:
            self._trace(event)
        self.assert_conservation()
        return {
            "terminated": self.terminated,
            "truncated": self.truncated,
            "events": deepcopy(events),
        }

    def view(self, seat: int) -> PolicyView:
        if not self._valid_seat(seat):
            raise ValueError("seat must be in range 0..7")
        player = self.players[seat]
        return PolicyView(
            seat=seat,
            round=self.round,
            hp=player.hp,
            gold=player.gold,
            tier=player.tier,
            upgrade_cost=player.upgrade_cost,
            frozen=player.frozen,
            actions_remaining=player.actions_remaining,
            swaps_remaining=player.swaps_remaining,
            hand=tuple(_minion_view(minion) for minion in player.hand),
            board=tuple(_minion_view(minion) for minion in player.board),
            shop=tuple(None if minion is None else _minion_view(minion) for minion in player.shop),
            pending_discover=tuple(_minion_view(minion) for minion in player.pending_discover),
            public_players=tuple(
                PublicPlayerView(
                    seat=public.seat,
                    hp=public.hp,
                    tier=public.tier,
                    alive=public.alive,
                    rank=public.rank,
                )
                for public in self.players
            ),
            legal_actions=tuple(self.legal_actions(seat)),
            cards=self.cards,
        )

    def assert_conservation(self) -> None:
        counts = dict(self.pool.available)
        entity_ids: set[int] = set()
        for player in self.players:
            for minion in player.hand + player.board:
                self._count_persistent(minion, entity_ids, counts, 3 if minion.golden else 1)
            for minion in player.shop:
                if minion is not None:
                    self._count_persistent(minion, entity_ids, counts, 1)
            for minion in player.pending_discover:
                self._count_persistent(minion, entity_ids, counts, 1)
        if counts != self.pool.initial:
            raise AssertionError(
                f"Pool conservation failed: expected {self.pool.initial}, observed {counts}"
            )

    def _apply_action(self, player: PlayerState, kind: str, index: int | None) -> list[dict]:
        if kind == "reroll":
            player.gold -= 1
            self._return_shop(player)
            self._refresh_shop(player)
            return [{"event": "reroll", "seat": player.seat}]
        if kind == "upgrade":
            player.gold -= player.upgrade_cost
            player.tier += 1
            player.upgrade_cost = 0
            return [{"event": "upgrade", "seat": player.seat, "tier": player.tier}]
        if kind == "freeze":
            player.frozen = not player.frozen
            return [{"event": "freeze", "seat": player.seat, "frozen": player.frozen}]
        if kind == "buy":
            assert index is not None
            minion = player.shop[index]
            if minion is None:
                raise AssertionError("legal action allowed an empty shop slot")
            player.gold -= 3
            player.hand.append(minion)
            player.shop[index] = None
            return [{"event": "buy", "seat": player.seat, "card_id": minion.card_id}, *self._triples(player)]
        if kind == "sell":
            assert index is not None
            minion = player.board.pop(index)
            self._return_owned_minion(minion)
            player.gold = min(10, player.gold + 1)
            return [{"event": "sell", "seat": player.seat, "card_id": minion.card_id}]
        if kind == "play":
            assert index is not None
            minion = player.hand.pop(index)
            player.board.append(minion)
            events = [{"event": "play", "seat": player.seat, "card_id": minion.card_id}]
            if minion.golden:
                events.extend(self._open_discover(player))
            events.extend(self._triples(player))
            return events
        if kind == "swap":
            assert index is not None
            player.board[index], player.board[index + 1] = player.board[index + 1], player.board[index]
            player.swaps_remaining -= 1
            return [{"event": "swap", "seat": player.seat, "index": index}]
        if kind == "discover":
            assert index is not None
            chosen = player.pending_discover[index]
            returned = [minion.card_id for i, minion in enumerate(player.pending_discover) if i != index]
            self.pool.return_cards(returned)
            player.pending_discover = []
            player.hand.append(chosen)
            return [
                {
                    "event": "discover",
                    "seat": player.seat,
                    "card_id": chosen.card_id,
                    "returned": returned,
                },
                *self._triples(player),
            ]
        raise AssertionError(f"unhandled action kind: {kind}")

    def _finish_recruit(self, seat: int, forced: bool) -> list[dict]:
        player = self.players[seat]
        player.ready = True
        events = [{"event": "recruit_end", "round": self.round, "seat": seat, "forced": forced}]
        if all(not player.alive or player.ready for player in self.players):
            previous_current = self.current_seat
            self.current_seat = None
            try:
                events.extend(self._resolve_combat_round())
            except Exception:
                player.ready = False
                self.current_seat = previous_current
                raise
            return events
        self._round_index += 1
        while self._round_index < len(self._round_order):
            next_seat = self._round_order[self._round_index]
            if self.players[next_seat].alive:
                self.current_seat = next_seat
                events.extend(self._begin_recruit(next_seat))
                return events
            self._round_index += 1
        raise AssertionError("round order ended before all alive players were ready")

    def _resolve_combat_round(self) -> list[dict]:
        alive = [player.seat for player in self.players if player.alive]
        if len(alive) <= 1:
            self._finish_lobby(alive)
            return [{"event": "lobby_finished", "round": self.round, "alive": alive}]
        precombat = {
            seat: {
                "tier": self.players[seat].tier,
                "board": deepcopy(self.players[seat].board),
                "board_snapshot": self._board_snapshot(self.players[seat].board),
            }
            for seat in alive
        }
        match_rng_state = self._match_rng.getstate()
        combat_rng_state = self._combat_rng.getstate()
        try:
            pairs, cost = match_players(
                alive,
                self._matching_history,
                self._ghost_counts,
                self._last_ghost,
                self._match_rng,
            )
            results = []
            for pair_index, (left, right) in enumerate(pairs):
                if right is None:
                    enemy_board = deepcopy(self._ghost_board)
                    enemy_tier = self._ghost_tier
                    enemy_seat = self._ghost_source_seat
                else:
                    enemy_board = precombat[right]["board"]
                    enemy_tier = precombat[right]["tier"]
                    enemy_seat = right
                combat = simulate(
                    precombat[left]["board"],
                    enemy_board,
                    self._combat_rng,
                    self.catalog,
                    player_tier=precombat[left]["tier"],
                    enemy_tier=enemy_tier,
                )
                results.append((pair_index, left, right, enemy_seat, combat))
        except Exception:
            self._match_rng.setstate(match_rng_state)
            self._combat_rng.setstate(combat_rng_state)
            raise
        self._matching_history.append(pairs)
        ghost_this_round = None
        for left, right in pairs:
            if right is None:
                self._ghost_counts[left] = self._ghost_counts.get(left, 0) + 1
                ghost_this_round = left
        self._last_ghost = ghost_this_round
        events: list[dict] = [
            {
                "event": "matching",
                "round": self.round,
                "pairs": [list(pair) for pair in pairs],
                "cost": list(cost),
            }
        ]
        for pair_index, left, right, enemy_seat, combat in results:
            if right is None:
                self.players[left].hp -= combat.damage_taken
            else:
                self.players[left].hp -= combat.damage_taken
                self.players[right].hp -= combat.damage_dealt
            events.append(self._combat_event(pair_index, left, right, enemy_seat, combat))
        eliminated = [seat for seat in alive if self.players[seat].hp <= 0]
        survivors = [seat for seat in alive if self.players[seat].hp > 0]
        if eliminated:
            rank = (len(survivors) + 1 + len(alive)) / 2
            for seat in eliminated:
                self.players[seat].alive = False
                self.players[seat].rank = rank
                self._return_all_player_cards(self.players[seat])
            ghost_seat = min(eliminated)
            self._ghost_board = precombat[ghost_seat]["board"]
            self._ghost_tier = precombat[ghost_seat]["tier"]
            self._ghost_source_seat = ghost_seat
            self._ghost_death_round = self.round
            events.append(
                {
                    "event": "eliminations",
                    "round": self.round,
                    "seats": eliminated,
                    "rank": rank,
                    "ghost_source": ghost_seat,
                    "ghost_board": precombat[ghost_seat]["board_snapshot"],
                }
            )
        if len(survivors) == 1:
            self.players[survivors[0]].rank = 1.0
            self._finish_lobby(survivors)
            events.append({"event": "lobby_finished", "round": self.round, "alive": survivors})
        elif len(survivors) == 0:
            self._finish_lobby(survivors)
            events.append({"event": "lobby_finished", "round": self.round, "alive": []})
        elif self.round >= self.max_rounds:
            self.truncated = True
            self.done = True
            self.current_seat = None
            events.append({"event": "lobby_truncated", "round": self.round, "alive": survivors})
        else:
            self.round += 1
            self._begin_round(events)
        return events

    def _begin_round(self, events: list[dict] | None = None) -> None:
        alive = [player.seat for player in self.players if player.alive]
        start = (self.round - 1) % PLAYER_COUNT
        self._round_order = [seat for offset in range(PLAYER_COUNT) if (seat := (start + offset) % PLAYER_COUNT) in alive]
        self._round_index = 0
        for player in self.players:
            player.ready = not player.alive
        if not self._round_order:
            self._finish_lobby([])
            return
        self.current_seat = self._round_order[0]
        recruit_events = self._begin_recruit(self.current_seat)
        if events is None:
            for event in recruit_events:
                self._trace(event)
        else:
            events.extend(recruit_events)

    def _begin_recruit(self, seat: int) -> list[dict]:
        player = self.players[seat]
        player.ready = False
        player.gold = min(10, self.round + 2)
        player.actions_remaining = MAX_ACTIONS
        player.swaps_remaining = MAX_SWAPS
        if player.tier < MAX_TIER:
            player.upgrade_cost = max(0, 6 - self.round)
        if player.frozen:
            player.frozen = False
        else:
            self._return_shop(player)
            self._refresh_shop(player)
        return [
            {
                "event": "recruit_start",
                "round": self.round,
                "seat": seat,
                "gold": player.gold,
                "tier": player.tier,
                "shop": self._shop_snapshot(player.shop),
            }
        ]

    def _refresh_shop(self, player: PlayerState) -> None:
        count = 3 if player.tier == 1 else 4
        card_ids = self.pool.draw(self._shop_rngs[player.seat], player.tier, count)
        player.shop = [self._spawn(card_id) for card_id in card_ids]
        player.shop.extend([None] * (SHOP_SIZE - len(player.shop)))

    def _return_shop(self, player: PlayerState) -> None:
        self.pool.return_cards([minion.card_id for minion in player.shop if minion is not None])
        player.shop = [None] * SHOP_SIZE

    def _open_discover(self, player: PlayerState) -> list[dict]:
        tier = min(MAX_TIER, player.tier + 1)
        card_ids = self.pool.draw(self._shop_rngs[player.seat], tier, 3, exact=True)
        player.pending_discover = [self._spawn(card_id) for card_id in card_ids]
        if not player.pending_discover:
            return [{"event": "discover_exhausted", "seat": player.seat, "tier": tier}]
        return [
            {
                "event": "discover_start",
                "seat": player.seat,
                "tier": tier,
                "candidates": [minion.card_id for minion in player.pending_discover],
            }
        ]

    def _triples(self, player: PlayerState) -> list[dict]:
        events = []
        changed = True
        while changed:
            changed = False
            for card in self.catalog.cards:
                if not card.triple_allowed:
                    continue
                copies = [
                    minion
                    for minion in player.hand + player.board
                    if minion.card_id == card.dbf_id and not minion.golden
                ]
                if len(copies) < 3:
                    continue
                chosen = copies[:3]
                remaining_hand_size = len(player.hand) - sum(minion in chosen for minion in player.hand)
                if remaining_hand_size >= HAND_SIZE:
                    continue
                golden = self._spawn(card.dbf_id, golden=True)
                golden.attack += sum(minion.attack - card.attack for minion in chosen)
                golden.health += sum(minion.health - card.health for minion in chosen)
                golden.taunt = any(minion.taunt for minion in chosen)
                golden.divine_shield = any(minion.divine_shield for minion in chosen)
                player.hand = [minion for minion in player.hand if minion not in chosen]
                player.board = [minion for minion in player.board if minion not in chosen]
                player.hand.append(golden)
                events.append(
                    {
                        "event": "triple",
                        "seat": player.seat,
                        "card_id": card.dbf_id,
                        "golden_entity_id": golden.entity_id,
                    }
                )
                changed = True
                break
        return events

    def _spawn(self, card_id: int, golden: bool = False) -> Minion:
        self._entity_id += 1
        return self.catalog.spawn(card_id, self._entity_id, golden=golden)

    def _return_owned_minion(self, minion: Minion) -> None:
        self.pool.return_cards([minion.card_id] * (3 if minion.golden else 1))

    def _return_all_player_cards(self, player: PlayerState) -> None:
        returned = []
        for minion in player.hand + player.board:
            returned.extend([minion.card_id] * (3 if minion.golden else 1))
        returned.extend(minion.card_id for minion in player.shop if minion is not None)
        returned.extend(minion.card_id for minion in player.pending_discover)
        self.pool.return_cards(returned)
        player.hand = []
        player.board = []
        player.shop = [None] * SHOP_SIZE
        player.pending_discover = []

    def _finish_lobby(self, alive: list[int]) -> None:
        self.terminated = True
        self.truncated = False
        self.done = True
        self.current_seat = None
        for seat in alive:
            if self.players[seat].rank is None and len(alive) == 1:
                self.players[seat].rank = 1.0

    def _count_persistent(
        self, minion: Minion, entity_ids: set[int], counts: dict[int, int], amount: int
    ) -> None:
        if minion.entity_id in entity_ids:
            raise AssertionError(f"Duplicate persistent entity ID: {minion.entity_id}")
        if minion.entity_id <= 0:
            raise AssertionError(f"Persistent entity IDs must be positive: {minion.entity_id}")
        entity_ids.add(minion.entity_id)
        if minion.card_id not in self.pool.initial:
            raise AssertionError(f"Token or non-pool card is persistent: {minion.card_id}")
        counts[minion.card_id] = counts.get(minion.card_id, 0) + amount

    def _combat_event(
        self,
        pair_index: int,
        left: int,
        right: int | None,
        enemy_seat: int | None,
        combat: CombatResult,
    ) -> dict:
        return {
            "event": "combat",
            "round": self.round,
            "pair_index": pair_index,
            "left": left,
            "right": right,
            "enemy_seat": enemy_seat,
            "result": combat.result,
            "damage_taken": combat.damage_taken,
            "damage_dealt": combat.damage_dealt,
            "reason": combat.reason,
            "attacks": list(combat.attacks),
            "deaths": list(combat.deaths),
            "summons": list(combat.summons),
        }

    def _trace(self, event: dict) -> None:
        self.trace.append(deepcopy(event))

    def _shop_snapshot(self, shop: list[Minion | None]) -> list[dict | None]:
        return [None if minion is None else self._minion_snapshot(minion) for minion in shop]

    def _board_snapshot(self, board: list[Minion]) -> list[dict]:
        return [self._minion_snapshot(minion) for minion in board]

    @staticmethod
    def _minion_snapshot(minion: Minion) -> dict:
        return {
            "entity_id": minion.entity_id,
            "card_id": minion.card_id,
            "attack": minion.attack,
            "health": minion.health,
            "golden": minion.golden,
            "taunt": minion.taunt,
            "divine_shield": minion.divine_shield,
        }

    @staticmethod
    def _valid_seat(seat: int) -> bool:
        return type(seat) is int and 0 <= seat < PLAYER_COUNT


def _minion_view(minion: Minion) -> MinionView:
    return MinionView(
        card_id=minion.card_id,
        attack=minion.attack,
        health=minion.health,
        golden=minion.golden,
        taunt=minion.taunt,
        divine_shield=minion.divine_shield,
    )
