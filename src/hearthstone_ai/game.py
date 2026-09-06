"""Bounded custom training rules; not a complete retail Battlegrounds engine."""

import random
from dataclasses import dataclass

from .actions import decode_action
from .cards import Card, Catalog
from .combat import simulate

OPPONENT_MODE_FIXED = "fixed-v1"
OPPONENT_MODE_DIVERSE = "diverse-v1"
OPPONENT_MODES = frozenset({OPPONENT_MODE_FIXED, OPPONENT_MODE_DIVERSE})
OPPONENT_TURN_COUNTS = (1, 1, 2, 2, 3, 3, 4, 4)
FIXED_OPPONENT_CARD_ID = 72387
DIVERSE_OPPONENT_CARD_IDS = (42467, 72387)


@dataclass(frozen=True)
class StepResult:
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class Game:
    def __init__(
        self,
        catalog=None,
        max_turns=8,
        max_actions=24,
        opponents=None,
        opponent_mode=OPPONENT_MODE_FIXED,
    ):
        if type(max_turns) is not int or max_turns < 1:
            raise ValueError("max_turns must be a positive integer")
        if type(max_actions) is not int or max_actions < 1:
            raise ValueError("max_actions must be a positive integer")
        if not isinstance(opponent_mode, str) or opponent_mode not in OPPONENT_MODES:
            raise ValueError(f"opponent_mode must be one of {sorted(OPPONENT_MODES)}")
        if opponents is not None and opponent_mode != OPPONENT_MODE_FIXED:
            raise ValueError("Explicit opponents cannot be combined with diverse opponent mode")
        self.catalog = catalog if catalog is not None else Catalog()
        self.max_turns = max_turns
        self.max_actions = max_actions
        self.opponent_mode = opponent_mode
        if opponents is None:
            self.opponent_spec = opponent_distribution_spec(self.catalog, opponent_mode)
            opponents = _fixed_opponents()
        else:
            self.opponent_spec = _explicit_opponent_spec(opponents)
        if not opponents:
            raise ValueError("Opponent templates cannot be empty")
        self._base_opponents = _validate_opponents(self.catalog, opponents)
        self._opponents = self._base_opponents
        self.reset()

    def reset(self, seed: int = 0):
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.seed = seed
        self._opponents = self._opponents_for_seed(seed)
        self._shop_rng = random.Random(f"shop:{seed}")
        self._combat_rng = random.Random(f"combat:{seed}")
        self._entity_id = 0
        self.hp, self.gold, self.tier, self.turn = 30, 3, 1, 1
        self.upgrade_cost = 5
        self.frozen = False
        self.actions_remaining = self.max_actions
        self.swaps_remaining = 3
        self.board, self.hand, self.pending_discover = [], [], []
        self.done = False
        self.termination_reason = None
        self._refresh_shop()
        return self

    def _spawn(self, card_id, golden=False):
        self._entity_id += 1
        return self.catalog.spawn(card_id, self._entity_id, golden=golden)

    def _opponents_for_seed(self, seed: int):
        if self.opponent_mode == OPPONENT_MODE_FIXED:
            return self._base_opponents
        rng = random.Random(f"opponent:{seed}")
        return tuple(
            (
                1,
                tuple(rng.choice(DIVERSE_OPPONENT_CARD_IDS) for _ in range(count)),
            )
            for count in OPPONENT_TURN_COUNTS
        )

    def _refresh_shop(self):
        count = 3 if self.tier == 1 else 4
        self.shop = [
            self._spawn(card.dbf_id)
            for card in self.catalog.sample(self._shop_rng, self.tier, count)
        ]
        self.shop += [None] * (7 - count)

    def legal_actions(self) -> list[int]:
        if self.done:
            return []
        if self.pending_discover:
            return (
                [34 + i for i in range(len(self.pending_discover))] if len(self.hand) < 10 else []
            )
        legal = [0, 3]
        if self.gold >= 1:
            legal.append(1)
        if self.tier < 2 and self.gold >= self.upgrade_cost:
            legal.append(2)
        if self.gold >= 3 and len(self.hand) < 10:
            legal.extend(4 + i for i, card in enumerate(self.shop) if card is not None)
        legal.extend(11 + i for i in range(len(self.board)))
        if len(self.board) < 7:
            legal.extend(18 + i for i in range(len(self.hand)))
        if self.swaps_remaining:
            legal.extend(28 + i for i in range(max(0, len(self.board) - 1)))
        return sorted(legal)

    def step(self, action: int) -> StepResult:
        if self.done:
            raise RuntimeError("Episode is complete; call reset before step")
        decoded = decode_action(action)
        if action not in self.legal_actions():
            raise ValueError(f"Illegal action {action}: {decoded}")
        kind, index = decoded.kind, decoded.index
        if kind == "end":
            return self._end_turn(False)
        if kind == "reroll":
            self.gold -= 1
            self._refresh_shop()
        elif kind == "upgrade":
            self.gold -= self.upgrade_cost
            self.tier += 1
            self.upgrade_cost = 0
        elif kind == "freeze":
            self.frozen = not self.frozen
        elif kind == "buy":
            self.gold -= 3
            self.hand.append(self.shop[index])
            self.shop[index] = None
            self._triples()
        elif kind == "sell":
            self.board.pop(index)
            self.gold = min(10, self.gold + 1)
        elif kind == "play":
            minion = self.hand.pop(index)
            self.board.append(minion)
            if minion.golden:
                self.pending_discover = self.catalog.sample(
                    self._shop_rng, min(2, self.tier + 1), 3, exact=True
                )
            self._triples()
        elif kind == "swap":
            self.board[index], self.board[index + 1] = self.board[index + 1], self.board[index]
            self.swaps_remaining -= 1
        elif kind == "discover":
            self.hand.append(self._spawn(self.pending_discover[index].dbf_id))
            self.pending_discover = []
            self._triples()
        # A modal choice resolves even when the triggering play used the last action.
        # This avoids losing or silently choosing an earned discover reward.
        self.actions_remaining = max(0, self.actions_remaining - 1)
        if not self.actions_remaining and not self.pending_discover:
            return self._end_turn(True)
        return StepResult(0.0, False, False, {"action": action})

    def _triples(self):
        for card in self.catalog.cards:
            if not card.triple_allowed:
                continue
            copies = [
                m for m in self.hand + self.board if m.card_id == card.dbf_id and not m.golden
            ]
            while len(copies) >= 3:
                chosen, copies = copies[:3], copies[3:]
                # Removing three copies must leave a slot for the golden result.
                if len(self.hand) - sum(any(m is c for c in chosen) for m in self.hand) >= 10:
                    break
                golden = self._spawn(card.dbf_id, golden=True)
                golden.attack += sum(m.attack - card.attack for m in chosen)
                golden.health += sum(m.health - card.health for m in chosen)
                golden.taunt = any(m.taunt for m in chosen)
                golden.divine_shield = any(m.divine_shield for m in chosen)
                self.hand = [m for m in self.hand if not any(m is c for c in chosen)]
                self.board = [m for m in self.board if not any(m is c for c in chosen)]
                self.hand.append(golden)

    def _end_turn(self, forced):
        enemy_tier, enemy_ids = self._opponents[(self.turn - 1) % len(self._opponents)]
        enemy = [self.catalog.spawn(card_id, -(i + 1)) for i, card_id in enumerate(enemy_ids)]
        combat = simulate(
            self.board,
            enemy,
            self._combat_rng,
            self.catalog,
            player_tier=self.tier,
            enemy_tier=enemy_tier,
        )
        self.hp -= combat.damage_taken
        info = {
            "combat_result": combat.result,
            "damage_taken": combat.damage_taken,
            "damage_dealt": combat.damage_dealt,
            "attacks": combat.attacks,
            "combat_reason": combat.reason,
            "forced_end_turn": forced,
            "turn": self.turn,
            "opponent_mode": self.opponent_mode,
            "opponent_tier": enemy_tier,
            "opponent_board": list(enemy_ids),
        }
        if self.hp <= 0 or self.turn >= self.max_turns:
            self.done = True
            self.termination_reason = "death" if self.hp <= 0 else "horizon"
            info["termination_reason"] = self.termination_reason
        else:
            self.turn += 1
            self.gold = min(10, self.turn + 2)
            if self.tier < 2:
                self.upgrade_cost = max(0, self.upgrade_cost - 1)
            self.actions_remaining = self.max_actions
            self.swaps_remaining = 3
            if not self.frozen:
                self._refresh_shop()
            self.frozen = False
        return StepResult(float(combat.result), self.done, False, info)


def opponent_distribution_spec(catalog: Catalog | None = None, mode: str = OPPONENT_MODE_FIXED):
    if not isinstance(mode, str) or mode not in OPPONENT_MODES:
        raise ValueError(f"opponent_mode must be one of {sorted(OPPONENT_MODES)}")
    catalog = catalog if catalog is not None else Catalog()
    candidate_ids = (
        (FIXED_OPPONENT_CARD_ID,) if mode == OPPONENT_MODE_FIXED else DIVERSE_OPPONENT_CARD_IDS
    )
    for card_id in candidate_ids:
        try:
            card = catalog.by_id[card_id]
        except KeyError as error:
            raise ValueError(f"Opponent card is missing from catalog: {card_id}") from error
        _require_tier_one_supported(card)
    return {
        "version": mode,
        "turn_counts": list(OPPONENT_TURN_COUNTS),
        "tiers": [1 for _ in OPPONENT_TURN_COUNTS],
        "candidate_card_ids": list(candidate_ids),
        "sampling": "fixed" if mode == OPPONENT_MODE_FIXED else "iid-uniform-with-replacement",
    }


def _require_tier_one_supported(card: Card) -> None:
    if card.tier != 1:
        raise ValueError(f"Opponent card must be tier one: {card.dbf_id}")


def _fixed_opponents():
    return [{"tier": 1, "board": [FIXED_OPPONENT_CARD_ID] * count} for count in OPPONENT_TURN_COUNTS]


def _explicit_opponent_spec(opponents):
    return {
        "version": "explicit-v1",
        "turn_counts": [len(template["board"]) for template in opponents],
        "tiers": [template["tier"] for template in opponents],
        "sampling": "explicit-templates",
    }


def _validate_opponents(catalog: Catalog, opponents):
    checked = []
    for template in opponents:
        tier, board = template["tier"], tuple(template["board"])
        if type(tier) is not int or tier not in (1, 2) or len(board) > 7:
            raise ValueError("Invalid opponent tier or board size")
        for card_id in board:
            card = catalog.by_id[card_id]
            if not card.pool:
                raise ValueError(f"Opponent template card is not in the supported pool: {card_id}")
        checked.append((tier, board))
    return tuple(checked)
