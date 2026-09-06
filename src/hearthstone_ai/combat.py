"""Seeded combat for attack, health, taunt, divine shield and explicit summons."""

from dataclasses import dataclass, replace
import random

from .cards import Catalog, Minion


@dataclass(frozen=True)
class CombatResult:
    result: int
    damage_taken: int
    damage_dealt: int
    attacks: tuple[dict, ...]
    reason: str
    deaths: tuple[dict, ...] = ()
    summons: tuple[dict, ...] = ()


def simulate(
    player: list[Minion],
    enemy: list[Minion],
    rng: random.Random,
    catalog: Catalog,
    player_tier: int = 1,
    enemy_tier: int = 1,
) -> CombatResult:
    """Do not mutate inputs; resolved deaths keep the attack ring stable.

    Zero-attack living minions are skipped as attackers but still participate in
    targeting and hero damage.
    """
    boards = [[replace(minion) for minion in board] for board in (player, enemy)]
    next_entity_id = max((m.entity_id for board in boards for m in board), default=0) + 1
    for board in boards:
        if len(board) > 7:
            raise ValueError("Combat boards cannot exceed seven minions")
        if len({m.entity_id for m in board}) != len(board):
            raise ValueError("Entity IDs must be unique within each board")
        for minion in board:
            card = catalog.by_id.get(minion.card_id)
            if card is None:
                raise ValueError(f"Unknown card ID: {minion.card_id}")
            if minion.golden and card.deathrattle_token_id is not None:
                raise ValueError("Golden deathrattle minions are not supported")
            if minion.attack < 0 or minion.health <= 0:
                raise ValueError("Combat inputs must be alive with nonnegative attack")
    if player_tier < 1 or enemy_tier < 1:
        raise ValueError("Tavern tiers must be positive")
    cursors = [0, 0]
    events = []
    death_events = []
    summon_events = []
    side = (0 if len(player) > len(enemy) else 1) if len(player) != len(enemy) else rng.randrange(2)
    reason = "elimination"

    def alive(index):
        return [minion for minion in boards[index] if minion.health > 0]

    def damage(target, amount):
        if amount <= 0:
            return False
        if target.divine_shield:
            target.divine_shield = False
            return True
        target.health -= amount
        return False

    def side_name(index):
        return "player" if index == 0 else "enemy"

    def resolve_deaths(attacking_side):
        nonlocal next_entity_id
        deaths_by_side = []
        for board in boards:
            deaths_by_side.append(
                [
                    (position, minion)
                    for position, minion in enumerate(board)
                    if minion.health <= 0
                ]
            )
        if not any(deaths_by_side):
            return
        old_cursors = cursors[:]
        ordered_sides = (attacking_side, 1 - attacking_side)
        summons_before_cursor = [0, 0]
        for board_side in ordered_sides:
            for position, minion in deaths_by_side[board_side]:
                death_events.append(
                    dict(
                        side=side_name(board_side),
                        entity_id=minion.entity_id,
                        card_id=minion.card_id,
                        position=position,
                    )
                )
        for board_side, board in enumerate(boards):
            if not deaths_by_side[board_side]:
                continue
            dead_positions = {position for position, _ in deaths_by_side[board_side]}
            boards[board_side] = [
                minion for position, minion in enumerate(board) if position not in dead_positions
            ]
        for board_side in ordered_sides:
            inserted_count = 0
            for death_index, (position, minion) in enumerate(deaths_by_side[board_side]):
                card = catalog.by_id[minion.card_id]
                if card.deathrattle_token_id is None:
                    continue
                insertion_index = position - death_index + inserted_count
                omitted = 0
                for _ in range(card.deathrattle_count):
                    if len(boards[board_side]) >= 7:
                        omitted += 1
                        continue
                    token = catalog.spawn(card.deathrattle_token_id, next_entity_id)
                    next_entity_id += 1
                    insertion_index = min(insertion_index, len(boards[board_side]))
                    boards[board_side].insert(insertion_index, token)
                    summon_events.append(
                        dict(
                            event="summon",
                            side=side_name(board_side),
                            source=minion.entity_id,
                            entity_id=token.entity_id,
                            card_id=token.card_id,
                            position=insertion_index,
                        )
                    )
                    insertion_index += 1
                    inserted_count += 1
                    summons_before_cursor[board_side] += int(position < old_cursors[board_side])
                if omitted:
                    summon_events.append(
                        dict(
                            event="summon_omitted",
                            side=side_name(board_side),
                            source=minion.entity_id,
                            card_id=card.deathrattle_token_id,
                            omitted=omitted,
                        )
                    )
        for board_side, board in enumerate(boards):
            if not board:
                cursors[board_side] = 0
                continue
            removed_before_cursor = sum(
                1 for position, _ in deaths_by_side[board_side] if position < old_cursors[board_side]
            )
            cursors[board_side] = (
                old_cursors[board_side] - removed_before_cursor + summons_before_cursor[board_side]
            ) % len(board)

    while alive(0) and alive(1):
        if not any(minion.attack > 0 for index in (0, 1) for minion in alive(index)):
            reason = "stalemate"
            break
        if len(events) >= 10000:
            reason = "combat_limit"
            break
        attacker = None
        for _ in range(len(boards[side])):
            candidate = boards[side][cursors[side]]
            cursors[side] = (cursors[side] + 1) % len(boards[side])
            if candidate.health > 0 and candidate.attack > 0:
                attacker = candidate
                break
        if attacker is None:
            side = 1 - side
            continue
        targets = alive(1 - side)
        defender = rng.choice([m for m in targets if m.taunt] or targets)
        defender_shield = damage(defender, attacker.attack)
        attacker_shield = damage(attacker, defender.attack)
        events.append(
            dict(
                side=side_name(side),
                attacker=attacker.entity_id,
                defender=defender.entity_id,
                attacker_health=attacker.health,
                defender_health=defender.health,
                attacker_shield_broken=attacker_shield,
                defender_shield_broken=defender_shield,
            )
        )
        resolve_deaths(side)
        side = 1 - side
    survivors = [alive(0), alive(1)]
    result = int(bool(survivors[0])) - int(bool(survivors[1]))
    dealt = (
        player_tier + sum(catalog.by_id[m.card_id].tier for m in survivors[0]) if result == 1 else 0
    )
    taken = (
        enemy_tier + sum(catalog.by_id[m.card_id].tier for m in survivors[1]) if result == -1 else 0
    )
    return CombatResult(result, taken, dealt, tuple(events), reason, tuple(death_events), tuple(summon_events))
