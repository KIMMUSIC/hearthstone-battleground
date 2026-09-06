"""Numeric observations from immutable lobby policy views; no training adapter."""

import numpy as np
from gymnasium import spaces

from .cards import Catalog


OBSERVATION_SCALE_RAW = "raw"
OBSERVATION_SCALE_FIXED_V1 = "fixed-v1"
OBSERVATION_SCALES = {OBSERVATION_SCALE_RAW, OBSERVATION_SCALE_FIXED_V1}
PLAYER_FIXED_V1_SCALE = np.array([7, 100, 30, 10, 2, 5, 1, 24, 3], dtype=np.float32)
PUBLIC_FIXED_V1_SCALE = np.array([30, 2, 1, 8], dtype=np.float32)


class LobbyObservation:
    version = "lobby-own-public-v1"

    def __init__(self, catalog=None, *, observation_scale: str = OBSERVATION_SCALE_RAW):
        if observation_scale not in OBSERVATION_SCALES:
            raise ValueError(f"observation_scale must be one of {sorted(OBSERVATION_SCALES)}")
        catalog = catalog or Catalog()
        self.observation_scale = observation_scale
        self.mapping = dict(catalog.mapping)
        self.tiers = {card.dbf_id: card.tier for card in catalog.cards}
        self.width = max(self.mapping.values()) + 7
        self.space = spaces.Dict({
            "player": spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32),
            "public_players": spaces.Box(-np.inf, np.inf, shape=(8, 4), dtype=np.float32),
            **{name: spaces.Box(0, np.inf, shape=(capacity, self.width), dtype=np.float32)
               for name, capacity in (("shop", 7), ("board", 7), ("hand", 10), ("discover", 3))},
            "action_mask": spaces.MultiBinary(37),
        })

    def _slots(self, minions, capacity):
        if len(minions) > capacity:
            raise ValueError("Collection exceeds declared capacity")
        result = np.zeros((capacity, self.width), dtype=np.float32)
        result[:, 0] = 1
        for position, minion in enumerate(minions):
            if minion is None:
                continue
            index = self.mapping[minion.card_id]
            tier = self.tiers[minion.card_id]
            result[position, 0] = 0
            result[position, index] = 1
            result[position, -6:] = [minion.attack / 100, minion.health / 100,
                                     float(minion.golden), float(minion.taunt),
                                     float(minion.divine_shield), tier / 2]
        return result

    def encode(self, view):
        if len(view.public_players) != 8 or tuple(p.seat for p in view.public_players) != tuple(range(8)):
            raise ValueError("Exactly eight public seats in order required")
        mask = np.zeros(37, dtype=np.int8)
        for action in view.legal_actions:
            if type(action) is not int or not 0 <= action < 37:
                raise ValueError("Invalid legal action")
            mask[action] = 1
        observation = {
            "player": self._scale_player(np.array(
                [view.seat, view.round, view.hp, view.gold, view.tier,
                 view.upgrade_cost, float(view.frozen), view.actions_remaining,
                 view.swaps_remaining],
                dtype=np.float32,
            )),
            "public_players": self._scale_public(np.array(
                [[p.hp, p.tier, float(p.alive), 0 if p.rank is None else p.rank]
                 for p in view.public_players],
                dtype=np.float32,
            )),
            "shop": self._slots(view.shop, 7), "board": self._slots(view.board, 7),
            "hand": self._slots(view.hand, 10), "discover": self._slots(view.pending_discover, 3),
            "action_mask": mask,
        }
        if not all(np.isfinite(value).all() for value in observation.values()) or not self.space.contains(observation):
            raise ValueError("Observation outside declared finite space")
        return observation

    def _scale_player(self, values):
        if self.observation_scale == OBSERVATION_SCALE_FIXED_V1:
            return values / PLAYER_FIXED_V1_SCALE
        return values

    def _scale_public(self, values):
        if self.observation_scale == OBSERVATION_SCALE_FIXED_V1:
            return values / PUBLIC_FIXED_V1_SCALE
        return values
