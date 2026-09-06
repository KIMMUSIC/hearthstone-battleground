"""Gymnasium boundary. The engine itself does not depend on Gymnasium or NumPy."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .game import Game


class BgEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, *, opponents=None, opponent_mode="fixed-v1", max_turns=8, max_actions=24):
        super().__init__()
        self.game = Game(
            opponents=opponents,
            opponent_mode=opponent_mode,
            max_turns=max_turns,
            max_actions=max_actions,
        )
        self.max_turns = max_turns
        self.max_actions = max_actions
        self.opponent_mode = opponent_mode
        self.opponent_spec = self.game.opponent_spec
        self.slot_width = max(self.game.catalog.mapping.values()) + 1 + 6
        self.action_space = spaces.Discrete(37)
        # Signed HP can occur after lethal combat. Finite values checked independently.
        self.observation_space = spaces.Dict(
            {
                "player": spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32),
                "shop": spaces.Box(0, np.inf, shape=(7, self.slot_width), dtype=np.float32),
                "board": spaces.Box(0, np.inf, shape=(7, self.slot_width), dtype=np.float32),
                "hand": spaces.Box(0, np.inf, shape=(10, self.slot_width), dtype=np.float32),
                "discover": spaces.Box(0, np.inf, shape=(3, self.slot_width), dtype=np.float32),
            }
        )

    def _slots(self, minions, capacity, *, definitions=False):
        if len(minions) > capacity:
            raise ValueError("Engine collection exceeds observation capacity")
        result = np.zeros((capacity, self.slot_width), dtype=np.float32)
        vocab = self.slot_width - 6
        result[:, 0] = 1  # Explicit empty-slot identity, distinct from every card.
        for position, minion in enumerate(minions):
            if minion is None:
                continue
            card_id = minion.dbf_id if definitions else minion.card_id
            index = self.game.catalog.mapping[card_id]  # Unknown is an error, never padding.
            result[position, 0] = 0
            result[position, index] = 1
            result[position, vocab:] = [
                minion.attack / 100,
                minion.health / 100,
                0 if definitions else float(minion.golden),
                float(minion.taunt),
                float(minion.divine_shield),
                self.game.catalog.by_id[card_id].tier / 2,
            ]
        return result

    def observation(self):
        game = self.game
        obs = {
            "player": np.array(
                [
                    game.hp / 40,
                    game.gold / 10,
                    game.tier / 2,
                    game.turn / self.max_turns,
                    game.upgrade_cost / 5,
                    float(game.frozen),
                    game.actions_remaining / self.max_actions,
                    game.swaps_remaining / 3,
                    float(bool(game.pending_discover)),
                ],
                dtype=np.float32,
            ),
            "shop": self._slots(game.shop, 7),
            "board": self._slots(game.board, 7),
            "hand": self._slots(game.hand, 10),
            "discover": self._slots(game.pending_discover, 3, definitions=True),
        }
        if not all(np.isfinite(value).all() for value in obs.values()):
            raise ValueError("Non-finite observation")
        if not self.observation_space.contains(obs):
            raise ValueError("Observation outside declared space")
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        actual_seed = int(self.np_random.integers(0, 2**32)) if seed is None else int(seed)
        self.game.reset(seed=actual_seed)
        return self.observation(), {"seed": actual_seed}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action!r}")
        result = self.game.step(int(action))
        return self.observation(), result.reward, result.terminated, result.truncated, result.info

    def action_masks(self):
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[self.game.legal_actions()] = True
        return mask
