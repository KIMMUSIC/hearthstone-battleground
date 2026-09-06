"""Gymnasium adapter for one learner inside the restricted eight-player lobby."""

from __future__ import annotations

import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .lobby import LobbyGame, PLAYER_COUNT
from .lobby_observation import LobbyObservation, OBSERVATION_SCALES
from .lobby_policies import heuristic_policy, random_policy


OPPONENT_PROTOCOL = "lobby-opponent-mix-v1"


class LobbyEnv(gym.Env):
    """Expose one lobby seat as the learner and drive all other seats by policy."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        learner_seat: int = 0,
        max_rounds: int = 100,
        train_seed_min: int = 0,
        train_seed_max: int = 9999,
        heuristic_opponents: int = 7,
        observation_scale: str = "raw",
    ):
        super().__init__()
        if type(learner_seat) is not int or not 0 <= learner_seat < PLAYER_COUNT:
            raise ValueError("learner_seat must be an integer in 0..7")
        if type(max_rounds) is not int or max_rounds < 1:
            raise ValueError("max_rounds must be a positive integer")
        if type(heuristic_opponents) is not int or not 0 <= heuristic_opponents < PLAYER_COUNT:
            raise ValueError("heuristic_opponents must be an integer in 0..7")
        if observation_scale not in OBSERVATION_SCALES:
            raise ValueError(f"observation_scale must be one of {sorted(OBSERVATION_SCALES)}")
        if (
            type(train_seed_min) is not int
            or type(train_seed_max) is not int
            or train_seed_min < 0
            or train_seed_max < train_seed_min
        ):
            raise ValueError("training seed range must be nonnegative and ordered")
        self.learner_seat = learner_seat
        self.max_rounds = max_rounds
        self.lobby_max_rounds = max_rounds
        self.environment = "lobby"
        self.heuristic_opponents = heuristic_opponents
        self.observation_scale = observation_scale
        self.train_seed_min = train_seed_min
        self.train_seed_max = train_seed_max
        self.encoder = LobbyObservation(observation_scale=observation_scale)
        self.action_space = spaces.Discrete(37)
        self.observation_space = self.encoder.space
        self.game: LobbyGame | None = None
        self.seed: int | None = None
        self._opponent_policies: dict[int, str] = {}
        self._opponent_rngs: dict[int, random.Random] = {}
        self._done = False
        self._reward_paid = False

    def reset(self, *, seed=None, options=None):
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError("Explicit reset seed must be a nonnegative integer")
        super().reset(seed=seed)
        actual_seed = self._seed_from_reset(seed)
        self.seed = actual_seed
        self.game = LobbyGame(seed=actual_seed, max_rounds=self.max_rounds)
        self._configure_opponents(actual_seed)
        self._done = False
        self._reward_paid = False
        self._advance_opponents()
        return self.observation(), self._info("reset")

    def step(self, action):
        if self.game is None:
            raise RuntimeError("Environment must be reset before stepping")
        if self._done:
            raise RuntimeError("Episode is complete")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action!r}")
        action = int(action)
        legal = self.game.legal_actions(self.learner_seat)
        if action not in legal:
            raise ValueError(f"Illegal action {action}")
        self.game.step(self.learner_seat, action)
        self._advance_opponents()
        terminated = self._learner_rank() is not None or self.game.terminated
        truncated = self.game.truncated and self._learner_rank() is None
        reward = self._reward(terminated)
        self._done = bool(terminated or truncated)
        return self.observation(), reward, bool(terminated), bool(truncated), self._info("step")

    def observation(self):
        if self.game is None:
            raise RuntimeError("Environment must be reset before observing")
        return self.encoder.encode(self.game.view(self.learner_seat))

    def action_masks(self):
        if self.game is None:
            return np.zeros(self.action_space.n, dtype=bool)
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[self.game.legal_actions(self.learner_seat)] = True
        return mask

    @property
    def opponent_spec(self):
        opponent_seats = [seat for seat in range(PLAYER_COUNT) if seat != self.learner_seat]
        heuristic_seats = opponent_seats[: self.heuristic_opponents]
        random_seats = opponent_seats[self.heuristic_opponents :]
        return {
            "version": OPPONENT_PROTOCOL,
            "learner_seat": self.learner_seat,
            "heuristic_opponents": self.heuristic_opponents,
            "opponent_count": PLAYER_COUNT - 1,
            "ordering": "ascending-seat-excluding-learner",
            "heuristic_policy": "heuristic_policy",
            "random_policy": "random_policy",
            "random_seed_protocol": "lobby-probe:{seed}:policy:{seat}",
            "opponent_seats": opponent_seats,
            "heuristic_seats": heuristic_seats,
            "random_seats": random_seats,
        }

    def _configure_opponents(self, seed: int) -> None:
        opponent_seats = [seat for seat in range(PLAYER_COUNT) if seat != self.learner_seat]
        heuristic_seats = set(opponent_seats[: self.heuristic_opponents])
        self._opponent_policies = {
            seat: "heuristic" if seat in heuristic_seats else "random" for seat in opponent_seats
        }
        self._opponent_rngs = {
            seat: random.Random(f"lobby-probe:{seed}:policy:{seat}")
            for seat in opponent_seats
            if seat not in heuristic_seats
        }

    def _seed_from_reset(self, seed):
        if seed is not None:
            if type(seed) is not int or seed < 0:
                raise ValueError("Explicit reset seed must be a nonnegative integer")
            return seed
        return int(self.np_random.integers(self.train_seed_min, self.train_seed_max + 1))

    def _advance_opponents(self):
        assert self.game is not None
        guard = 0
        while (
            not self.game.done
            and self.game.current_seat is not None
            and self.game.current_seat != self.learner_seat
            and self._learner_rank() is None
        ):
            seat = self.game.current_seat
            view = self.game.view(seat)
            if self._opponent_policies[seat] == "heuristic":
                action = heuristic_policy(view)
            else:
                action = random_policy(view, self._opponent_rngs[seat])
            self.game.step(seat, action)
            guard += 1
            if guard > PLAYER_COUNT * self.max_rounds * 30:
                raise RuntimeError("Opponent auto-play exceeded the lobby action bound")

    def _learner_rank(self):
        assert self.game is not None
        return self.game.players[self.learner_seat].rank

    def _reward(self, terminated):
        if not terminated or self._reward_paid:
            return 0.0
        rank = self._learner_rank()
        self._reward_paid = True
        return 0.0 if rank is None else float((4.5 - rank) / 3.5)

    def _info(self, reason):
        assert self.game is not None
        rank = self._learner_rank()
        if self.game.truncated and rank is None:
            termination_reason = "max_rounds"
        elif rank is not None:
            termination_reason = "learner_ranked"
        elif self.game.terminated:
            termination_reason = "lobby_finished"
        else:
            termination_reason = None
        return {
            "seed": self.seed,
            "round": self.game.round,
            "rank": rank,
            "learner_alive": self.game.players[self.learner_seat].alive,
            "termination_reason": termination_reason,
            "adapter_event": reason,
        }
