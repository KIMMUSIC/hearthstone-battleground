import json

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from hearthstone_ai import lobby_training


class TinyLobbyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        learner_seat=0,
        max_rounds=100,
        heuristic_opponents=7,
        observation_scale="raw",
    ):
        super().__init__()
        self.learner_seat = learner_seat
        self.max_rounds = max_rounds
        self.heuristic_opponents = heuristic_opponents
        self.observation_scale = observation_scale
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict(
            {
                "player": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "action_mask": spaces.MultiBinary(2),
            }
        )
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return self._obs(), {"seed": seed}

    def step(self, action):
        self.step_count += 1
        terminated = self.step_count >= 2
        info = {"rank": 4.0, "truncated": False} if terminated else {}
        return self._obs(), 4.0 if terminated else 0.0, terminated, False, info

    def action_masks(self):
        return np.array([True, True])

    @property
    def opponent_spec(self):
        return {
            "version": "lobby-opponent-mix-v1",
            "learner_seat": self.learner_seat,
            "heuristic_opponents": self.heuristic_opponents,
            "observation_scale": self.observation_scale,
        }

    def _obs(self):
        return {
            "player": np.array([self.step_count], dtype=np.float32),
            "action_mask": np.array([1, 1], dtype=np.int8),
        }


class TruncatingTinyLobbyEnv(TinyLobbyEnv):
    def step(self, action):
        self.step_count += 1
        return self._obs(), 0.0, False, True, {"rank": None, "termination_reason": "max_rounds"}


def install_tiny_env(monkeypatch):
    monkeypatch.setattr(lobby_training, "LobbyEnv", TinyLobbyEnv)


@pytest.mark.parametrize(
    "config",
    [
        dict(max_steps=0),
        dict(max_steps=8193),
        dict(max_steps=33),
        dict(max_seconds=float("nan")),
        dict(max_seconds=61),
        dict(threads=True),
        dict(threads=2),
        dict(batch_size=3),
        dict(gamma=0),
        dict(gamma=1.01),
        dict(learner_seat=8),
        dict(heuristic_opponents=-1),
        dict(heuristic_opponents=8),
        dict(observation_scale="bad-scale"),
        dict(observation_scale=1),
        dict(seed=20001),
        dict(unknown=True),
    ],
)
def test_invalid_config_does_not_create_output(tmp_path, config):
    with pytest.raises(ValueError):
        lobby_training.train(config, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_existing_output_is_preserved(tmp_path, monkeypatch):
    install_tiny_env(monkeypatch)
    output = tmp_path / "run"
    output.mkdir()
    sentinel = output / "status.json"
    sentinel.write_text("existing data", encoding="utf-8")
    with pytest.raises(FileExistsError, match="empty"):
        lobby_training.train({}, output)
    assert sentinel.read_text(encoding="utf-8") == "existing data"
    assert not (output / ".lobby-training.lock").exists()


def test_tiny_training_saves_loadable_lobby_checkpoint(tmp_path, monkeypatch):
    install_tiny_env(monkeypatch)
    config = dict(max_steps=32, checkpoint_interval=32, max_seconds=60, threads=1)

    result = lobby_training.train(config, tmp_path / "run")

    assert result["num_timesteps"] == 32
    assert result["episode_count"] > 0
    assert result["episodes"]
    assert result["episodes"][0]["rank"] == 4.0
    checkpoint = tmp_path / "run" / result["checkpoint"]
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["compatibility"]["observation_version"] == "lobby-own-public-v1"
    assert metadata["compatibility"]["reward_contract"] == lobby_training.REWARD_CONTRACT
    assert metadata["compatibility"]["learner_seat"] == 0
    assert metadata["compatibility"]["max_rounds"] == 100
    assert metadata["compatibility"]["opponent_protocol"] == "lobby-opponent-mix-v1"
    assert metadata["compatibility"]["heuristic_opponents"] == 7
    assert metadata["compatibility"]["observation_scale"] == "raw"
    assert metadata["config"]["heuristic_opponents"] == 7
    assert metadata["config"]["observation_scale"] == "raw"
    assert metadata["opponent_spec"]["heuristic_opponents"] == 7
    assert metadata["opponent_spec"]["observation_scale"] == "raw"
    env = TinyLobbyEnv()
    model = lobby_training.load_model(checkpoint, env)
    try:
        obs, _ = env.reset(seed=20001)
        first, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
        loaded = lobby_training.load_model(checkpoint, env)
        second, _ = loaded.predict(obs, deterministic=True, action_masks=env.action_masks())
        loaded.env.close()
        assert int(first) == int(second)
    finally:
        model.env.close()


def test_incompatible_metadata_rejected_before_deserialize(tmp_path, monkeypatch):
    install_tiny_env(monkeypatch)
    (tmp_path / "model.zip").write_bytes(b"not a trusted model")
    (tmp_path / "metadata.json").write_text(
        json.dumps({"compatibility": {"wrong": "contract"}, "config": {}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        lobby_training.MaskablePPO,
        "load",
        lambda *args, **kwargs: pytest.fail("unsafe deserialize"),
    )
    with pytest.raises(ValueError, match="compatibility"):
        lobby_training.load_model(tmp_path, TinyLobbyEnv())


def test_environment_mismatch_rejected_before_deserialize(tmp_path, monkeypatch):
    install_tiny_env(monkeypatch)
    config = lobby_training._config({})
    (tmp_path / "model.zip").write_bytes(b"not a trusted model")
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {"compatibility": lobby_training.lobby_compatibility_signature(config), "config": config}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        lobby_training.MaskablePPO,
        "load",
        lambda *args, **kwargs: pytest.fail("unsafe deserialize"),
    )
    with pytest.raises(ValueError, match="max_rounds"):
        lobby_training.load_model(tmp_path, TinyLobbyEnv(max_rounds=50))


def test_opponent_count_mismatch_rejected_unless_explicitly_allowed(tmp_path, monkeypatch):
    install_tiny_env(monkeypatch)
    config = lobby_training._config({})
    (tmp_path / "model.zip").write_bytes(b"not a trusted model")
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {"compatibility": lobby_training.lobby_compatibility_signature(config), "config": config}
        ),
        encoding="utf-8",
    )
    sentinel = object()
    monkeypatch.setattr(lobby_training.MaskablePPO, "load", lambda *args, **kwargs: sentinel)

    with pytest.raises(ValueError, match="heuristic_opponents"):
        lobby_training.load_model(tmp_path, TinyLobbyEnv(heuristic_opponents=3))

    assert (
        lobby_training.load_model(
            tmp_path,
            TinyLobbyEnv(heuristic_opponents=3),
            allow_opponent_shift=True,
        )
        is sentinel
    )


def test_observation_scale_mismatch_rejected_even_with_opponent_shift(tmp_path, monkeypatch):
    install_tiny_env(monkeypatch)
    config = lobby_training._config({"observation_scale": "raw"})
    (tmp_path / "model.zip").write_bytes(b"not a trusted model")
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {"compatibility": lobby_training.lobby_compatibility_signature(config), "config": config}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        lobby_training.MaskablePPO,
        "load",
        lambda *args, **kwargs: pytest.fail("unsafe deserialize"),
    )

    with pytest.raises(ValueError, match="observation_scale"):
        lobby_training.load_model(
            tmp_path,
            TinyLobbyEnv(heuristic_opponents=3, observation_scale="fixed-v1"),
            allow_opponent_shift=True,
        )


def test_fixed_observation_scale_training_config_is_recorded(tmp_path, monkeypatch):
    install_tiny_env(monkeypatch)
    config = dict(
        max_steps=32,
        checkpoint_interval=32,
        max_seconds=60,
        threads=1,
        observation_scale="fixed-v1",
    )

    result = lobby_training.train(config, tmp_path / "run")

    checkpoint = tmp_path / "run" / result["checkpoint"]
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    assert metadata["compatibility"]["observation_scale"] == "fixed-v1"
    assert metadata["config"]["observation_scale"] == "fixed-v1"
    assert metadata["opponent_spec"]["observation_scale"] == "fixed-v1"
    assert manifest["opponent_spec"]["observation_scale"] == "fixed-v1"


def test_real_sb3_truncation_callback_accounting(tmp_path, monkeypatch):
    monkeypatch.setattr(lobby_training, "LobbyEnv", TruncatingTinyLobbyEnv)

    result = lobby_training.train(
        dict(max_steps=32, checkpoint_interval=32, max_seconds=60, threads=1),
        tmp_path / "run",
    )

    assert result["episode_truncations"] == 32
    assert result["episode_count"] == 0
    assert result["episode_mean_rank"] is None
    assert len(result["episodes"]) == 32
    assert all(row["rank"] is None and row["truncated"] for row in result["episodes"])
    events = [
        json.loads(row)
        for row in (tmp_path / "run" / "train.log").read_text(encoding="utf-8").splitlines()
    ]
    episode_events = [event for event in events if event["event"] == "episode"]
    assert len(episode_events) == 32
    assert all(event["truncated"] and event["rank"] is None for event in episode_events)
