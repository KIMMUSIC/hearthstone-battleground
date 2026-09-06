from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_lobby_045 import (  # noqa: E402
    DATASET_DEV_SEEDS,
    expected_jobs,
    json_digest,
    validate_045_spec,
    verify_collector_dataset,
)
from lobby_032 import digest, dump, read  # noqa: E402


class AlwaysContains:
    def contains(self, _obs):
        return True


class FakeGame:
    def __init__(self):
        self.players = [SimpleNamespace(rank=None)]
        self.trace = []
        self._step = 0

    def view(self, _seat):
        return {"step": self._step, "legal_actions": list(range(37))}

    def assert_conservation(self):
        return None


class FakeEnv:
    learner_seat = 0
    observation_space = AlwaysContains()

    def __init__(self):
        self.game = FakeGame()
        self._seed = None
        self._step = 0

    def _obs(self):
        return {
            "feature": np.asarray([self._seed, self._step], dtype=np.int64),
            "action_mask": np.ones(37, dtype=bool),
        }

    def reset(self, *, seed=None):
        self._seed = int(seed)
        self._step = 0
        self.game = FakeGame()
        return self._obs(), {}

    def action_masks(self):
        return np.ones(37, dtype=bool)

    def step(self, action):
        self.game.trace.append({"seed": self._seed, "step": self._step, "action": int(action)})
        self._step += 1
        self.game._step = self._step
        terminated = self._step == 2
        if terminated:
            self.game.players[0].rank = 1.0
        return self._obs(), 1.0 if terminated else 0.0, terminated, False, {}


class FakeModel:
    def predict(self, obs, deterministic=True, action_masks=None):
        step = int(obs["feature"][1])
        return np.asarray(1 if step == 0 else 0), None


def fake_teacher(view):
    return 2 if view["step"] == 0 else 3


def write_npz(path, rows, actions, seeds):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        obs__feature=np.asarray(rows, dtype=np.int64),
        obs__action_mask=np.ones((len(actions), 37), dtype=bool),
        actions=np.asarray(actions, dtype=np.int64),
        seeds=np.asarray(seeds, dtype=np.int64),
    )


def write_manifest(directory, train_rows, train_seeds):
    dump(
        directory / "manifest.json",
        {
            "config": {"train_seeds": train_seeds, "dev_seeds": DATASET_DEV_SEEDS, "max_rounds": 100},
            "split_counts": {"train": train_rows, "dev": 1},
            "split_hashes": {
                "train": digest(directory / "train.npz"),
                "dev": digest(directory / "dev.npz"),
            },
            "core_config": {"learner_seat": 0, "max_rounds": 100, "heuristic_opponents": 7, "observation_scale": "fixed-v1"},
            "compatibility": {},
        },
    )


def spec():
    return read(ROOT / "configs/lobby045_comparison.json")


def test_045_spec_and_job_contract_matches_runner_config():
    plan = spec()
    validate_045_spec(plan)
    jobs = expected_jobs(plan)
    assert len(jobs) == 17
    assert jobs[:4] == [("dataset", 7, "teacher"), ("collect", 7, "candidate"), ("collect", 17, "candidate"), ("collect", 27, "candidate")]
    assert jobs[4:8] == [("train", 7, "control"), ("train", 7, "candidate"), ("train", 17, "control"), ("train", 17, "candidate")]
    assert jobs[-1] == ("baseline", 7, "control")
    changed = spec()
    changed["arms"]["control"] = "uniform-choice"
    with pytest.raises(ValueError, match="balanced-choice"):
        validate_045_spec(changed)


def test_verify_collector_dataset_replays_augmented_rows_and_schema(tmp_path):
    base = tmp_path / "base"
    collect = tmp_path / "collect-s000-candidate"
    write_npz(base / "train.npz", [[0, 99]], [2], [0])
    write_npz(base / "dev.npz", [[20001, 0]], [2], [20001])
    write_manifest(base, 1, [0])
    write_npz(collect / "train.npz", [[0, 99], [0, 0], [0, 1]], [2, 2, 3], [0, 0, 0])
    write_npz(collect / "dev.npz", [[20001, 0]], [2], [20001])
    write_manifest(collect, 3, [0])
    trace = [{"seed": 0, "step": 0, "action": 1}, {"seed": 0, "step": 1, "action": 0}]
    dump(
        collect / "collection.json",
        {
            "model_seed": 0,
            "checkpoint_model_sha256": "a" * 64,
            "checkpoint_metadata_sha256": "b" * 64,
            "base_train_rows": 1,
            "added_rows": 2,
            "episodes": [
                {
                    "seed": 0,
                    "executed_actions": [1, 0],
                    "teacher_actions": [2, 3],
                    "row_start": 1,
                    "row_end": 3,
                    "rank": 1.0,
                    "reward": 1.0,
                    "trace_sha256": json_digest(trace),
                }
            ],
        },
    )

    record = read(collect / "collection.json")
    inputs = {"base_train": digest(base / "train.npz"), "base_dev": digest(base / "dev.npz"),
              "checkpoint_model": "a" * 64, "checkpoint_metadata": "b" * 64}
    record.update(input_hashes_before=inputs, input_hashes_after=inputs,
                  output_hashes={"train": digest(collect / "train.npz"), "dev": digest(collect / "dev.npz")})
    dump(collect / "collection.json", record)
    manifest = read(collect / "manifest.json")
    manifest["provenance"] = {k: v for k, v in record.items() if k not in ("episodes", "output_hashes")}
    manifest["provenance"].update(kind="lobby-045-visited-training", trajectory={
        "prefix": "teacher-train", "append": "current-parent-model-visited-states-with-heuristic-labels",
        "row_start": 1, "row_end": 3})
    dump(collect / "manifest.json", manifest)

    proof = verify_collector_dataset(
        base,
        collect,
        {"model_sha256": "a" * 64, "metadata_sha256": "b" * 64},
        FakeEnv(),
        FakeModel(),
        fake_teacher,
        train_seeds=[0],
    )

    assert proof["base_teacher_rows"] == 1
    assert proof["collected_rows"] == 2
    assert proof["augmented_train_rows"] == 3
    assert proof["model_prediction_matches_collection"] == 2
    assert proof["teacher_action_histogram"] == {"2": 1, "3": 1}
    assert proof["executed_action_histogram"] == {"0": 1, "1": 1}

    bad = read(collect / "collection.json")
    bad["episodes"][0]["teacher_actions"] = [2, 2]
    dump(collect / "collection.json", bad)
    with pytest.raises(ValueError, match="teacher labels"):
        verify_collector_dataset(
            base,
            collect,
            {"model_sha256": "a" * 64, "metadata_sha256": "b" * 64},
            FakeEnv(),
            FakeModel(),
            fake_teacher,
            train_seeds=[0],
        )


def test_real_collector_and_independent_audit_agree(tmp_path):
    from collect_lobby_045 import collect
    from audit_lobby_045 import verify_provenance
    from hearthstone_ai.lobby_imitation import build_dataset, train_imitation
    from hearthstone_ai.lobby_env import LobbyEnv
    from hearthstone_ai.lobby_training import load_model
    plan = spec()
    base, parent, output = tmp_path / "base", tmp_path / "parent", tmp_path / "collect"
    build_dataset(plan["dataset"] | {"train_seeds": [0]}, base)
    status = train_imitation(plan["train"], plan["imitation"] | {"updates": 2}, base, parent)
    checkpoint = parent / status["checkpoint"]
    collect(base, checkpoint, output, model_seed=7, seeds=[0])
    verify_provenance(base, output, checkpoint, 7)
    from hearthstone_ai.lobby_policies import heuristic_policy
    env = LobbyEnv(observation_scale="fixed-v1")
    model = load_model(checkpoint, env)
    collector = {"model_sha256": digest(checkpoint / "model.zip"),
                 "metadata_sha256": digest(checkpoint / "metadata.json")}
    try:
        proof = verify_collector_dataset(base, output, collector, env, model, heuristic_policy, [0])
        assert proof["collected_rows"] > 0
        assert proof["model_prediction_matches_collection"] == proof["collected_rows"]
    finally:
        env.close()
    record = read(output / "collection.json")
    record["input_hashes_after"]["base_train"] = "0" * 64
    dump(output / "collection.json", record)
    with pytest.raises(ValueError, match="provenance"):
        verify_provenance(base, output, checkpoint, 7)
