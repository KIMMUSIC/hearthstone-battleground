import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from hearthstone_ai.artifacts import file_hash
from hearthstone_ai.lobby_env import LobbyEnv
from hearthstone_ai.lobby_imitation import build_dataset, train_imitation
from hearthstone_ai.lobby_training import load_model


def _collector_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "collect_lobby_045.py"
    spec = importlib.util.spec_from_file_location("collect_lobby_045", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


collect = _collector_module().collect


def training_config(seed=7):
    return {
        "max_steps": 32,
        "max_seconds": 60,
        "threads": 1,
        "seed": seed,
        "n_steps": 32,
        "batch_size": 32,
        "n_epochs": 4,
        "checkpoint_interval": 32,
        "learner_seat": 0,
        "max_rounds": 100,
        "heuristic_opponents": 7,
        "observation_scale": "fixed-v1",
        "gamma": 1.0,
    }


def base_and_checkpoint(tmp_path):
    base = tmp_path / "base"
    build_dataset({"train_seeds": [0], "dev_seeds": [20001], "max_rounds": 100}, base)
    trained = tmp_path / "trained"
    status = train_imitation(
        training_config(seed=7),
        {
            "updates": 2,
            "batch_size": 4,
            "learning_rate": 0.0003,
            "max_seconds": 60,
            "sampling": "balanced-choice",
        },
        base,
        trained,
    )
    return base, trained / status["checkpoint"]


def test_collect_appends_parent_visited_rows_preserves_prefix_and_copies_dev(tmp_path):
    base, checkpoint = base_and_checkpoint(tmp_path)
    base_train_bytes = (base / "train.npz").read_bytes()
    base_dev_bytes = (base / "dev.npz").read_bytes()

    summary = collect(base, checkpoint, tmp_path / "collect", model_seed=7, seeds=[0])

    assert summary["status"] == "completed"
    assert summary["base_train_rows"] > 0
    assert summary["added_rows"] > 0
    assert summary["split_counts"]["train"] == summary["base_train_rows"] + summary["added_rows"]
    assert summary["split_counts"]["dev"] > 0
    assert summary["provenance"]["checkpoint_model_sha256"] == file_hash(checkpoint / "model.zip")
    assert summary["provenance"]["input_hashes_before"] == summary["provenance"]["input_hashes_after"]

    with np.load(base / "train.npz", allow_pickle=False) as base_train:
        base_arrays = {key: base_train[key].copy() for key in base_train.files}
    with np.load(tmp_path / "collect" / "train.npz", allow_pickle=False) as augmented:
        assert set(augmented.files) == set(base_arrays)
        for key, value in base_arrays.items():
            np.testing.assert_array_equal(augmented[key][: len(value)], value)
        assert augmented["actions"].shape[0] == summary["split_counts"]["train"]
        assert augmented["seeds"][summary["base_train_rows"] :].tolist() == [0] * summary["added_rows"]
    assert (base / "dev.npz").read_bytes() == base_dev_bytes
    assert (tmp_path / "collect" / "dev.npz").read_bytes() == base_dev_bytes
    assert (base / "train.npz").read_bytes() == base_train_bytes

    collection = json.loads((tmp_path / "collect" / "collection.json").read_text(encoding="utf-8"))
    assert collection["base_train_rows"] == summary["base_train_rows"]
    assert collection["added_rows"] == summary["added_rows"]
    assert len(collection["episodes"]) == 1
    episode = collection["episodes"][0]
    assert episode["seed"] == 0
    assert episode["row_start"] == summary["base_train_rows"]
    assert episode["row_end"] == summary["base_train_rows"] + summary["added_rows"]
    assert len(episode["executed_actions"]) == len(episode["teacher_actions"]) == summary["added_rows"]
    assert isinstance(episode["rank"], float)
    assert isinstance(episode["reward"], float)
    assert len(episode["trace_sha256"]) == 64


def test_collect_manifest_remains_loadable_for_training_without_dev_leak(tmp_path):
    base, checkpoint = base_and_checkpoint(tmp_path)
    output = tmp_path / "collect"

    summary = collect(base, checkpoint, output, model_seed=7, seeds=[0])

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    base_manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"] == base_manifest["config"]
    assert manifest["core_config"] == base_manifest["core_config"]
    assert manifest["compatibility"] == base_manifest["compatibility"]
    assert manifest["split_hashes"]["dev"] == base_manifest["split_hashes"]["dev"]
    assert manifest["split_hashes"]["train"] == file_hash(output / "train.npz")
    assert manifest["provenance"]["trajectory"]["row_start"] == summary["base_train_rows"]

    train_output = tmp_path / "retrain"
    status = train_imitation(
        training_config(seed=17),
        {
            "updates": 1,
            "batch_size": 4,
            "learning_rate": 0.0003,
            "max_seconds": 60,
            "sampling": "balanced-choice",
        },
        output,
        train_output,
    )
    env = LobbyEnv(observation_scale="fixed-v1")
    model = load_model(train_output / status["checkpoint"], env)
    try:
        assert model.num_timesteps == 0
    finally:
        model.env.close()


def test_collect_rejects_seed_domain_manifest_mismatch_and_existing_output(tmp_path):
    base, checkpoint = base_and_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="0..9999"):
        collect(base, checkpoint, tmp_path / "bad-domain", model_seed=7, seeds=[10000])
    with pytest.raises(ValueError, match="train_seeds"):
        collect(base, checkpoint, tmp_path / "bad-subset", model_seed=7, seeds=[1])

    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "sentinel").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        collect(base, checkpoint, existing, model_seed=7, seeds=[0])
    assert (existing / "sentinel").read_text(encoding="utf-8") == "keep"


def test_collect_preserves_half_rank_values_in_collection(tmp_path, monkeypatch):
    base, checkpoint = base_and_checkpoint(tmp_path)
    module = _collector_module()
    real_rows = module._collection_rows

    def half_rank_rows(*args, **kwargs):
        rows, teacher_actions, seeds, episode = real_rows(*args, **kwargs)
        episode = {**episode, "rank": 4.5}
        return rows, teacher_actions, seeds, episode

    monkeypatch.setattr(module, "_collection_rows", half_rank_rows)
    summary = module.collect(base, checkpoint, tmp_path / "half", model_seed=7, seeds=[0])

    assert summary["collection"]["episodes"][0]["rank"] == 4.5
    assert isinstance(summary["collection"]["episodes"][0]["rank"], float)


def test_collect_fails_if_watched_inputs_change_before_success(tmp_path, monkeypatch):
    base, checkpoint = base_and_checkpoint(tmp_path)
    module = _collector_module()
    real_file_hash = module.file_hash
    calls = {"base_train": 0}

    def drifting_hash(path):
        if Path(path).name == "train.npz" and Path(path).parent == base:
            calls["base_train"] += 1
            if calls["base_train"] >= 2:
                return "changed"
        return real_file_hash(path)

    monkeypatch.setattr(module, "file_hash", drifting_hash)

    with pytest.raises(RuntimeError, match="input hashes changed"):
        module.collect(base, checkpoint, tmp_path / "drift", model_seed=7, seeds=[0])

    assert not (tmp_path / "drift" / "manifest.json").exists()
    assert not (tmp_path / "drift" / "collection.json").exists()
