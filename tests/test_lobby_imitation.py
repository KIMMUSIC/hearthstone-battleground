import json

import numpy as np
import pytest

from hearthstone_ai.artifacts import file_hash
from hearthstone_ai.lobby_env import LobbyEnv
from hearthstone_ai.lobby_imitation import (
    build_dataset,
    evaluate_labels,
    sampling_probabilities,
    train_imitation,
)
from hearthstone_ai.lobby_training import load_model


def dataset(tmp_path):
    output = tmp_path / "dataset"
    summary = build_dataset({"train_seeds": [0], "dev_seeds": [20001], "max_rounds": 100}, output)
    return output, summary


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


def test_build_dataset_writes_numeric_npz_manifest_and_split_hashes(tmp_path):
    output, summary = dataset(tmp_path)

    assert set(summary["config"]) == {"train_seeds", "dev_seeds", "max_rounds"}
    assert summary["config"]["train_seeds"] == [0]
    assert summary["config"]["dev_seeds"] == [20001]
    assert summary["core_config"]["heuristic_opponents"] == 7
    assert summary["core_config"]["observation_scale"] == "fixed-v1"
    assert summary["split_counts"]["train"] > 0
    assert summary["split_counts"]["dev"] > 0
    assert summary["split_hashes"]["train"] != summary["split_hashes"]["dev"]

    with np.load(output / "train.npz", allow_pickle=False) as train:
        assert {"actions", "seeds", "obs__action_mask", "obs__player", "obs__public_players"} <= set(train.files)
        assert train["actions"].dtype == np.int64
        assert train["seeds"].tolist() == [0] * len(train["actions"])
        assert train["obs__action_mask"].shape == (len(train["actions"]), 37)
    with np.load(output / "dev.npz", allow_pickle=False) as dev:
        assert dev["seeds"].tolist() == [20001] * len(dev["actions"])

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["split_hashes"] == summary["split_hashes"]


def test_train_imitation_saves_loadable_zero_timestep_checkpoints_and_metrics(tmp_path):
    dataset_dir, _ = dataset(tmp_path)
    train_hash = file_hash(dataset_dir / "train.npz")
    dev_hash = file_hash(dataset_dir / "dev.npz")
    output = tmp_path / "imitation"

    status = train_imitation(
        training_config(seed=7),
        {"updates": 2, "batch_size": 4, "learning_rate": 0.0003, "max_seconds": 60},
        dataset_dir,
        output,
    )

    assert status["status"] == "completed"
    assert status["stop_reason"] == "updates"
    assert status["optimizer_steps"] == 2
    assert status["rows_seen"] == 8
    assert status["num_timesteps"] == 0
    assert status["sb3_n_updates"] == 0
    assert status["initial_checkpoint"] == "checkpoint-initial"
    assert status["checkpoint"] == "checkpoint-final"
    assert status["critic_unchanged"] is True
    assert status["finite_parameters"] is True
    assert status["sampling"]["mode"] == "legacy-integers"
    assert status["sampling"]["protocol"] == "legacy-integers-v1"
    assert status["sampling"]["hash"] == file_hash(output / "sampling.npz")
    assert file_hash(dataset_dir / "train.npz") == train_hash
    assert file_hash(dataset_dir / "dev.npz") == dev_hash

    with np.load(output / "sampling.npz", allow_pickle=False) as sampling:
        assert sampling["indices"].dtype == np.int64
        assert sampling["indices"].shape == (2, 4)
        assert sampling["probabilities"].dtype == np.float64
        assert sampling["probabilities"].shape == (status["dataset_train_rows"],)
        np.testing.assert_array_equal(
            sampling["indices"],
            np.random.default_rng(7).integers(0, status["dataset_train_rows"], size=(2, 4)),
        )

    for checkpoint in ("checkpoint-initial", "checkpoint-final"):
        metadata = json.loads((output / checkpoint / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["training_method"] == "behavior-cloning-diagnostic-v1"
        assert metadata["num_timesteps"] == 0
        assert metadata["additional_steps"] == 0
        assert metadata["config"]["observation_scale"] == "fixed-v1"
        assert metadata["config"]["heuristic_opponents"] == 7
        assert metadata["supervised"]["dataset_train_rows"] == status["dataset_train_rows"]
        assert metadata["supervised"]["finite_parameters"] is True
        if checkpoint == "checkpoint-final":
            assert metadata["supervised"]["sampling"] == status["sampling"]

    env = LobbyEnv(heuristic_opponents=7, observation_scale="fixed-v1")
    model = load_model(output / status["checkpoint"], env)
    try:
        metrics = evaluate_labels(model, dataset_dir / "dev.npz")
    finally:
        model.env.close()
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["nll"] >= 0.0
    assert 0.0 <= metrics["macro_action_accuracy"] <= 1.0
    assert np.asarray(metrics["confusion_matrix"]).shape == (37, 37)
    assert sum(row["count"] for row in metrics["per_action"].values()) == metrics["rows"]


def test_sampling_probabilities_match_uniform_and_balanced_formulas():
    actions = np.array([1, 1, 2, 3], dtype=np.int64)

    np.testing.assert_allclose(
        sampling_probabilities(actions, "uniform-choice"),
        np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64),
    )
    np.testing.assert_allclose(
        sampling_probabilities(actions, "balanced-choice"),
        np.array([
            0.5 / 4 + 0.5 / (3 * 2),
            0.5 / 4 + 0.5 / (3 * 2),
            0.5 / 4 + 0.5 / (3 * 1),
            0.5 / 4 + 0.5 / (3 * 1),
        ], dtype=np.float64),
    )
    with pytest.raises(ValueError, match="sampling mode"):
        sampling_probabilities(actions, "legacy-integers")


@pytest.mark.parametrize("mode", ["uniform-choice", "balanced-choice"])
def test_explicit_choice_sampling_records_replayable_actual_indices(tmp_path, mode):
    dataset_dir, _ = dataset(tmp_path)
    output = tmp_path / mode

    status = train_imitation(
        training_config(seed=17),
        {"updates": 3, "batch_size": 4, "learning_rate": 0.0003, "max_seconds": 60, "sampling": mode},
        dataset_dir,
        output,
    )

    with np.load(dataset_dir / "train.npz", allow_pickle=False) as train:
        probabilities = sampling_probabilities(train["actions"], mode)
        train_actions = train["actions"].copy()
    with np.load(output / "sampling.npz", allow_pickle=False) as sampling:
        np.testing.assert_allclose(sampling["probabilities"], probabilities)
        expected = np.random.default_rng(17).choice(
            status["dataset_train_rows"],
            size=(3, 4),
            replace=True,
            p=probabilities,
        )
        np.testing.assert_array_equal(sampling["indices"], expected)
        sampled_actions = train_actions[sampling["indices"].reshape(-1)]
    histogram = {
        str(int(action)): int(count)
        for action, count in zip(*np.unique(sampled_actions, return_counts=True), strict=True)
    }
    assert status["sampling"]["mode"] == mode
    assert status["sampling"]["protocol"] == "choice-probabilities-v1"
    assert status["sampling"]["hash"] == file_hash(output / "sampling.npz")
    assert status["sampling"]["histogram"] == histogram


def test_invalid_configs_and_fresh_output_refusals(tmp_path):
    with pytest.raises(ValueError, match="disjoint"):
        build_dataset({"train_seeds": [0], "dev_seeds": [0], "max_rounds": 100}, tmp_path / "bad")
    with pytest.raises(ValueError, match="observation_scale"):
        train_imitation(
            {**training_config(), "observation_scale": "raw"},
            {"updates": 1, "batch_size": 1, "learning_rate": 0.0003, "max_seconds": 60},
            tmp_path,
            tmp_path / "unused",
        )
    dataset_dir, _ = dataset(tmp_path / "sampling")
    with pytest.raises(ValueError, match="sampling"):
        train_imitation(
            training_config(),
            {
                "updates": 1,
                "batch_size": 1,
                "learning_rate": 0.0003,
                "max_seconds": 60,
                "sampling": "bad-mode",
            },
            dataset_dir,
            tmp_path / "bad-sampling",
        )

    dataset_dir, _ = dataset(tmp_path)
    with pytest.raises(FileExistsError):
        build_dataset({"train_seeds": [1], "dev_seeds": [20002], "max_rounds": 100}, dataset_dir)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "sentinel").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        train_imitation(
            training_config(),
            {"updates": 1, "batch_size": 1, "learning_rate": 0.0003, "max_seconds": 60},
            dataset_dir,
            output,
        )
    assert (output / "sentinel").read_text(encoding="utf-8") == "keep"


def test_train_imitation_rejects_tampered_manifest_and_npz_before_training(tmp_path):
    dataset_dir, _ = dataset(tmp_path)
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["compatibility"]["observation_scale"] = "raw"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="compatibility"):
        train_imitation(
            training_config(),
            {"updates": 1, "batch_size": 1, "learning_rate": 0.0003, "max_seconds": 60},
            dataset_dir,
            tmp_path / "bad-manifest",
        )

    dataset_dir, _ = dataset(tmp_path / "seed")
    with np.load(dataset_dir / "train.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["seeds"] = arrays["seeds"].copy()
    arrays["seeds"][0] = 99
    np.savez_compressed(dataset_dir / "train.npz", **arrays)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["split_hashes"]["train"] = "tampered-to-force-hash-first"
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        train_imitation(
            training_config(),
            {"updates": 1, "batch_size": 1, "learning_rate": 0.0003, "max_seconds": 60},
            dataset_dir,
            tmp_path / "bad-hash",
        )


def test_train_imitation_rejects_bad_npz_seed_and_illegal_label_with_matching_hash(tmp_path):
    dataset_dir, _ = dataset(tmp_path)
    with np.load(dataset_dir / "train.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["seeds"] = arrays["seeds"].copy()
    arrays["seeds"][0] = 99
    np.savez_compressed(dataset_dir / "train.npz", **arrays)
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split_hashes"]["train"] = file_hash(dataset_dir / "train.npz")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="seeds mismatch"):
        train_imitation(
            training_config(),
            {"updates": 1, "batch_size": 1, "learning_rate": 0.0003, "max_seconds": 60},
            dataset_dir,
            tmp_path / "bad-seed",
        )

    dataset_dir, _ = dataset(tmp_path / "label")
    with np.load(dataset_dir / "train.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["actions"] = arrays["actions"].copy()
    arrays["obs__action_mask"] = arrays["obs__action_mask"].copy()
    arrays["actions"][0] = 36
    arrays["obs__action_mask"][0, 36] = 0
    np.savez_compressed(dataset_dir / "train.npz", **arrays)
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split_hashes"]["train"] = file_hash(dataset_dir / "train.npz")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="teacher actions"):
        train_imitation(
            training_config(),
            {"updates": 1, "batch_size": 1, "learning_rate": 0.0003, "max_seconds": 60},
            dataset_dir,
            tmp_path / "bad-label",
        )


def test_train_imitation_rejects_config_mismatch_with_dataset_core(tmp_path):
    dataset_dir, _ = dataset(tmp_path)
    with pytest.raises(ValueError, match="max_rounds"):
        train_imitation(
            {**training_config(), "max_rounds": 99},
            {"updates": 1, "batch_size": 1, "learning_rate": 0.0003, "max_seconds": 60},
            dataset_dir,
            tmp_path / "bad-config",
        )


def test_train_imitation_time_stop_is_not_reported_as_success(tmp_path):
    dataset_dir, _ = dataset(tmp_path)
    output = tmp_path / "short"

    status = train_imitation(
        training_config(seed=17),
        {"updates": 1024, "batch_size": 4, "learning_rate": 0.0003, "max_seconds": 1e-12},
        dataset_dir,
        output,
    )

    assert status["status"] == "stopped"
    assert status["stop_reason"] == "time"
    assert status["optimizer_steps"] < 1024
    assert status["checkpoint"] == "checkpoint-final"
