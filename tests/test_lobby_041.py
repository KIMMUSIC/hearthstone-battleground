"""Exercise diagnostic orchestration and independently inspect its real artifacts."""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import lobby_041 as runner  # noqa: E402
from audit_lobby_041 import actor_optimizer, classify, verify_dataset  # noqa: E402


def test_classification_and_game_gates_are_separate():
    labels = {s: {"initial": {"accuracy": .1}, "final": {"accuracy": .8}} for s in (7, 17, 27)}
    games = {s: {"initial": {"mean_rank": 8, "truncated": 0},
                 "final": {"mean_rank": 8, "top4_fraction_ranked": 0, "truncated": 0}} for s in labels}
    result = runner.decide(labels, games)
    assert result["imitation_supported"] and not result["game_transfer_supported"]
    for game in games.values():
        game["final"].update(mean_rank=7, top4_fraction_ranked=.1)
    assert runner.decide(labels, games)["game_transfer_supported"]
    games[7]["final"]["truncated"] = 1
    assert not runner.decide(labels, games)["game_transfer_supported"]


def test_real_dataset_training_and_evaluation_phases(tmp_path, monkeypatch):
    import torch
    from hearthstone_ai.lobby_env import LobbyEnv
    from hearthstone_ai.lobby_policies import heuristic_policy
    from hearthstone_ai.lobby_training import load_model

    spec = runner.read(ROOT / "configs/lobby041_imitation.json")
    spec["dataset"].update(train_seeds=[0, 1], dev_seeds=[20001])
    spec["imitation"]["updates"] = 2
    spec["eval_seeds"] = spec["replay_seeds"] = [20001]
    runner.dump(tmp_path / "configs/lobby041_imitation.json", spec)
    monkeypatch.setattr(runner, "verify_files", lambda root: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda n: None)
    assert len(runner.jobs(spec)) == 8
    runner.phase(tmp_path, 0)
    status = runner.phase(tmp_path, 1)
    assert status["num_timesteps"] == status["sb3_n_updates"] == 0
    dataset = tmp_path / "outputs/dataset-s007-student"
    folder = tmp_path / "outputs/train-s007-student"
    env = LobbyEnv(observation_scale="fixed-v1")
    try:
        verify_dataset(dataset / "train.npz", [0, 1], env, heuristic_policy)
        dev = verify_dataset(dataset / "dev.npz", [20001], env, heuristic_policy)
        model = load_model(folder / status["checkpoint"], env)
        assert actor_optimizer(folder / status["checkpoint"] / "model.zip", model, 2)["critic_parameter_states"] == 0
        with pytest.raises(ValueError, match="optimizer state"):
            actor_optimizer(folder / status["checkpoint"] / "model.zip", model, 3)
        metrics = classify(model, dev)
        recorded = runner.read(folder / "classification.json")["final"]
        assert metrics["accuracy"] == recorded["accuracy"]
        assert metrics["nll"] == pytest.approx(recorded["nll"], abs=1e-5)
        reports = runner.phase(tmp_path, 4)
        assert reports["final"]["truncated"] == 0
        assert reports["final"]["replay_seeds"] == [20001]
    finally:
        env.close()
