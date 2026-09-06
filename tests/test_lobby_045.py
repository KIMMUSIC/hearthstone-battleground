"""Paired initialization and independent gates for the fixed sampling comparison."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import lobby_045 as runner  # noqa: E402
from audit_lobby_032 import model_evidence  # noqa: E402


def test_sampling_gates_require_both_classification_and_game_improvement():
    labels = {s: {arm: {"final": {"dev": {"accuracy": .8, "macro_action_accuracy": macro}}}
                  for arm, macro in (("control", .3), ("candidate", .5))} for s in (7, 17, 27)}
    games = {s: {arm: {"final": {"mean_rank": rank, "top4_fraction_ranked": .1, "truncated": 0}}
                 for arm, rank in (("control", 7), ("candidate", 6.5))} for s in labels}
    assert runner.decide(labels, games)["candidate_keep"]
    for seed in labels:
        labels[seed]["candidate"]["final"]["dev"]["macro_action_accuracy"] = .2
    assert runner.decide(labels, games)["candidate_keep"]
    games[7]["candidate"]["final"]["mean_rank"] = 7.0
    assert not runner.decide(labels, games)["candidate_keep"]
    games[7]["candidate"]["final"]["mean_rank"] = 6.5
    labels[7]["candidate"]["final"]["dev"]["accuracy"] = .74
    result = runner.decide(labels, games)
    assert result["game_transfer_supported"] and not result["candidate_keep"]
    labels[7]["candidate"]["final"]["dev"]["accuracy"] = .8
    games[7]["candidate"]["final"]["truncated"] = 1
    assert not runner.decide(labels, games)["candidate_keep"]


def test_paired_sampling_phases_keep_initial_models_equal(tmp_path, monkeypatch):
    import numpy as np
    import torch
    from audit_lobby_043 import _compare_metric, classify, verify_sampling
    from hearthstone_ai.lobby_env import LobbyEnv
    from hearthstone_ai.lobby_training import load_model
    spec = runner.read(ROOT / "configs/lobby045_comparison.json")
    spec["dataset"].update(train_seeds=[0, 1], dev_seeds=[20001])
    spec["imitation"]["updates"] = 2
    spec["eval_seeds"] = spec["replay_seeds"] = [20001]
    runner.dump(tmp_path / "configs/lobby045_comparison.json", spec)
    monkeypatch.setattr(runner, "verify_files", lambda root: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda n: None)
    assert len(runner.jobs(spec)) == 17
    runner.phase(tmp_path, 0)
    from hearthstone_ai.lobby_imitation import train_imitation
    from lobby_032 import digest
    base = tmp_path / "outputs/dataset-s007-teacher"
    parent = tmp_path / "parent"
    status = train_imitation(spec["train"], spec["imitation"], base, parent)
    checkpoint = parent / status["checkpoint"]
    spec["collectors"]["7"] = {"checkpoint": str(checkpoint),
        "model_sha256": digest(checkpoint / "model.zip"),
        "metadata_sha256": digest(checkpoint / "metadata.json")}
    runner.dump(tmp_path / "configs/lobby045_comparison.json", spec)
    runner.phase(tmp_path, 1)
    weights, batches = [], []
    for index, arm in ((4, "control"), (5, "candidate")):
        result = runner.phase(tmp_path, index)
        folder = tmp_path / f"outputs/train-s007-{arm}"
        weights.append(model_evidence(folder / result["initial_checkpoint"] / "model.zip")[1])
        assert result["optimizer_steps"] == 2 and result["num_timesteps"] == 0
        with np.load(folder / "sampling.npz", allow_pickle=False) as sampled:
            assert sampled["indices"].shape == (2, 64)
            batches.append(sampled["indices"].copy())
        metrics = runner.read(folder / "classification.json")["final"]["dev"]
        assert sum(map(sum, metrics["confusion_matrix"])) == metrics["rows"]
        dataset = tmp_path / "outputs/dataset-s007-teacher"
        if arm == "candidate":
            dataset = tmp_path / "outputs/collect-s007-candidate"
        with np.load(dataset / "train.npz", allow_pickle=False) as archive:
            evidence = verify_sampling(folder, archive["actions"], spec["arms"][arm], 7, 2, 64)
        assert evidence["action_exposure_histogram"] == result["sampling"]["histogram"]
        with np.load(dataset / "dev.npz", allow_pickle=False) as archive:
            dev = {k: archive[k] for k in archive.files}
        env = LobbyEnv(observation_scale="fixed-v1")
        try:
            model = load_model(folder / result["checkpoint"], env)
            _compare_metric(metrics, classify(model, dev), arm)
        finally:
            env.close()
    assert all(torch.equal(weights[0][k], weights[1][k]) for k in weights[0])
    assert not np.array_equal(batches[0], batches[1])
    report = runner.phase(tmp_path, 11)
    assert report["final"]["truncated"] == 0 and report["final"]["replay_seeds"] == [20001]

    visited = runner.read(tmp_path / "outputs/evaluate-s007-candidate/visited.json")
    assert visited["episodes"] == 1 and visited["rows"] > 0
    assert sum(map(sum, visited["confusion_matrix"])) == visited["rows"]

    runner.phase(tmp_path, 10)
    from audit_lobby_045 import visited_state_metrics_from_reports
    from hearthstone_ai.lobby_policies import heuristic_policy
    env = LobbyEnv(observation_scale="fixed-v1")
    try:
        evidence = visited_state_metrics_from_reports(tmp_path, spec | {"seeds": [7]}, env, load_model, heuristic_policy)
        assert evidence["summary"]["episodes"] == 2
    finally:
        env.close()
