"""Fixed-scale campaign config and real train/evaluation contract coverage."""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import lobby_035 as runner  # noqa: E402


def test_only_observation_scale_changes_between_arms():
    spec = runner.read(ROOT / "configs/lobby037_comparison.json")
    assert len(runner.jobs(spec)) == 13
    for seed in (7, 17, 27):
        control = runner.training_config(spec, seed, "control")
        candidate = runner.training_config(spec, seed, "candidate")
        assert control.pop("observation_scale") == "raw"
        assert candidate.pop("observation_scale") == "fixed-v1"
        assert control == candidate
        assert control["heuristic_opponents"] == 3


def test_scaled_phase_evaluates_with_matching_encoding(tmp_path, monkeypatch):
    import torch
    from hearthstone_ai.lobby_env import LobbyEnv
    from hearthstone_ai.lobby_training import load_model
    from hearthstone_ai.lobby_policies import heuristic_policy, random_policy
    from audit_lobby_035 import replay_report

    spec = runner.read(ROOT / "configs/lobby037_comparison.json")
    spec["train"]["max_steps"] = 32
    spec["eval_seeds"] = [20001]
    spec["replay_seeds"] = [20001]
    filename = "lobby037_comparison.json"
    runner.dump(tmp_path / "configs" / filename, spec)
    monkeypatch.setattr(runner, "verify_files", lambda root: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda n: None)
    result = runner.phase(tmp_path, 1, filename)
    assert result["num_timesteps"] == 32
    runner.phase(tmp_path, 7, filename)
    context = runner.read(tmp_path / "outputs/evaluate-s007-candidate/comparison-context.json")
    assert context["observation_scale"] == "fixed-v1"
    assert context["evaluation_heuristic_opponents"] == 7
    checkpoint = tmp_path / "outputs/train-s007-candidate" / result["checkpoint"]
    with pytest.raises(ValueError):
        load_model(checkpoint, LobbyEnv(), allow_opponent_shift=True)
    env = LobbyEnv(observation_scale="fixed-v1")
    model = load_model(checkpoint, env, allow_opponent_shift=True)
    report = runner.read(tmp_path / "outputs/evaluate-s007-candidate/final.json")
    metrics, actions = replay_report(report, env, model, None, spec, heuristic_policy, random_policy)
    assert metrics["truncated"] == 0 and actions > 0
    env.close()
