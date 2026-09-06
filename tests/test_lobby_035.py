"""Fixed comparison schedule, decision gate, and real phase integration."""

import copy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import lobby_035 as runner  # noqa: E402


def test_schedule_covers_all_seeds_and_one_common_baseline():
    jobs = runner.jobs(runner.read(ROOT / "configs/lobby035_comparison.json"))
    assert len(jobs) == 13 and len(set(jobs)) == 13
    assert all(j[0] == "train" for j in jobs[:6])
    assert all(j[0] == "evaluate" for j in jobs[6:12])
    assert jobs[-1][0] == "baseline"


def test_decision_requires_learning_and_multiple_positive_seeds():
    initial = {"mean_rank": 8, "top4_fraction_ranked": 0, "truncated": 0}
    final = {"mean_rank": 7, "top4_fraction_ranked": 0, "truncated": 0}
    rows = {s: {"control": {"initial": initial, "final": initial},
                "candidate": {"initial": initial, "final": final}} for s in (7, 17, 27)}
    assert runner.decide(rows)["candidate_keep"]
    weak = copy.deepcopy(rows)
    for seed in (17, 27):
        weak[seed]["candidate"]["final"] = initial
    assert not runner.decide(weak)["candidate_keep"]
    with pytest.raises(ValueError):
        runner.decide({7: rows[7]})


def test_actual_candidate_phase_and_common_evaluation(tmp_path, monkeypatch):
    import torch
    spec = runner.read(ROOT / "configs/lobby035_comparison.json")
    spec["train"]["max_steps"] = 32
    spec["eval_seeds"] = [20001]
    spec["replay_seeds"] = [20001]
    runner.dump(tmp_path / "configs/lobby035_comparison.json", spec)
    monkeypatch.setattr(runner, "verify_files", lambda root: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda n: None)
    result = runner.phase(tmp_path, 1)
    assert result["num_timesteps"] == 32
    reports = runner.phase(tmp_path, 7)
    assert reports["final"]["replay_seeds"] == [20001]
    context = runner.read(tmp_path / "outputs/evaluate-s007-candidate/comparison-context.json")
    assert context["training_heuristic_opponents"] == 3
    assert context["evaluation_heuristic_opponents"] == 7
    from audit_lobby_035 import replay_report
    from hearthstone_ai.lobby_env import LobbyEnv
    from hearthstone_ai.lobby_training import load_model
    from hearthstone_ai.lobby_policies import heuristic_policy, random_policy
    env = LobbyEnv()
    model = load_model(tmp_path / "outputs/train-s007-candidate" / result["checkpoint"], env, allow_opponent_shift=True)
    report = runner.read(tmp_path / "outputs/evaluate-s007-candidate/final.json")
    metrics, actions = replay_report(report, env, model, None, spec, heuristic_policy, random_policy)
    assert actions > 0 and metrics["truncated"] == 0
    report["episodes"][0]["reward"] += 1
    with pytest.raises(ValueError, match="trace mismatch"):
        replay_report(report, env, model, None, spec, heuristic_policy, random_policy)
