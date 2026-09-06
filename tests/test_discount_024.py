"""Operational checks for the fixed local discount experiment."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("discount_024", ROOT / "scripts/discount_024.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_schedule_has_six_paired_fresh_trains_then_twelve_evaluations():
    jobs = runner.jobs(runner.read(ROOT / "configs/discount024_campaign.json"))
    assert len(jobs) == 18 and len({runner.identity(j) for j in jobs}) == 18
    assert all(j[0] == "train" for j in jobs[:6])
    for seed in (7, 17, 27):
        assert [(j[2], j[3]) for j in jobs[6:] if j[1] == seed] == [
            ("control", "initial"), ("control", "final"), ("candidate", "initial"), ("candidate", "final")]


def test_no_seed_selection():
    config = runner.read(ROOT / "configs/discount024_campaign.json")
    config["seeds"] = [7]
    with pytest.raises(ValueError):
        runner.jobs(config)


def test_preserves_existing_package(tmp_path):
    (tmp_path / "handoff/existing").mkdir(parents=True)
    marker = tmp_path / "handoff/existing/evidence"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        runner.package(tmp_path, "existing")
    assert marker.read_text() == "preserve"


def test_partial_campaign_cannot_be_audited(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "verify_files", lambda root: None)
    runner.dump(tmp_path / "configs/discount024_campaign.json", runner.read(ROOT / "configs/discount024_campaign.json"))
    runner.dump(tmp_path / "outputs/driver-status.json", {"status": "failed", "completed_phases": 17})
    with pytest.raises(ValueError, match="incomplete"):
        runner.audit(tmp_path)
    assert not (tmp_path / "audit.json").exists()


def test_actual_checkpoint_counters_and_gamma(tmp_path):
    from hearthstone_ai.training import train
    result = train(dict(max_steps=32, max_seconds=60, threads=1, n_epochs=4, gamma=1.0), tmp_path / "model")
    model = tmp_path / "model" / result["checkpoint"]
    assert runner.checkpoint(model, 32, 1.0)
    for steps, gamma in ((64, 1.0), (32, 0.99)):
        with pytest.raises(ValueError, match="counters or gamma"):
            runner.checkpoint(model, steps, gamma)


def test_input_tampering_rejected(tmp_path):
    path = tmp_path / "source.py"
    path.write_text("original")
    runner.dump(tmp_path / "FILES.json", {"source.py": runner.digest(path)})
    path.write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.verify_files(tmp_path)


def test_declared_limits_match_precommitted_experiment():
    config = json.loads((ROOT / "configs/discount024_campaign.json").read_text())
    assert config["train"]["max_steps"] * 6 == 49152
    assert config["train"]["threads"] == 1 and config["train"]["max_seconds"] == 60
    assert config["limits"] == {"phase_seconds": 180, "train_seconds": 360,
                                "evaluate_seconds": 600, "total_seconds": 1200,
                                "rss_bytes": 2**31, "disk_bytes": 2**30}


def test_real_phase_train_and_evaluate_smoke(tmp_path, monkeypatch):
    import torch
    config = runner.read(ROOT / "configs/discount024_campaign.json")
    config["train"]["max_steps"] = 32
    runner.dump(tmp_path / "configs/discount024_campaign.json", config)
    evaluation = runner.read(ROOT / "configs/expansion020_eval.json")
    evaluation["seeds"] = [20001, 20002]
    evaluation["opponent_path"] = str(ROOT / evaluation["opponent_path"])
    runner.dump(tmp_path / "configs/expansion020_eval.json", evaluation)
    monkeypatch.setattr(runner, "verify_files", lambda root: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda value: None)
    runner.phase(tmp_path, 0)
    runner.phase(tmp_path, 6)
    report = runner.read(tmp_path / "outputs/evaluate-s007-control-initial/report.json")
    assert report["episode_count"] == 2
    assert report["seeds"] == [20001, 20002]
