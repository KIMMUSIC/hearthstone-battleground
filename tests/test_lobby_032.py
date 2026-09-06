"""Real save/load/evaluation integration and fixed-run preservation."""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import lobby_032 as runner  # noqa: E402
import audit_lobby_032 as auditor  # noqa: E402


def test_package_preserves_previous_attempt_and_rejects_escape(tmp_path):
    (tmp_path / "handoff/old").mkdir(parents=True)
    marker = tmp_path / "handoff/old/marker"
    marker.write_text("original")
    with pytest.raises(FileExistsError):
        runner.package(tmp_path, "old")
    assert marker.read_text() == "original"
    with pytest.raises(ValueError):
        runner.package(tmp_path, "../../escape")


def test_real_initial_final_and_baseline_evaluation(tmp_path):
    from hearthstone_ai.lobby_training import train
    spec = runner.read(ROOT / "configs/lobby032_smoke.json")
    spec["train"]["max_steps"] = 32
    spec["eval_seeds"] = [20001, 20002]
    spec["replay_seeds"] = [20001]
    result = train(spec["train"], tmp_path / "outputs/train")
    assert result["num_timesteps"] == 32
    evidence, policy = auditor.model_evidence(tmp_path / "outputs/train" / result["checkpoint"] / "model.zip")
    assert evidence["num_timesteps"] == 32 and evidence["_n_updates"] == 4
    assert policy
    reports = runner.evaluate(tmp_path, spec)
    assert set(reports) == {"initial", "final", "heuristic"}
    for label in reports:
        report = runner.read(tmp_path / f"outputs/evaluate/{label}.json")
        assert [e["seed"] for e in report["episodes"]] == spec["eval_seeds"]
        assert all(e["terminated"] and e["rank"] is not None for e in report["episodes"])
    assert reports["final"]["replay_seeds"] == [20001]
    with pytest.raises(FileExistsError):
        runner.evaluate(tmp_path, spec)


def test_audit_rejects_incomplete_run(tmp_path, monkeypatch):
    monkeypatch.setattr(auditor, "verify_files", lambda root: None)
    runner.dump(tmp_path / "configs/lobby032_smoke.json", runner.read(ROOT / "configs/lobby032_smoke.json"))
    runner.dump(tmp_path / "outputs/driver-status.json", {"status": "failed", "completed_phases": ["train"]})
    with pytest.raises(ValueError, match="Incomplete"):
        auditor.audit(tmp_path)
    assert not (tmp_path / "audit.json").exists()
