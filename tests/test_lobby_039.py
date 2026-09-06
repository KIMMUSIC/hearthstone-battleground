"""Rollout changes preserve optimizer work while changing SB3 epoch counters."""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import lobby_035 as runner  # noqa: E402
from audit_lobby_032 import model_evidence  # noqa: E402
from audit_lobby_035 import optimizer_steps  # noqa: E402


def test_039_changes_only_rollout_length():
    spec = runner.read(ROOT / "configs/lobby039_comparison.json")
    control = runner.training_config(spec, 17, "control")
    candidate = runner.training_config(spec, 17, "candidate")
    assert control.pop("n_steps") == 32
    assert candidate.pop("n_steps") == 128
    assert control == candidate
    assert len(runner.jobs(spec)) == 13


def test_real_models_have_equal_adam_steps_and_distinct_sb3_counts(tmp_path, monkeypatch):
    import torch
    spec = runner.read(ROOT / "configs/lobby039_comparison.json")
    spec["train"]["max_steps"] = 256
    spec["eval_seeds"] = [20001]
    spec["replay_seeds"] = [20001]
    filename = "lobby039_comparison.json"
    runner.dump(tmp_path / "configs" / filename, spec)
    monkeypatch.setattr(runner, "verify_files", lambda root: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda n: None)
    initial_weights = []
    for index, arm, sb3_count in ((0, "control", 32), (1, "candidate", 8)):
        result = runner.phase(tmp_path, index, filename)
        train = tmp_path / f"outputs/train-s007-{arm}"
        final_path = train / result["checkpoint"] / "model.zip"
        data, _ = model_evidence(final_path)
        assert data["_n_updates"] == sb3_count
        assert optimizer_steps(final_path, 32)["min_step"] == 32
        with pytest.raises(ValueError, match="Adam steps"):
            optimizer_steps(final_path, 31)
        initial_weights.append(model_evidence(train / result["initial_checkpoint"] / "model.zip")[1])
    assert all(torch.equal(initial_weights[0][k], initial_weights[1][k]) for k in initial_weights[0])
    reports = runner.phase(tmp_path, 7, filename)
    assert reports["final"]["truncated"] == 0 and reports["final"]["replay_seeds"] == [20001]
