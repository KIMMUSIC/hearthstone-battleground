from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_lobby_043 import (  # noqa: E402
    _compare_metric,
    _validate_spec,
    expected_jobs,
    expected_sampling_indices,
    sampling_probabilities,
    verify_sampling,
)


def spec():
    return {
        "seeds": [7, 17, 27],
        "arms": {"control": "uniform-choice", "candidate": "balanced-choice"},
        "train": {
            "max_steps": 8192,
            "max_seconds": 60.0,
            "threads": 1,
            "seed": 7,
            "n_steps": 32,
            "batch_size": 32,
            "n_epochs": 4,
            "gamma": 1.0,
            "learner_seat": 0,
            "max_rounds": 100,
            "heuristic_opponents": 7,
            "observation_scale": "fixed-v1",
        },
        "eval_seeds": list(range(20001, 20021)),
        "replay_seeds": [20001, 20002],
        "limits": {"phase_seconds": 180, "rss_bytes": 2147483648, "disk_bytes": 1073741824},
        "dataset": {"train_seeds": list(range(100)), "dev_seeds": list(range(20001, 20021)), "max_rounds": 100},
        "imitation": {"updates": 1024, "batch_size": 64, "learning_rate": 0.0003, "max_seconds": 60},
    }


def test_043_spec_and_job_contract():
    plan = spec()

    _validate_spec(plan)

    jobs = expected_jobs(plan)
    assert len(jobs) == 14
    assert jobs[:3] == [("dataset", 7, "student"), ("train", 7, "control"), ("train", 7, "candidate")]
    assert jobs[-3:] == [("evaluate", 27, "control"), ("evaluate", 27, "candidate"), ("baseline", 7, "control")]
    changed = spec()
    changed["arms"]["candidate"] = "uniform-choice"
    with pytest.raises(ValueError, match="arm"):
        _validate_spec(changed)


def test_sampling_probabilities_and_replay_are_deterministic(tmp_path):
    actions = np.array([0, 0, 1, 2, 2, 2], dtype=np.int64)

    uniform = sampling_probabilities(actions, "uniform-choice")
    balanced = sampling_probabilities(actions, "balanced-choice")

    np.testing.assert_allclose(uniform, np.full(6, 1 / 6))
    np.testing.assert_allclose(balanced, np.array([1/6, 1/6, 1/4, 5/36, 5/36, 5/36]))
    assert balanced[2] > balanced[0] > balanced[3]

    indices, probabilities = expected_sampling_indices(actions, "balanced-choice", seed=7, updates=3, batch_size=4)
    np.savez_compressed(tmp_path / "sampling.npz", indices=indices, probabilities=probabilities)
    proof = verify_sampling(tmp_path, actions, "balanced-choice", 7, updates=3, batch_size=4)
    assert proof["indices_shape"] == [3, 4]
    assert sum(proof["action_exposure_histogram"].values()) == 12
    assert proof["metadata"] == {
        "mode": "balanced-choice",
        "protocol": "choice-probabilities-v1",
        "hash": proof["sampling_sha256"],
        "histogram": proof["action_exposure_histogram"],
    }

    bad = indices.copy()
    bad[0, 0] = (bad[0, 0] + 1) % len(actions)
    np.savez_compressed(tmp_path / "sampling.npz", indices=bad, probabilities=probabilities)
    with pytest.raises(ValueError, match="indices"):
        verify_sampling(tmp_path, actions, "balanced-choice", 7, updates=3, batch_size=4)


def test_classification_comparison_requires_macro_and_confusion():
    expected = {
        "accuracy": 0.75,
        "macro_action_accuracy": 0.5,
        "nll": 1.25,
        "rows": 4,
        "per_action": {"0": {"count": 2, "accuracy": 1.0}},
        "confusion_matrix": [[2] + [0] * 36] + [[0] * 37 for _ in range(36)],
    }

    _compare_metric(dict(expected), expected, "ok")
    missing = dict(expected)
    missing.pop("macro_action_accuracy")
    with pytest.raises(ValueError, match="macro_action_accuracy"):
        _compare_metric(missing, expected, "missing")
    wrong = dict(expected)
    wrong["confusion_matrix"] = [[0] * 37 for _ in range(37)]
    wrong["confusion_matrix"][0][1] = 2
    with pytest.raises(ValueError, match="confusion_matrix"):
        _compare_metric(wrong, expected, "wrong")
