"""Candidate selection cannot cherry-pick seeds or bypass baseline quality."""

import importlib.util
from pathlib import Path

import pytest


def decide(rows):
    path = Path(__file__).resolve().parents[1] / "scripts/discount_decision.py"
    spec = importlib.util.spec_from_file_location("discount_decision", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decide(rows)


def reports(finals=(3, 3, 3), control=2):
    return {str(seed): {
        arm: {"initial": {"mean_reward": -6, "survival_rate": 0},
              "final": {"mean_reward": final if arm == "candidate" else control,
                        "survival_rate": 0.9}}
        for arm in ("control", "candidate")}
        for seed, final in zip((7, 17, 27), finals, strict=True)}


def test_selects_all_three_or_none():
    result = decide(reports())
    assert result["candidate_selected"]
    assert result["selected_training_seeds"] == [7, 17, 27]
    assert result["paired_reward_delta_descriptive_95"] == [1, 1]


def test_pairwise_improvement_is_not_enough_without_heuristic_gate():
    result = decide(reports((-4, -4, -4), control=-6))
    assert result["hypothesis_supported"]
    assert not result["candidate_selected"]
    assert result["selected_training_seeds"] == []


def test_one_good_seed_cannot_hide_two_regressions():
    result = decide(reports((8, 2, 2), control=3))
    assert result["mean_paired_reward_delta"] > 0.5
    assert not result["hypothesis_supported"]


def test_survival_regression_blocks_selection():
    rows = reports()
    for row in rows.values():
        row["candidate"]["final"]["survival_rate"] = 0.5
    assert not decide(rows)["candidate_selected"]


def test_candidate_must_improve_its_own_initial_policy():
    rows = reports()
    for row in rows.values():
        row["candidate"]["initial"]["mean_reward"] = 3
    result = decide(rows)
    assert result["hypothesis_supported"]
    assert not result["candidate_learning_progress"]
    assert not result["candidate_selected"]


@pytest.mark.parametrize("fault", ["missing_seed", "missing_initial", "nan", "bad_survival"])
def test_incomplete_or_invalid_evidence_rejected(fault):
    rows = reports()
    if fault == "missing_seed":
        del rows["27"]
    elif fault == "missing_initial":
        del rows["7"]["candidate"]["initial"]
    elif fault == "nan":
        rows["7"]["candidate"]["final"]["mean_reward"] = float("nan")
    else:
        rows["7"]["candidate"]["final"]["survival_rate"] = 1.1
    with pytest.raises(ValueError):
        decide(rows)
