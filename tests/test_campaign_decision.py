"""Predetermined gates apply across all independent training seeds."""

import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/campaign_decision.py"
    spec = importlib.util.spec_from_file_location("campaign_decision", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def rows(deltas):
    return {seed: {step: {"mean_reward": 0 if step == 0 else delta,
                          "survival_rate": 0.5}
                   for step in (0, 49152, 100000)}
            for seed, delta in zip((7, 17, 27), deltas, strict=True)}


GATES = dict(mean_seed_reward_delta_min=0.5, positive_seeds_min=2,
             mean_seed_survival_delta_min=-0.05, million_step_candidate_heuristic_reward_gap_min=-0.5)


def test_no_cherry_picking_one_good_seed():
    result = module().decide(rows([6, -1, -1]), [7, 17, 27], GATES, 0)
    assert result["mean_reward_delta"] > 0.5
    assert not result["learning_progress"] and not result["million_step_candidate"]


def test_heuristic_gap_is_additional_gate():
    result = module().decide(rows([1, 1, 1]), [7, 17, 27], GATES, 3.4)
    assert result["learning_progress"] and not result["million_step_candidate"]


def test_missing_seed_cannot_pass():
    data = rows([1, 1, 1])
    del data[17]
    with pytest.raises(ValueError):
        module().decide(data, [7, 17, 27], GATES, 0)


def test_all_gates_and_seed_unit_interval():
    result = module().decide(rows([1, 1, 1]), [7, 17, 27], GATES, 1.5)
    assert result["million_step_candidate"]
    assert result["independent_training_seeds"] == 3
    assert result["reward_delta_descriptive_bootstrap_95"] == [1, 1]
