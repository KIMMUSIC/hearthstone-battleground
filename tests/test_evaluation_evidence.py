"""Reports must be proven by their action traces before comparison."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluation_evidence.py"
    spec = importlib.util.spec_from_file_location("evaluation_evidence", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def report():
    episode = dict(seed=1, reward=-1.0, wins=0, draws=0, losses=1, survived=False,
                   actions=1, freezes=1, rerolls=0, swaps=0, forced_end_turns=1)
    step = dict(action=3, legal_actions=[0, 3], action_mask=[True, False, False, True],
                reward=-1.0, combat_result=-1, forced_end_turn=True, terminated=True,
                truncated=False)
    return dict(seeds=[1], episodes=[episode], episode_count=1, mean_reward=-1.0,
                combat_count=1, combat_win_rate=0.0, survival_rate=0.0,
                trace=dict(episodes=[dict(seed=1, steps=[step])]))


def test_trace_proves_summary():
    assert module().audit_report(report(), [1])["episodes"] == 1


@pytest.mark.parametrize("corruption", ["illegal", "reward", "freeze", "duplicate", "unterminated"])
def test_corrupt_evidence_rejected(corruption):
    data = deepcopy(report())
    step = data["trace"]["episodes"][0]["steps"][0]
    if corruption == "illegal":
        step["legal_actions"] = [0]
    elif corruption == "reward":
        data["mean_reward"] = 1.0
    elif corruption == "freeze":
        data["episodes"][0]["freezes"] = 0
    elif corruption == "duplicate":
        data["episodes"].append(data["episodes"][0])
    else:
        step["terminated"] = False
    with pytest.raises(ValueError):
        module().audit_report(data, [1])
