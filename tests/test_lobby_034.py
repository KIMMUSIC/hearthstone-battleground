"""Probe decision gates and actual rule-policy replay."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import lobby_034 as probe  # noqa: E402


def test_end_freeze_have_same_terminal_objective_despite_more_actions():
    end = probe.play(42, 7, "end", 100)
    freeze = probe.play(42, 7, "freeze", 100)
    assert end["rank"] == freeze["rank"] == 8
    assert end["reward"] == freeze["reward"] == -1
    assert len(freeze["learner_actions"]) == 24 * len(end["learner_actions"])
    assert freeze == probe.play(42, 7, "freeze", 100)


def test_real_mixed_policy_replay_and_truncation_count():
    first = probe.play(43, 3, "random", 2)
    assert first == probe.play(43, 3, "random", 2)
    assert first["truncated"] and first["rank"] is None
    summary = probe.aggregate([first])
    assert summary["ranked"] == 0 and summary["mean_rank"] is None
    assert summary["truncated"] == 1


def test_decision_cannot_select_all_random_when_mixed_gate_fails():
    fake = {"h7-random": {"last_fraction": 1},
            "h3-random": {"distinct_ranks": [4, 6, 8], "last_fraction": .5, "rank_variance": 1,
                          "truncated": 0, "mean_rank": 6},
            "h3-heuristic": {"truncated": 0, "mean_rank": 2}}
    assert probe.decision(fake)["selected_heuristic_opponents"] == 3
    fake["h3-random"]["last_fraction"] = .8
    assert probe.decision(fake)["selected_heuristic_opponents"] is None
