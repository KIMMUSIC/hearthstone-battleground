import json
from dataclasses import replace

from hearthstone_ai.artifacts import ROOT, file_hash
from hearthstone_ai.evaluation import evaluate, heuristic_action
from hearthstone_ai.game import Game


def test_evaluation_repeats_without_mutating_inputs():
    source = ROOT / "configs/eval_opponents.json"
    original = file_hash(source)
    first = evaluate(policy="heuristic", seeds=[101, 102])
    # Unrelated episodes cannot contaminate a fixed evaluation.
    evaluate(policy="random", seeds=[999])
    assert first == evaluate(policy="heuristic", seeds=[101, 102])
    assert original == file_hash(source)
    assert first["combat_count"] > 0
    assert all(row["termination_reason"] in {"death", "horizon"} for row in first["episodes"])


def test_random_baseline_is_seeded():
    assert evaluate(policy="random", seeds=[10]) == evaluate(policy="random", seeds=[10])


def test_freeze_usage_is_reported_separately_from_other_actions(monkeypatch):
    monkeypatch.setattr("hearthstone_ai.evaluation.heuristic_action", lambda game: 3)
    result = evaluate(policy="heuristic", seeds=[17], max_turns=2, max_actions=3, trace=True)
    row = result["episodes"][0]
    assert row["freezes"] == row["actions"] == 6
    assert row["rerolls"] == row["swaps"] == 0
    assert row["forced_end_turns"] == 2


def test_evaluation_trace_is_opt_in_and_records_action_context(tmp_path):
    trace_path = tmp_path / "trace.json"
    result = evaluate(
        policy="heuristic",
        seeds=[123],
        max_turns=2,
        max_actions=3,
        trace_output=trace_path,
    )
    trace = json.loads(trace_path.read_text())
    assert result["episodes"][0]["actions"] == len(trace["episodes"][0]["steps"])
    first = trace["episodes"][0]["steps"][0]
    assert trace["policy"] == "heuristic"
    assert trace["seeds"] == [123]
    assert len(first["action_mask"]) == 37
    assert len(first["player_observation"]) == 9
    assert first["action"] in first["legal_actions"]
    assert {"turn", "gold", "board_count", "hand_count", "forced_end_turn"} <= set(first)


def test_evaluation_can_embed_trace_in_result():
    result = evaluate(policy="heuristic", seeds=[123], max_turns=2, max_actions=3, trace=True)
    assert result["trace"]["episodes"][0]["seed"] == 123
    assert result["episodes"][0]["actions"] == len(result["trace"]["episodes"][0]["steps"])


def test_heuristic_does_not_chase_disabled_triples():
    game = Game()
    game.reset(17)
    game.catalog.by_id[96763] = replace(game.catalog.by_id[96763], triple_allowed=False)
    game.board = [game.catalog.spawn(96763, 100), game.catalog.spawn(96763, 101)]
    game.hand = []
    game.shop[0] = game.catalog.spawn(96763, 102)
    game.shop[1] = game.catalog.spawn(99159, 103)
    game.shop[2] = None
    assert heuristic_action(game) == 5
