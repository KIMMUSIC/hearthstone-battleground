import importlib.util
import json
from pathlib import Path
import sys

import pytest


scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))

baseline_spec = importlib.util.spec_from_file_location("baseline_020", scripts / "baseline_020.py")
baseline = importlib.util.module_from_spec(baseline_spec)
baseline_spec.loader.exec_module(baseline)

runner_spec = importlib.util.spec_from_file_location(
    "run_expansion_019", scripts / "run_expansion_019.py"
)
runner = importlib.util.module_from_spec(runner_spec)
runner_spec.loader.exec_module(runner)


def write_dev_opponents(root: Path) -> None:
    configs = root / "configs"
    configs.mkdir(exist_ok=True)
    (configs / "expansion020_dev_opponents.json").write_text(
        json.dumps(
            {
                "version": "test-dev",
                "purpose": "development evaluation only; never training",
                "opponents": [{"tier": 1, "board": []}],
            }
        ),
        encoding="utf-8",
    )


def write_eval_config(root: Path, **overrides) -> None:
    configs = root / "configs"
    configs.mkdir(exist_ok=True)
    config = {
        "seeds": list(range(20001, 20101)),
        "opponent_path": "configs/expansion020_dev_opponents.json",
        "max_turns": 8,
        "max_actions": 24,
    }
    config.update(overrides)
    (configs / "expansion020_eval.json").write_text(json.dumps(config), encoding="utf-8")


def report(policy: str, seeds: list[int], reward: float = 1.0) -> dict:
    episodes = [
        {
            "seed": seed,
            "reward": reward,
            "wins": 1,
            "draws": 0,
            "losses": 0,
            "damage_taken": 0,
            "damage_dealt": 3,
            "forced_end_turns": 0,
            "rerolls": 0,
            "swaps": 0,
            "actions": 8,
            "hp": 40,
            "last_turn": 8,
            "survived": True,
            "termination_reason": "horizon",
        }
        for seed in seeds
    ]
    return {
        "policy": policy,
        "seeds": seeds,
        "episodes": episodes,
        "episode_count": len(episodes),
        "combat_count": len(episodes),
        "combat_win_rate": 1.0,
        "survival_rate": 1.0,
        "mean_reward": reward,
        "trace": {"episodes": []},
    }


def test_baseline_phase_writes_two_100_episode_reports_and_repro_check(tmp_path, monkeypatch):
    write_dev_opponents(tmp_path)
    write_eval_config(tmp_path)
    calls = []

    def fake_evaluate(**kwargs):
        calls.append((kwargs["policy"], tuple(kwargs["seeds"]), kwargs["opponent_path"]))
        return report(kwargs["policy"], kwargs["seeds"])

    monkeypatch.setattr(baseline, "configure_torch_threads", lambda: None)
    monkeypatch.setattr(baseline, "evaluate_policy", fake_evaluate)
    result = baseline.phase_baseline(tmp_path, tmp_path / "outputs", {"source_sha256": "abc"})
    assert sorted(result["reports"]) == ["heuristic", "random"]
    assert result["config"]["seeds"] == list(range(20001, 20101))
    assert result["reserved_final_inputs_used"] is False
    assert all(call[2] == (tmp_path / "configs/expansion020_dev_opponents.json").resolve() for call in calls)
    assert calls == [
        ("heuristic", tuple(range(20001, 20101)), calls[0][2]),
        ("heuristic", tuple(range(20001, 20004)), calls[0][2]),
        ("random", tuple(range(20001, 20101)), calls[0][2]),
        ("random", tuple(range(20001, 20004)), calls[0][2]),
    ]
    assert (tmp_path / "outputs/baseline/heuristic.json").exists()
    assert (tmp_path / "outputs/baseline/random.json").exists()
    assert (tmp_path / "outputs/baseline/summary.json").exists()


def test_baseline_rejects_reproducibility_drift(tmp_path, monkeypatch):
    write_dev_opponents(tmp_path)
    write_eval_config(tmp_path)
    call_count = {"heuristic": 0, "random": 0}

    def fake_evaluate(**kwargs):
        call_count[kwargs["policy"]] += 1
        reward = 2.0 if kwargs["policy"] == "heuristic" and call_count["heuristic"] == 2 else 1.0
        return report(kwargs["policy"], kwargs["seeds"], reward=reward)

    monkeypatch.setattr(baseline, "configure_torch_threads", lambda: None)
    monkeypatch.setattr(baseline, "evaluate_policy", fake_evaluate)
    with pytest.raises(RuntimeError, match="reproducibility"):
        baseline.phase_baseline(tmp_path, tmp_path / "outputs", {"source_sha256": "abc"})


def test_baseline_config_requires_dev_seeds_and_dev_opponents(tmp_path):
    write_dev_opponents(tmp_path)
    write_eval_config(tmp_path, seeds=list(range(1, 101)))
    with pytest.raises(ValueError, match="20001..20100"):
        baseline.load_baseline_config(tmp_path)
    write_eval_config(tmp_path, opponent_path="experiments/reserved/expansion020_final_opponents.json")
    with pytest.raises(ValueError, match="dev_opponents"):
        baseline.load_baseline_config(tmp_path)


def test_baseline_never_reads_reserved_final_opponents(tmp_path, monkeypatch):
    write_dev_opponents(tmp_path)
    write_eval_config(tmp_path)
    original = Path.read_text

    def guarded_read(path, *args, **kwargs):
        if Path(path).as_posix().endswith("experiments/reserved/expansion020_final_opponents.json"):
            raise AssertionError("baseline read reserved final opponents")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    monkeypatch.setattr(baseline, "configure_torch_threads", lambda: None)
    monkeypatch.setattr(baseline, "evaluate_policy", lambda **kwargs: report(kwargs["policy"], kwargs["seeds"]))
    baseline.phase_baseline(tmp_path, tmp_path / "outputs", {"source_sha256": "abc"})


def test_training_path_does_not_read_020_dev_or_reserved_inputs(monkeypatch):
    from hearthstone_ai.game import Game

    original = Path.read_text

    def guarded_read(path, *args, **kwargs):
        name = Path(path).name
        if name in {"expansion020_dev_opponents.json", "expansion020_final_opponents.json"}:
            raise AssertionError(f"training path read {name}")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    for seed in range(20001, 20006):
        game = Game(opponent_mode="fixed-v1")
        game.reset(seed)
        assert game.opponent_mode == "fixed-v1"


def test_runner_accepts_baseline_phase_and_blocks_prior_evidence(tmp_path):
    assert runner.PHASE_OUTPUTS["baseline"] == "baseline"
    prior = tmp_path / "outputs" / "baseline"
    prior.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="already attempted"):
        runner.require_fresh_phase(tmp_path, "baseline")
