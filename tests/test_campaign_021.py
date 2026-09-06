import argparse
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import pytest


scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))

campaign_spec = importlib.util.spec_from_file_location("campaign_021", scripts / "campaign_021.py")
campaign = importlib.util.module_from_spec(campaign_spec)
campaign_spec.loader.exec_module(campaign)

runner_spec = importlib.util.spec_from_file_location("run_campaign_021", scripts / "run_campaign_021.py")
runner = importlib.util.module_from_spec(runner_spec)
runner_spec.loader.exec_module(runner)

package_spec = importlib.util.spec_from_file_location(
    "package_campaign_021", scripts / "package_campaign_021.py"
)
packager = importlib.util.module_from_spec(package_spec)
package_spec.loader.exec_module(packager)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_model_checkpoint(path: Path, timesteps: int, updates: int, additional: int = 0) -> None:
    path.mkdir(parents=True)
    with zipfile.ZipFile(path / "model.zip", "w") as zf:
        zf.writestr("data", json.dumps({"num_timesteps": timesteps, "_n_updates": updates}))
    write_json(path / "metadata.json", {"num_timesteps": timesteps, "additional_steps": additional})


def write_campaign_inputs(root: Path) -> None:
    write_json(
        root / "experiments/expansion020_campaign.json",
        {
            "status": "designed-not-executed",
            "training_seeds": [7, 17, 27],
            "steps_per_seed": 100000,
            "segment_additional_steps": [8192] * 12 + [1696],
            "evaluation_milestones": [0, 49152, 100000],
            "expected_updates_per_seed": 12500,
            "limits": {
                "cpu_threads": 1,
                "train_seconds_per_segment": 60,
                "supervisor_seconds_per_phase": 180,
                "sampled_rss_bytes": 2147483648,
                "all_outputs_and_return_zip_bytes": 1073741824,
                "campaign_train_wall_seconds": 1200,
                "campaign_dev_eval_wall_seconds": 600,
                "campaign_total_wall_seconds": 1800,
                "automatic_retry": False,
            },
        },
    )
    write_json(
        root / "configs/expansion019_train.json",
        {
            "max_steps": 8192,
            "max_seconds": 60,
            "threads": 1,
            "seed": 7,
            "n_steps": 32,
            "batch_size": 32,
            "n_epochs": 4,
            "checkpoint_interval": 8192,
            "max_turns": 8,
            "max_actions": 24,
            "opponent_mode": "fixed-v1",
        },
    )
    write_json(
        root / "configs/expansion020_eval.json",
        {
            "seeds": list(range(20001, 20101)),
            "opponent_path": "configs/expansion020_dev_opponents.json",
            "max_turns": 8,
            "max_actions": 24,
        },
    )
    write_json(
        root / "configs/expansion020_dev_opponents.json",
        {"version": "dev", "opponents": [{"tier": 1, "board": []}]},
    )


def write_runner_plan(root: Path, *, train_wall: float = 1200, dev_wall: float = 600, total_wall: float = 1800) -> None:
    write_json(
        root / "execution.json",
        {
            "python": sys.executable,
            "source_sha256": "abc",
            "minimum_free_bytes": 1,
            "output_disk_limit_bytes": 1024 * 1024,
            "campaign_spec": {
                "training_seeds": [7, 17, 27],
                "segment_additional_steps": [8192] * 12 + [1696],
                "evaluation_milestones": [0, 49152, 100000],
                "limits": {
                    "campaign_train_wall_seconds": train_wall,
                    "campaign_dev_eval_wall_seconds": dev_wall,
                    "campaign_total_wall_seconds": total_wall,
                },
            },
        },
    )
    write_json(root / "FILES.json", {})


def evaluation_report(seeds: list[int]) -> dict:
    episodes = []
    traces = []
    for seed in seeds:
        episodes.append(
            {
                "seed": seed,
                "reward": 1.0,
                "wins": 1,
                "draws": 0,
                "losses": 0,
                "damage_taken": 0,
                "damage_dealt": 3,
                "forced_end_turns": 0,
                "rerolls": 0,
                "freezes": 0,
                "swaps": 0,
                "actions": 1,
                "hp": 40,
                "last_turn": 8,
                "survived": True,
                "termination_reason": "horizon",
            }
        )
        traces.append(
            {
                "seed": seed,
                "steps": [
                    {
                        "action": 0,
                        "action_mask": [True],
                        "legal_actions": [0],
                        "reward": 1.0,
                        "combat_result": 1,
                        "forced_end_turn": False,
                        "terminated": True,
                        "truncated": False,
                    }
                ],
            }
        )
    return {
        "policy": "model",
        "seeds": seeds,
        "episodes": episodes,
        "episode_count": len(seeds),
        "combat_count": len(seeds),
        "combat_win_rate": 1.0,
        "survival_rate": 1.0,
        "mean_reward": 1.0,
        "trace": {"episodes": traces},
    }


def test_model_counters_reads_zip_data_and_rejects_metadata_drift(tmp_path):
    checkpoint = tmp_path / "checkpoint-000000008192"
    write_model_checkpoint(checkpoint, 8192, 1024, additional=8192)
    assert campaign.model_counters(checkpoint) == {
        "num_timesteps": 8192,
        "_n_updates": 1024,
        "metadata_additional_steps": 8192,
    }
    write_json(checkpoint / "metadata.json", {"num_timesteps": 1, "additional_steps": 8192})
    with pytest.raises(RuntimeError, match="metadata disagrees"):
        campaign.model_counters(checkpoint)


def test_train_segment_uses_resume_and_writes_progress_counters(tmp_path, monkeypatch):
    write_campaign_inputs(tmp_path)
    seed_dir = tmp_path / "outputs/seed-007"
    prior = seed_dir / "segment-01/run/checkpoint-000000008192"
    write_model_checkpoint(prior, 8192, 1024, additional=8192)
    write_json(
        seed_dir / "segment-01/summary.json",
        {
            "status": {"num_timesteps": 8192},
            "final_checkpoint": str(prior),
        },
    )
    seen = {}

    def fake_train(config, output_dir, resume):
        seen["config"] = config
        seen["resume"] = resume
        write_model_checkpoint(output_dir / "checkpoint-000000008192", 8192, 1024, additional=0)
        write_model_checkpoint(output_dir / "checkpoint-000000016384", 16384, 2048, additional=8192)
        return {
            "status": "completed",
            "stop_reason": "steps",
            "initial_checkpoint": "checkpoint-000000008192",
            "checkpoint": "checkpoint-000000016384",
            "additional_steps": 8192,
            "num_timesteps": 16384,
        }

    monkeypatch.setattr(campaign, "configure_torch_threads", lambda: None)
    monkeypatch.setattr(campaign, "train_model", fake_train)
    result = campaign.phase_train(tmp_path, tmp_path / "outputs", {"source_sha256": "abc"}, 7, 2)
    assert seen["resume"] == prior
    assert seen["config"]["seed"] == 7
    assert result["final_counters"]["_n_updates"] == 2048
    progress = json.loads((tmp_path / "outputs/progress.json").read_text(encoding="utf-8"))
    assert progress["last_event"]["num_timesteps"] == 16384


def test_train_segment_rejects_partial_or_time_stopped_status(tmp_path, monkeypatch):
    write_campaign_inputs(tmp_path)

    def partial_train(config, output_dir, resume):
        write_model_checkpoint(output_dir / "checkpoint-000000000000", 0, 0, additional=0)
        write_model_checkpoint(output_dir / "checkpoint-000000004096", 4096, 512, additional=4096)
        return {
            "status": "completed",
            "stop_reason": "time",
            "initial_checkpoint": "checkpoint-000000000000",
            "checkpoint": "checkpoint-000000004096",
            "additional_steps": 4096,
            "num_timesteps": 4096,
        }

    monkeypatch.setattr(campaign, "configure_torch_threads", lambda: None)
    monkeypatch.setattr(campaign, "train_model", partial_train)
    with pytest.raises(RuntimeError, match="did not complete by steps"):
        campaign.phase_train(tmp_path, tmp_path / "outputs", {"source_sha256": "abc"}, 7, 1)


def test_evaluate_uses_dev_config_trace_audit_and_preserves_checkpoint(tmp_path, monkeypatch):
    write_campaign_inputs(tmp_path)
    checkpoint = tmp_path / "outputs/seed-007/segment-01/run/checkpoint-000000000000"
    write_model_checkpoint(checkpoint, 0, 0, additional=0)
    write_json(
        tmp_path / "outputs/seed-007/segment-01/summary.json",
        {"initial_checkpoint": str(checkpoint)},
    )

    def fake_evaluate(**kwargs):
        assert kwargs["policy"] == "model"
        assert kwargs["opponent_path"] == (tmp_path / "configs/expansion020_dev_opponents.json").resolve()
        assert kwargs["trace"] is True
        return evaluation_report(kwargs["seeds"])

    monkeypatch.setattr(campaign, "configure_torch_threads", lambda: None)
    monkeypatch.setattr(campaign, "evaluate_policy", fake_evaluate)
    result = campaign.phase_evaluate(tmp_path, tmp_path / "outputs", {"source_sha256": "abc"}, 7, 0)
    assert result["trace_audit"]["trace_verified"] is True
    assert result["report"]["episode_count"] == 100
    assert (tmp_path / "outputs/seed-007/eval-000000/model.json").exists()


def test_evaluate_rejects_trace_mismatch(tmp_path, monkeypatch):
    write_campaign_inputs(tmp_path)
    checkpoint = tmp_path / "outputs/seed-007/segment-01/run/checkpoint-000000000000"
    write_model_checkpoint(checkpoint, 0, 0, additional=0)
    write_json(tmp_path / "outputs/seed-007/segment-01/summary.json", {"initial_checkpoint": str(checkpoint)})
    bad = evaluation_report(list(range(20001, 20101)))
    bad["trace"]["episodes"][0]["steps"][0]["reward"] = 2.0
    monkeypatch.setattr(campaign, "configure_torch_threads", lambda: None)
    monkeypatch.setattr(campaign, "evaluate_policy", lambda **kwargs: bad)
    with pytest.raises(ValueError, match="Trace reward mismatch"):
        campaign.phase_evaluate(tmp_path, tmp_path / "outputs", {"source_sha256": "abc"}, 7, 0)


def test_runner_blocks_repeated_phase_and_checks_running_budget(tmp_path):
    prior = tmp_path / "outputs/control-train-s007-seg01"
    prior.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="already attempted"):
        runner.require_fresh_phase(tmp_path, "train-s007-seg01")
    plan = {
        "campaign_spec": {
            "limits": {
                "campaign_train_wall_seconds": 1200,
                "campaign_dev_eval_wall_seconds": 600,
                "campaign_total_wall_seconds": 1800,
            }
        }
    }
    assert runner.budget_status(plan, "train", {"train": 1199.9, "evaluate": 0, "all": 1199.9}, 0.2) == "cumulative_wall_exceeded"
    assert runner.budget_status(plan, "evaluate", {"train": 1200, "evaluate": 599, "all": 1799.9}, 0.2) == "campaign_total_wall_exceeded"
    assert runner.budget_status(
        plan,
        "train",
        {"train": 0, "evaluate": 0, "all": 0},
        0.0,
        campaign_wall_elapsed=1800.1,
    ) == "campaign_total_wall_exceeded"


def test_runner_campaign_lock_blocks_second_phase_until_released(tmp_path):
    first = runner.acquire_campaign_lock(tmp_path, "train-s007-seg01")
    with pytest.raises(RuntimeError, match="already running"):
        runner.acquire_campaign_lock(tmp_path, "eval-s007-step000000")
    runner.release_campaign_lock(tmp_path, first)
    second = runner.acquire_campaign_lock(tmp_path, "eval-s007-step000000")
    runner.release_campaign_lock(tmp_path, second)


def test_runner_rejects_incomplete_prior_control_before_later_phase(tmp_path):
    (tmp_path / "outputs/control-train-s007-seg01").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="Incomplete prior control"):
        runner.require_no_incomplete_controls(tmp_path)
    write_json(tmp_path / "outputs/control-train-s007-seg01/result.json", {"status": "command_failed"})
    with pytest.raises(RuntimeError, match="Incomplete prior control"):
        runner.require_no_incomplete_controls(tmp_path)


def test_runner_uses_end_to_end_elapsed_for_prior_budget(tmp_path):
    write_json(
        tmp_path / "outputs/control-train-s017-seg01/result.json",
        {"status": "completed", "elapsed_seconds": 1.0, "end_to_end_elapsed_seconds": 2.5},
    )
    assert runner.prior_elapsed(tmp_path) == {"train": 2.5, "evaluate": 0.0, "all": 2.5}


def test_runner_reads_driver_status_for_total_budget(tmp_path, monkeypatch):
    write_json(tmp_path / "outputs/driver-status.json", {"started_unix": 100.0})
    monkeypatch.setattr(runner.time, "time", lambda: 1901.0)
    assert runner.driver_wall_elapsed(tmp_path) == 1801.0


def test_runner_invalid_invocation_does_not_create_phase_dirs(tmp_path, monkeypatch):
    write_runner_plan(tmp_path)
    monkeypatch.setattr(runner, "verify_files", lambda root: None)
    monkeypatch.setattr(runner, "require_runtime", lambda plan: Path(sys.executable))
    args = argparse.Namespace(mode="train", seed=99, segment=1)
    with pytest.raises(ValueError, match="campaign training seeds"):
        runner.run_phase(tmp_path, args)
    assert not (tmp_path / "outputs").exists()


def test_runner_budget_preflight_happens_before_popen_and_phase_dirs(tmp_path, monkeypatch):
    write_runner_plan(tmp_path, train_wall=1.0)
    write_json(
        tmp_path / "outputs/control-train-s017-seg01/result.json",
        {"status": "completed", "end_to_end_elapsed_seconds": 1.1},
    )
    monkeypatch.setattr(runner, "verify_files", lambda root: None)
    monkeypatch.setattr(runner, "require_runtime", lambda plan: Path(sys.executable))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Popen called"))
    args = argparse.Namespace(mode="train", seed=7, segment=1)
    with pytest.raises(RuntimeError, match="budget already exhausted"):
        runner.run_phase(tmp_path, args)
    assert not (tmp_path / "outputs/control-train-s007-seg01").exists()


def test_runner_postpoll_budget_catches_quick_process(tmp_path, monkeypatch):
    write_runner_plan(tmp_path, train_wall=0.000001)

    class DoneProcess:
        pid = 1234

        def __init__(self, argv, **kwargs):
            output = Path(argv[argv.index("--output") + 1])
            write_json(output / "result.json", {"status": "completed"})

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(runner, "verify_files", lambda root: None)
    monkeypatch.setattr(runner, "require_runtime", lambda plan: Path(sys.executable))
    monkeypatch.setattr(runner.subprocess, "Popen", DoneProcess)
    ticks = iter([0.0, 1.0, 1.0, 1.0])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(ticks))
    args = argparse.Namespace(mode="train", seed=7, segment=1)
    result = runner.run_phase(tmp_path, args)
    assert result["status"] == "cumulative_wall_exceeded"
    assert (tmp_path / "returns/return-train-s007-seg01.zip").exists()


def test_runner_incremental_zip_excludes_frozen_input_bytes(tmp_path):
    write_json(tmp_path / "execution.json", {"source_sha256": "abc"})
    write_json(tmp_path / "FILES.json", {"source.txt": "not-rebundled"})
    (tmp_path / "source.txt").write_text("large frozen input", encoding="utf-8")
    phase = tmp_path / "outputs/seed-007/segment-01"
    control = tmp_path / "outputs/control-train-s007-seg01"
    phase.mkdir(parents=True)
    control.mkdir(parents=True)
    (phase / "summary.json").write_text("{}", encoding="utf-8")
    (control / "result.json").write_text("{}", encoding="utf-8")
    package = runner.archive_phase(tmp_path, "train-s007-seg01", 1024 * 1024)
    with zipfile.ZipFile(package["archive"]) as zf:
        names = set(zf.namelist())
    assert "outputs/seed-007/segment-01/summary.json" in names
    assert "outputs/control-train-s007-seg01/result.json" in names
    assert "source.txt" not in names
    assert "FILES.json" in names


def test_training_tiny32_resume_restores_optimizer_and_counters(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("sb3_contrib")
    pytest.importorskip("stable_baselines3")
    if not sys.platform.startswith("linux"):
        pytest.skip("Actual tiny resume integration runs in the declared Linux venv")
    from hearthstone_ai.training import train

    config = {
        "max_steps": 32,
        "max_seconds": 60,
        "threads": 1,
        "seed": 7,
        "n_steps": 32,
        "batch_size": 32,
        "n_epochs": 4,
        "checkpoint_interval": 32,
        "max_turns": 1,
        "max_actions": 4,
        "opponent_mode": "fixed-v1",
    }
    first = train(config, tmp_path / "first")
    first_checkpoint = tmp_path / "first" / first["checkpoint"]
    second = train(config, tmp_path / "second", resume=first_checkpoint)
    second_initial = tmp_path / "second" / second["initial_checkpoint"]
    second_final = tmp_path / "second" / second["checkpoint"]
    assert first["num_timesteps"] == 32
    assert second["num_timesteps"] == 64
    assert campaign.model_counters(second_initial) == {
        "num_timesteps": 32,
        "_n_updates": 4,
        "metadata_additional_steps": 0,
    }
    assert campaign.model_counters(second_final) == {
        "num_timesteps": 64,
        "_n_updates": 8,
        "metadata_additional_steps": 32,
    }
    with zipfile.ZipFile(second_final / "model.zip") as zf:
        assert "policy.optimizer.pth" in zf.namelist()


def test_packager_collect_paths_does_not_bundle_reserved_final_files(tmp_path):
    for relative in (
        "src/hearthstone_ai/a.py",
        "tests/test_a.py",
        "tests/fixtures/a.json",
        "scripts/a.py",
        "data/cards.json",
        "configs/expansion020_eval.json",
        "AGENTS.md",
        "README.md",
        "pyproject.toml",
        "requirements-wsl.lock",
        "docs/RULES.md",
        "docs/DEATH_SUMMON_CONTRACT.md",
        "experiments/EXPANSION_020_CAMPAIGN.md",
        "experiments/expansion020_campaign.json",
        "handoff/holdout-008/reference-opponents.json",
        "handoff/holdout-008/shifted-opponents.json",
        "handoff/holdout-008/grok-holdout-008.zip",
        "handoff/snapshot-003/HearthStoneAI-Rebuild-003.zip",
        "experiments/reserved/expansion020_final_eval.json",
        "experiments/reserved/expansion020_final_opponents.json",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    collected = {path.relative_to(tmp_path).as_posix() for path in packager.collect_paths(tmp_path)}
    assert "experiments/reserved/expansion020_final_eval.json" not in collected
    assert "experiments/reserved/expansion020_final_opponents.json" not in collected
