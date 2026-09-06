import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hearthstone_ai import training


def test_interrupted_run_records_status_and_releases_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "compatibility_signature", lambda: {"source": "test"})

    def interrupt_environment(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(training, "BgEnv", interrupt_environment)
    with pytest.raises(KeyboardInterrupt):
        training.train({}, tmp_path)
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "interrupted"
    assert status["stop_reason"] == "keyboard_interrupt"
    assert not (tmp_path / ".training.lock").exists()
    assert (tmp_path / "environment.json").exists()
    events = [json.loads(line) for line in (tmp_path / "train.log").read_text().splitlines()]
    assert [event["event"] for event in events] == ["start", "stop"]
    assert events[0]["pid"] > 0
    assert events[-1]["status"] == "interrupted"


@pytest.mark.parametrize(
    "config",
    [
        dict(max_steps=0),
        dict(max_steps=33),
        dict(max_seconds=float("nan")),
        dict(threads=True),
        dict(batch_size=3),
        dict(unbounded=True),
        dict(gamma=True),
        dict(gamma=0),
        dict(gamma=1.01),
        dict(gamma=float("inf")),
    ],
)
def test_invalid_budget_does_not_create_output(tmp_path, config):
    output = tmp_path / "run"
    with pytest.raises(ValueError):
        training.train(config, output)
    assert not output.exists()


def test_checkpoint_mismatch_rejected_before_model_load(tmp_path, monkeypatch):
    (tmp_path / "metadata.json").write_text(json.dumps({"compatibility": {"wrong": "version"}}))
    monkeypatch.setattr(training, "compatibility_signature", lambda: {"correct": "version"})
    monkeypatch.setattr(training.MaskablePPO, "load", lambda *a, **k: pytest.fail("unsafe load"))
    with pytest.raises(ValueError, match="compatibility"):
        training.load_model(tmp_path, None)


def test_existing_output_is_preserved(tmp_path):
    sentinel = tmp_path / "status.json"
    sentinel.write_text("existing data")
    with pytest.raises(FileExistsError, match="empty"):
        training.train({}, tmp_path)
    assert sentinel.read_text() == "existing data"
    assert not (tmp_path / ".training.lock").exists()


def test_existing_lock_is_preserved(tmp_path):
    lock = tmp_path / ".training.lock"
    lock.write_text("other process")
    with pytest.raises(FileExistsError):
        training.train({}, tmp_path)
    assert lock.read_text() == "other process"


def test_environment_mismatch_rejected_before_model_load(tmp_path, monkeypatch):
    signature = {"source": "same"}
    monkeypatch.setattr(training, "compatibility_signature", lambda: signature)
    (tmp_path / "metadata.json").write_text(
        json.dumps({"compatibility": signature, "config": {"max_turns": 8, "max_actions": 24}})
    )
    monkeypatch.setattr(training.MaskablePPO, "load", lambda *a, **k: pytest.fail("unsafe load"))
    with pytest.raises(ValueError, match="environment mismatch"):
        training.load_model(tmp_path, SimpleNamespace(max_turns=9, max_actions=24))


def test_training_opponent_generation_does_not_read_evaluation_inputs(tmp_path, monkeypatch):
    original_read_text = Path.read_text

    def guard_eval_inputs(path, *args, **kwargs):
        if Path(path).name == "eval_opponents.json":
            raise AssertionError("training opponent generation read evaluation inputs")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guard_eval_inputs)
    training.train(
        dict(max_steps=32, checkpoint_interval=32, max_seconds=60, threads=1, opponent_mode="diverse-v1"),
        tmp_path,
    )


def test_checkpoint_opponent_mode_mismatch_is_allowed_for_evaluation_load(tmp_path, monkeypatch):
    signature = {"source": "same"}
    monkeypatch.setattr(training, "compatibility_signature", lambda: signature)
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "compatibility": signature,
                "config": {"max_turns": 8, "max_actions": 24, "opponent_mode": "diverse-v1"},
            }
        )
    )
    sentinel = object()
    monkeypatch.setattr(training.MaskablePPO, "load", lambda *a, **k: sentinel)
    env = SimpleNamespace(max_turns=8, max_actions=24, opponent_mode="fixed-v1")
    assert training.load_model(tmp_path, env) is sentinel


def test_failed_run_records_error_and_releases_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "compatibility_signature", lambda: {"source": "test"})

    def fail_environment(**kwargs):
        raise RuntimeError("environment unavailable")

    monkeypatch.setattr(training, "BgEnv", fail_environment)
    with pytest.raises(RuntimeError, match="environment unavailable"):
        training.train({}, tmp_path)
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "failed"
    assert "environment unavailable" in (tmp_path / "failure.log").read_text()
    assert not (tmp_path / ".training.lock").exists()


def test_fresh_training_saves_initialized_checkpoint_before_learning(tmp_path):
    result = training.train(
        dict(max_steps=32, checkpoint_interval=32, max_seconds=60, threads=1),
        tmp_path,
    )
    initialized = tmp_path / "checkpoint-000000000000"
    assert initialized.exists()
    assert result["initial_checkpoint"] == initialized.name
    assert initialized.name in result["checkpoints"]
    metadata = json.loads((initialized / "metadata.json").read_text())
    assert metadata["num_timesteps"] == 0
    assert metadata["additional_steps"] == 0


def test_opponent_mode_config_is_validated_and_recorded(tmp_path):
    with pytest.raises(ValueError, match="opponent_mode"):
        training.train(dict(opponent_mode="bad-mode"), tmp_path / "bad")
    result = training.train(
        dict(
            max_steps=32,
            checkpoint_interval=32,
            max_seconds=60,
            threads=1,
            opponent_mode="diverse-v1",
        ),
        tmp_path / "diverse",
    )
    manifest = json.loads((tmp_path / "diverse" / "manifest.json").read_text())
    metadata = json.loads(
        (tmp_path / "diverse" / result["initial_checkpoint"] / "metadata.json").read_text()
    )
    assert manifest["config"]["opponent_mode"] == "diverse-v1"
    assert manifest["opponent_distribution"]["candidate_card_ids"] == [42467, 72387]
    assert metadata["config"]["opponent_mode"] == "diverse-v1"
    assert metadata["opponent_distribution"]["version"] == "diverse-v1"


def test_training_gamma_is_validated_recorded_and_passed_to_model(tmp_path):
    result = training.train(
        dict(max_steps=32, checkpoint_interval=32, max_seconds=60, threads=1, gamma=1.0),
        tmp_path / "gamma",
    )

    manifest = json.loads((tmp_path / "gamma" / "manifest.json").read_text())
    metadata = json.loads(
        (tmp_path / "gamma" / result["initial_checkpoint"] / "metadata.json").read_text()
    )
    env = training.BgEnv()
    model = training.load_model(tmp_path / "gamma" / result["checkpoint"], env)
    try:
        assert manifest["config"]["gamma"] == 1.0
        assert metadata["config"]["gamma"] == 1.0
        assert model.gamma == 1.0
    finally:
        model.env.close()


def test_resume_requires_matching_training_opponent_mode(tmp_path):
    first = training.train(
        dict(max_steps=32, checkpoint_interval=32, max_seconds=60, threads=1, opponent_mode="fixed-v1"),
        tmp_path / "fixed",
    )
    with pytest.raises(ValueError, match="opponent_mode"):
        training.train(
            dict(max_steps=32, max_seconds=60, threads=1, opponent_mode="diverse-v1"),
            tmp_path / "resume",
            resume=tmp_path / "fixed" / first["checkpoint"],
        )


def test_resume_requires_matching_training_gamma_before_model_load(tmp_path, monkeypatch):
    first = training.train(
        dict(max_steps=32, checkpoint_interval=32, max_seconds=60, threads=1, gamma=1.0),
        tmp_path / "gamma-one",
    )
    monkeypatch.setattr(training.MaskablePPO, "load", lambda *a, **k: pytest.fail("unsafe load"))

    with pytest.raises(ValueError, match="gamma"):
        training.train(
            dict(max_steps=32, max_seconds=60, threads=1, gamma=0.99),
            tmp_path / "resume",
            resume=tmp_path / "gamma-one" / first["checkpoint"],
        )
    assert not any((tmp_path / "resume").glob("checkpoint-*"))
