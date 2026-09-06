import importlib.util
import json
import os
from pathlib import Path
import sys
import zipfile

import pytest


scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))

expansion_spec = importlib.util.spec_from_file_location("expansion_019", scripts / "expansion_019.py")
expansion = importlib.util.module_from_spec(expansion_spec)
expansion_spec.loader.exec_module(expansion)

runner_spec = importlib.util.spec_from_file_location(
    "run_expansion_019", scripts / "run_expansion_019.py"
)
runner = importlib.util.module_from_spec(runner_spec)
runner_spec.loader.exec_module(runner)


def write_model_checkpoint(path: Path, payload: bytes = b"model") -> None:
    path.mkdir(parents=True)
    (path / "model.zip").write_bytes(payload)
    (path / "metadata.json").write_text("{}", encoding="utf-8")


def write_configs(root: Path, *, seeds=None, max_steps=8192) -> None:
    configs = root / "configs"
    configs.mkdir()
    (configs / "expansion019_train.json").write_text(
        json.dumps(
            {
                "max_steps": max_steps,
                "max_seconds": 60,
                "threads": 1,
                "seed": 7,
                "n_steps": 32,
                "batch_size": 32,
                "n_epochs": 4,
                "checkpoint_interval": 8192,
                "max_turns": 8,
                "max_actions": 24,
            }
        ),
        encoding="utf-8",
    )
    (configs / "expansion019_eval.json").write_text(
        json.dumps(
            {
                "seeds": list(range(701, 721)) if seeds is None else seeds,
                "opponent_path": "configs/eval_opponents.json",
                "max_turns": 8,
                "max_actions": 24,
            }
        ),
        encoding="utf-8",
    )
    (configs / "eval_opponents.json").write_text(json.dumps({"opponents": [{"tier": 1, "board": []}]}))


def minimal_report(policy: str, seeds: list[int]) -> dict:
    return {
        "policy": policy,
        "seeds": seeds,
        "episodes": [{"reward": 1.0, "survived": True, "wins": 1, "draws": 0, "losses": 0}]
        * len(seeds),
        "episode_count": len(seeds),
        "combat_count": len(seeds),
        "combat_win_rate": 1.0,
        "survival_rate": 1.0,
        "mean_reward": 1.0,
        "trace": {"episodes": []},
    }


def test_train_phase_records_achieved_steps_and_checkpoint_hashes(tmp_path, monkeypatch):
    write_configs(tmp_path)

    def fake_train(config, output_dir):
        write_model_checkpoint(output_dir / "checkpoint-000000000000", b"initial")
        write_model_checkpoint(output_dir / "checkpoint-000000008192", b"final")
        return {
            "status": "completed",
            "stop_reason": "steps",
            "initial_checkpoint": "checkpoint-000000000000",
            "checkpoint": "checkpoint-000000008192",
            "additional_steps": 8192,
            "num_timesteps": 8192,
        }

    monkeypatch.setattr(expansion, "configure_torch_threads", lambda: None)
    monkeypatch.setattr(expansion, "train_model", fake_train)
    monkeypatch.setattr(expansion, "assert_checkpoint_loads", lambda *args, **kwargs: None)
    summary = expansion.phase_train(tmp_path, tmp_path / "outputs", {"source_sha256": "abc"})
    assert summary["achieved_steps"] == 8192
    assert summary["initial_model_sha256"] != summary["final_model_sha256"]
    assert (tmp_path / "outputs/train/summary.json").exists()


def test_train_phase_rejects_budget_above_8192(tmp_path):
    write_configs(tmp_path, max_steps=8193)
    with pytest.raises(ValueError, match="max_steps=8192"):
        expansion.load_training_config(tmp_path)


def test_train_phase_rejects_time_budget_above_60_seconds(tmp_path):
    write_configs(tmp_path)
    path = tmp_path / "configs" / "expansion019_train.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["max_seconds"] = 61
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="max_seconds <= 60"):
        expansion.load_training_config(tmp_path)


def test_evaluate_phase_rejects_checkpoint_mutation(tmp_path, monkeypatch):
    write_configs(tmp_path)
    train_dir = tmp_path / "outputs" / "train"
    initial = train_dir / "run" / "checkpoint-000000000000"
    final = train_dir / "run" / "checkpoint-000000008192"
    write_model_checkpoint(initial, b"initial")
    write_model_checkpoint(final, b"final")
    train_summary = {
        "initial_checkpoint": str(initial),
        "final_checkpoint": str(final),
        "initial_model_sha256": expansion.file_hash(initial / "model.zip"),
        "final_model_sha256": expansion.file_hash(final / "model.zip"),
    }
    expansion.write_json(train_dir / "summary.json", train_summary)

    def mutating_evaluate(**kwargs):
        checkpoint = kwargs["checkpoint"]
        if checkpoint is not None:
            (checkpoint / "model.zip").write_bytes(b"changed")
        return minimal_report(kwargs["policy"], kwargs["seeds"])

    monkeypatch.setattr(expansion, "configure_torch_threads", lambda: None)
    monkeypatch.setattr(expansion, "evaluate_policy", mutating_evaluate)
    with pytest.raises(RuntimeError, match="mutated checkpoint"):
        expansion.phase_evaluate(tmp_path, tmp_path / "outputs", {"source_sha256": "abc"})


def test_eval_config_requires_exactly_20_unique_seeds(tmp_path):
    write_configs(tmp_path, seeds=[1, 1])
    with pytest.raises(ValueError, match="20 unique"):
        expansion.load_eval_config(tmp_path)


@pytest.mark.parametrize("existing", ["return-evaluate.zip", "outputs/control-evaluate", "outputs/evaluate"])
def test_runner_blocks_prior_phase_evidence(tmp_path, existing):
    prior = tmp_path / existing
    prior.parent.mkdir(parents=True, exist_ok=True)
    prior.write_bytes(b"old")
    with pytest.raises(RuntimeError, match="already attempted"):
        runner.require_fresh_phase(tmp_path, "evaluate")
    assert prior.read_bytes() == b"old"


def test_runner_zip_contains_files_manifest_and_outputs(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("input", encoding="utf-8")
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "result.json").write_text("{}", encoding="utf-8")
    digest = runner.hashlib.sha256(source.read_bytes()).hexdigest()
    (tmp_path / "FILES.json").write_text(json.dumps({"source.txt": digest}), encoding="utf-8")
    package = runner.bounded_zip(tmp_path, out, tmp_path / "return-train.zip", 1024 * 1024)
    assert package["archive_sha256"] == runner.hashlib.sha256(
        (tmp_path / "return-train.zip").read_bytes()
    ).hexdigest()
    with zipfile.ZipFile(tmp_path / "return-train.zip") as zf:
        assert sorted(zf.namelist()) == ["FILES.json", "outputs/result.json", "source.txt"]
    assert package["all_return_zip_bytes"] == package["archive_bytes"]


def test_runner_counts_existing_return_zips_in_disk_budget(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("input", encoding="utf-8")
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "result.json").write_text("{}" * 100, encoding="utf-8")
    (tmp_path / "return-train.zip").write_bytes(b"x" * 100)
    digest = runner.hashlib.sha256(source.read_bytes()).hexdigest()
    (tmp_path / "FILES.json").write_text(json.dumps({"source.txt": digest}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="existing ZIPs"):
        runner.bounded_zip(tmp_path, out, tmp_path / "return-evaluate.zip", 120)


def test_runner_rejects_frozen_input_mutation(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("original", encoding="utf-8")
    digest = runner.hashlib.sha256(source.read_bytes()).hexdigest()
    (tmp_path / "FILES.json").write_text(json.dumps({"source.txt": digest}), encoding="utf-8")
    runner.verify_files(tmp_path)
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Input hash mismatch"):
        runner.verify_files(tmp_path)


def test_runner_propagates_supervisor_failure_status(tmp_path, monkeypatch):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    supervise = scripts_dir / "supervise.py"
    supervise.write_text(
        "\n".join(
            [
                "import argparse, json, pathlib, sys",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--spec')",
                "parser.add_argument('--output')",
                "args = parser.parse_args()",
                "out = pathlib.Path(args.output)",
                "out.mkdir(parents=True)",
                "payload = {'status': 'command_failed', 'commands': [{'returncode': 7}]}",
                "(out / 'result.json').write_text(json.dumps(payload), encoding='utf-8')",
                "raise SystemExit(1)",
            ]
        ),
        encoding="utf-8",
    )
    execution = {
        "python": sys.executable,
        "minimum_free_bytes": 1,
        "output_disk_limit_bytes": 1024 * 1024,
        "phases": {"train": {"requires": [], "commands": [[sys.executable, "-c", "pass"]]}},
    }
    (tmp_path / "execution.json").write_text(json.dumps(execution), encoding="utf-8")
    source = tmp_path / "source.txt"
    source.write_text("source", encoding="utf-8")
    entries = {
        path.relative_to(tmp_path).as_posix(): runner.hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (source, tmp_path / "execution.json", supervise)
    }
    (tmp_path / "FILES.json").write_text(json.dumps(entries), encoding="utf-8")
    monkeypatch.setattr(runner, "require_runtime", lambda plan: Path(sys.executable))
    monkeypatch.setattr(os, "environ", os.environ.copy())
    result = runner.run_phase(tmp_path, "train")
    assert result["status"] == "command_failed"
    assert result["returncode"] == 1
    assert Path(result["return_package"]["archive"]).exists()
