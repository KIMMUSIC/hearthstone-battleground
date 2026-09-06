"""One bounded CPU run proves the save/load/CLI/resume contract end to end."""

import json
import subprocess
import sys

from hearthstone_ai.artifacts import ROOT, file_hash
from hearthstone_ai.training import train


def test_checkpoint_fresh_process_evaluation_and_resume(tmp_path):
    config = dict(max_steps=32, max_seconds=60, threads=1, max_turns=2, max_actions=4)
    first_dir = tmp_path / "first"
    first = train(config, first_dir)
    checkpoint = first_dir / first["checkpoint"]
    assert first["num_timesteps"] == 32
    result_file = tmp_path / "evaluation.json"
    subprocess.run(
        [
            sys.executable,
            "-B",
            str(ROOT / "scripts/bgai.py"),
            "evaluate",
            "--policy",
            "model",
            "--checkpoint",
            str(checkpoint),
            "--max-turns",
            "2",
            "--max-actions",
            "4",
            "--seeds",
            "20",
            "21",
            "--output",
            str(result_file),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
    )
    result = json.loads(result_file.read_text())
    assert result["episode_count"] == 2
    assert result["model"]["sha256"] == file_hash(checkpoint / "model.zip")
    resumed = train(config, tmp_path / "resumed", resume=checkpoint)
    assert resumed["initial_steps"] == 32
    assert resumed["additional_steps"] == 32
    assert resumed["num_timesteps"] == 64
    assert (
        resumed["training_metrics"]["train/n_updates"]
        > first["training_metrics"]["train/n_updates"]
    )
    assert (first_dir / "environment.json").exists()
    assert (first_dir / "train.log").stat().st_size > 0
    records = [json.loads(row) for row in (first_dir / "metrics.jsonl").read_text().splitlines()]
    assert records[-1]["num_timesteps"] == 32
