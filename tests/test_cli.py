import json
import sys

from hearthstone_ai.cli import main


def test_evaluation_cli_passes_custom_horizon_and_action_budget(monkeypatch, tmp_path, capsys):
    target = tmp_path / "evaluation.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bgai",
            "evaluate",
            "--max-turns",
            "2",
            "--max-actions",
            "3",
            "--seeds",
            "123",
            "--output",
            str(target),
        ],
    )
    main()
    result = json.loads(target.read_text())
    assert result["max_turns"] == 2
    assert result["max_actions"] == 3
    assert result["episodes"][0]["last_turn"] == 2
    assert result["model"] is None
    capsys.readouterr()


def test_evaluation_cli_writes_trace_output(monkeypatch, tmp_path, capsys):
    target = tmp_path / "evaluation.json"
    trace = tmp_path / "trace.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bgai",
            "evaluate",
            "--seeds",
            "123",
            "--output",
            str(target),
            "--trace-output",
            str(trace),
        ],
    )
    main()
    assert json.loads(target.read_text())["episode_count"] == 1
    assert json.loads(trace.read_text())["episodes"][0]["seed"] == 123
    capsys.readouterr()


def test_evaluation_cli_accepts_opponent_path(monkeypatch, tmp_path, capsys):
    target = tmp_path / "evaluation.json"
    opponents = tmp_path / "opponents.json"
    opponents.write_text(
        json.dumps(
            {
                "version": "test-opponents",
                "purpose": "test",
                "opponents": [{"tier": 1, "board": [60628]}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bgai",
            "evaluate",
            "--seeds",
            "123",
            "--opponent-path",
            str(opponents),
            "--output",
            str(target),
        ],
    )
    main()
    result = json.loads(target.read_text())
    assert result["episode_count"] == 1
    assert result["opponent_set_sha256"]
    capsys.readouterr()


def test_evaluation_cli_embeds_trace(monkeypatch, tmp_path, capsys):
    target = tmp_path / "evaluation.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bgai",
            "evaluate",
            "--seeds",
            "123",
            "--output",
            str(target),
            "--trace",
        ],
    )
    main()
    result = json.loads(target.read_text())
    assert result["trace"]["episodes"][0]["seed"] == 123
    capsys.readouterr()
