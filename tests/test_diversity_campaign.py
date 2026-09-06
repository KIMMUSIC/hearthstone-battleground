import importlib.util
import io
import json
from pathlib import Path
import sys
import zipfile

import pytest
import torch


scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("diversity_009", scripts / "diversity_009.py")
diversity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diversity)
audit_spec = importlib.util.spec_from_file_location(
    "audit_diversity_009", scripts / "audit_diversity_009.py"
)
audit = importlib.util.module_from_spec(audit_spec)
audit_spec.loader.exec_module(audit)


def write_model_zip(path, *, timesteps, updates, tensor_value, n_epochs=4):
    buffer = io.BytesIO()
    torch.save({"linear.weight": torch.tensor([float(tensor_value)])}, buffer)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("policy.pth", buffer.getvalue())
        zf.writestr(
            "data",
            json.dumps(
                {
                    "num_timesteps": timesteps,
                    "_n_updates": updates,
                    "n_steps": 32,
                    "batch_size": 32,
                    "n_epochs": n_epochs,
                }
            ),
        )


def minimal_probe_report(*, reward, survived=True):
    probabilities = [0.0] * 37
    probabilities[0] = 1.0
    step = {
        "action": 0,
        "probabilities": probabilities,
        "entropy_nats": 0.0,
        "top_probability": 1.0,
        "top_margin": 1.0,
    }
    return {
        "campaign_id": diversity.CAMPAIGN_ID,
        "seeds": list(diversity.EVALUATION_SEEDS),
        "episodes": [
            {
                "reward": reward,
                "survived": survived,
                "wins": 1,
                "draws": 0,
                "losses": 0,
                "forced_end_turns": 0,
            }
        ],
        "trace": [{"steps": [step]}],
        "mean_reward": reward,
        "combat_win_rate": 1.0,
        "survival_rate": float(survived),
        "policy_seed": 11,
    }


def test_plan_phase_writes_exact_phase_commands(tmp_path, monkeypatch):
    monkeypatch.setattr(diversity, "file_hash", lambda path: f"hash:{Path(path).name}")
    root = tmp_path / "checkout"
    output = tmp_path / "out"
    signature = {"source_sha256": "source-123"}
    plan = diversity.phase_plan(root, output, signature, "/venv/bin/python")
    assert set(plan["commands"]) == {"smoke", "train", "reference", "shifted"}
    assert plan["commands"]["train"] == [
        "/venv/bin/python",
        "scripts/diversity_009.py",
        "--checkout",
        str(root.resolve()),
        "--output",
        str(output),
        "--expected-source-sha256",
        "source-123",
        "--phase",
        "train",
    ]
    assert (output / "plan" / "plan.json").exists()


def test_training_config_separates_string_opponent_mode(tmp_path):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "research_train.json").write_text(json.dumps({"seed": 99}), encoding="utf-8")
    config = diversity.training_config(tmp_path, seed=17, mode="diverse-v1", smoke=False)
    assert config["opponent_mode"] == "diverse-v1"
    assert config["seed"] == 17
    assert config["max_steps"] == 8192
    assert config["n_epochs"] == 4


def test_audit_rejects_zip_path_traversal(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe ZIP"):
        audit.safe_extract(archive, tmp_path / "extract")


def test_primary_decision_uses_stochastic_final_reports_only():
    eval_summaries = {}
    for cohort in diversity.COHORTS:
        reported = {}
        for seed in diversity.TRAINING_SEEDS:
            for mode, reward in (("fixed-v1", 1.0), ("diverse-v1", 1.4)):
                for policy_seed in diversity.POLICY_SEEDS:
                    reported[f"{cohort}-final-{mode}-seed-{seed}-sample-{policy_seed}"] = {
                        "mean_reward": reward,
                        "survival_rate": 1.0,
                    }
        eval_summaries[cohort] = {"reported": reported}
    decision = audit.primary_decision(eval_summaries)
    assert decision["mean_delta"] == pytest.approx(0.4)
    assert decision["positive_seed_count"] == 3
    assert decision["candidate_keep"] is True


def test_audit_training_remaps_cloud_checkpoint_paths(tmp_path):
    train_dir = tmp_path / "train"
    jobs = []
    for seed in diversity.TRAINING_SEEDS:
        for mode in diversity.MODES:
            label = f"{mode}-seed-{seed}-steps-8192"
            run_dir = train_dir / label
            initial = run_dir / "checkpoint-000000000000"
            final = run_dir / diversity.EXPECTED_FINAL_CHECKPOINT
            initial.mkdir(parents=True)
            final.mkdir()
            write_model_zip(initial / "model.zip", timesteps=0, updates=0, tensor_value=seed)
            write_model_zip(
                final / "model.zip",
                timesteps=diversity.FULL_STEPS,
                updates=diversity.EXPECTED_FINAL_UPDATES,
                tensor_value=seed + 100,
            )
            for checkpoint, timesteps in ((initial, 0), (final, diversity.FULL_STEPS)):
                (checkpoint / "metadata.json").write_text(
                    json.dumps(
                        {
                            "num_timesteps": timesteps,
                            "config": {
                                "seed": seed,
                                "opponent_mode": mode,
                                "max_steps": diversity.FULL_STEPS,
                                "max_turns": 8,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            (run_dir / "status.json").write_text(
                json.dumps(
                    {
                        "checkpoint": diversity.EXPECTED_FINAL_CHECKPOINT,
                        "num_timesteps": diversity.FULL_STEPS,
                        "additional_steps": diversity.FULL_STEPS,
                    }
                ),
                encoding="utf-8",
            )
            jobs.append(
                {
                    "label": label,
                    "seed": seed,
                    "opponent_mode": mode,
                    "status": {
                        "status": "completed",
                        "stop_reason": "steps",
                        "num_timesteps": 8192,
                        "checkpoint": diversity.EXPECTED_FINAL_CHECKPOINT,
                    },
                    "optimizer_updates": diversity.EXPECTED_FINAL_UPDATES,
                    "initial_checkpoint": f"/workspace/out/train/{label}/checkpoint-000000000000",
                    "final_checkpoint": (
                        f"/workspace/out/train/{label}/{diversity.EXPECTED_FINAL_CHECKPOINT}"
                    ),
                    "initial_model_sha256": diversity.file_hash(initial / "model.zip"),
                    "final_model_sha256": diversity.file_hash(final / "model.zip"),
                    "initial_policy_payload_sha256": diversity.zip_member_hash(
                        initial / "model.zip", "policy.pth"
                    ),
                    "final_policy_payload_sha256": diversity.zip_member_hash(
                        final / "model.zip", "policy.pth"
                    ),
                    "initial_parameter_digest": audit.policy_tensor_digest(
                        initial / "model.zip"
                    ),
                    "final_parameter_digest": audit.policy_tensor_digest(final / "model.zip"),
                }
            )
    (train_dir / "summary.json").write_text(
        json.dumps(
            {
                "jobs": jobs,
                "initial_parameter_pair_matches": {
                    str(seed): True for seed in diversity.TRAINING_SEEDS
                },
            }
        ),
        encoding="utf-8",
    )
    _, errors = audit.audit_training(tmp_path)
    assert errors == []


def read_008_report(kind):
    archive = Path(__file__).resolve().parents[1] / "handoff/holdout-008/grok-holdout-008.zip"
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if not name.startswith("results/") or not name.endswith(".json"):
                continue
            report = json.loads(zf.read(name))
            if kind == "baseline" and report.get("policy") == "heuristic":
                return report
            if kind == "model" and report.get("selection"):
                return report
    raise AssertionError(f"missing {kind} report")


def test_audit_payload_accepts_legacy_model_and_baseline_trace_shapes(monkeypatch):
    monkeypatch.setattr(audit, "EVALUATION_SEEDS", tuple(range(301, 321)))
    for kind in ("model", "baseline"):
        errors = []
        audit.audit_report_payload(kind, read_008_report(kind), errors)
        assert errors == []


def test_audit_payload_rejects_model_probability_mask_entropy_tampering(monkeypatch):
    monkeypatch.setattr(audit, "EVALUATION_SEEDS", tuple(range(301, 321)))
    report = read_008_report("model")

    missing_probabilities = json.loads(json.dumps(report))
    del missing_probabilities["trace"][0]["steps"][0]["probabilities"]
    errors = []
    audit.audit_report_payload("missing-probs", missing_probabilities, errors)
    assert any("missing probabilities" in error for error in errors)

    illegal_mask = json.loads(json.dumps(report))
    first_step = illegal_mask["trace"][0]["steps"][0]
    first_step["action_mask"][first_step["action"]] = False
    errors = []
    audit.audit_report_payload("illegal-mask", illegal_mask, errors)
    assert any("chosen illegal action" in error for error in errors)

    bad_entropy = json.loads(json.dumps(report))
    bad_entropy["trace"][0]["steps"][0]["entropy_nats"] += 1.0
    errors = []
    audit.audit_report_payload("bad-entropy", bad_entropy, errors)
    assert any("entropy_nats mismatch" in error for error in errors)


def test_audit_payload_rejects_baseline_mask_tampering(monkeypatch):
    monkeypatch.setattr(audit, "EVALUATION_SEEDS", tuple(range(301, 321)))
    report = read_008_report("baseline")
    first_step = report["trace"]["episodes"][0]["steps"][0]
    first_step["action_mask"][first_step["action"]] = False
    errors = []
    audit.audit_report_payload("baseline-mask", report, errors)
    assert any("chosen illegal action" in error for error in errors)


def test_report_summary_recalculates_selection_summary():
    summary = audit.report_summary(minimal_probe_report(reward=2.0))
    assert summary["mean_reward"] == 2.0
    assert summary["freeze_fraction"] == 0.0
    assert summary["end_fraction"] == 1.0
