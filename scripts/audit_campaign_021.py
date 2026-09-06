"""Audit all 39 training segments and nine fixed development evaluations."""

import argparse
import hashlib
import io
import json
from pathlib import Path
from statistics import mean
import zipfile

import torch

from campaign_decision import decide
from evaluation_evidence import audit_report


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal_state(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_state(v, right[k]) for k, v in left.items())
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal_state(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def model(checkpoint, expected_steps, expected_updates, signature):
    meta = read(checkpoint / "metadata.json")
    require(meta["compatibility"] == signature, "Checkpoint signature mismatch")
    require(meta["num_timesteps"] == expected_steps, "Metadata step mismatch")
    with zipfile.ZipFile(checkpoint / "model.zip") as z:
        require(z.testzip() is None, "Corrupt model archive")
        data = json.loads(z.read("data"))
        require(data["num_timesteps"] == expected_steps and data["_n_updates"] == expected_updates,
                "Model counters mismatch")
        policy = torch.load(io.BytesIO(z.read("policy.pth")), weights_only=True)
        optimizer = torch.load(io.BytesIO(z.read("policy.optimizer.pth")), weights_only=True)
    require(all(torch.isfinite(v).all() for v in policy.values()), "Nonfinite model")
    return policy, optimizer


def audit(run, baseline):
    files = read(run / "FILES.json")
    for name, digest in files.items():
        require(sha(run / name) == digest, f"Frozen input changed: {name}")
    require(not any("expansion020_final" in n for n in files), "Final holdout bundled")
    spec = read(run / "experiments/expansion020_campaign.json")
    signature = spec["compatibility"]
    require(baseline["compatibility"] == signature, "Baseline contract differs")
    out = run / "outputs"
    driver = read(out / "driver-status.json")
    require(driver["status"] == "completed" and driver["completed_phases"] == 48,
            "Campaign driver incomplete")
    require(driver["elapsed_seconds"] <= spec["limits"]["campaign_total_wall_seconds"],
            "Whole campaign wall budget exceeded")

    def local(path):
        text = str(path).replace("\\", "/")
        require("/outputs/" in text, "Checkpoint outside outputs")
        result = run / "outputs" / text.split("/outputs/", 1)[1]
        require(result.resolve().is_relative_to(out.resolve()), "Checkpoint path escapes run")
        return result

    summaries = {}
    trained = 0
    initial_policies = []
    for seed in spec["training_seeds"]:
        total = 0
        previous = None
        for index, steps in enumerate(spec["segment_additional_steps"], 1):
            summary = read(out / f"seed-{seed:03d}/segment-{index:02d}/summary.json")
            status = summary["status"]
            require(summary["compatibility"] == signature, "Segment signature mismatch")
            require(status["status"] == "completed" and status["stop_reason"] == "steps", "Incomplete training")
            require(status["initial_steps"] == total and status["additional_steps"] == steps, "Segment step mismatch")
            initial = local(summary["initial_checkpoint"])
            final = local(summary["final_checkpoint"])
            require(sha(initial / "model.zip") == summary["initial_model_sha256"], "Initial model changed")
            require(sha(final / "model.zip") == summary["final_model_sha256"], "Final model changed")
            initial_state = model(initial, total, total // 32 * 4, signature)
            if previous is not None:
                require(equal_state(initial_state, previous), "Resume lost policy or optimizer state")
            else:
                initial_policies.append(initial_state[0])
            total += steps
            previous = model(final, total, total // 32 * 4, signature)
            require(status["num_timesteps"] == total, "Final status steps mismatch")
            trained += steps
        require(total == 100000, "Seed incomplete")
        summaries[seed] = {}
        for milestone in spec["evaluation_milestones"]:
            report = read(out / f"seed-{seed:03d}/eval-{milestone:06d}/model.json")
            require(report["training_seed"] == seed and report["milestone"] == milestone, "Evaluation identity mismatch")
            require(report["compatibility"] == signature, "Evaluation signature mismatch")
            require(report["opponent_set_sha256"] == baseline["opponent_set_sha256"], "Evaluation opponent differs")
            audit_report(report, list(range(20001, 20101)))
            require(sha(local(report["model"]["path"])) == report["model"]["sha256"], "Evaluated model changed")
            summaries[seed][milestone] = {k: report[k] for k in ("mean_reward", "combat_win_rate", "survival_rate")}
            summaries[seed][milestone]["mean_metrics"] = {
                k: mean(e[k] for e in report["episodes"]) for k in
                ("wins", "draws", "losses", "damage_taken", "damage_dealt", "actions",
                 "forced_end_turns", "rerolls", "freezes", "swaps")}
    require(all(not equal_state(left, right) for i, left in enumerate(initial_policies)
                for right in initial_policies[i + 1:]), "Seeds initialized identically")
    controls = list(out.glob("control-*/result.json"))
    require(len(controls) == 48, "Expected 39 train and nine eval controls")
    totals = {"train": 0.0, "evaluate": 0.0}
    peak_rss = 0
    final_progress = read(out / "progress.json")["events"]
    require(len(final_progress) == 48, "Progress count differs")
    for path in controls:
        control = read(path)
        require(control["status"] == "completed" and control["returncode"] == 0, "Failed phase control")
        totals[control["mode"]] += control.get("end_to_end_elapsed_seconds", control["elapsed_seconds"])
        supervision = read(path.parent / "supervisor/result.json")
        require(supervision["status"] == "completed", "Supervisor failed")
        for command in supervision["commands"]:
            require(command["remaining_pids"] == [] and command["returncode"] == 0, "Leftover process")
            require(command["ended_unix"] - command["started_unix"] < 180, "Supervisor time violation")
            peak_rss = max(peak_rss, command["peak_sampled_rss"])
        archive = run / "returns" / ("return-" + control["phase_id"] + ".zip")
        require(sha(archive) == control["return_package"]["archive_sha256"], "Return archive changed")
        with zipfile.ZipFile(archive) as z:
            require(z.testzip() is None, "Corrupt return archive")
            require(json.loads(z.read("FILES.json")) == files, "Return input manifest differs")
            for name in z.namelist():
                if name == "outputs/progress.json":
                    progress = json.loads(z.read(name))["events"]
                    require(progress == final_progress[:len(progress)], "Progress snapshot differs")
                elif name.endswith("/result.json") and "/control-" in name and "/supervisor/" not in name:
                    archived = json.loads(z.read(name))
                    for key, value in archived.items():
                        if key not in {"elapsed_seconds", "ended_unix"}:
                            require(value == control[key], f"Control evidence differs: {key}")
                else:
                    require(z.read(name) == (run / name).read_bytes(), f"Archived file changed: {name}")
    limits = spec["limits"]
    require(totals["train"] <= limits["campaign_train_wall_seconds"], "Cumulative train budget")
    require(totals["evaluate"] <= limits["campaign_dev_eval_wall_seconds"], "Cumulative eval budget")
    require(sum(totals.values()) <= limits["campaign_total_wall_seconds"], "Cumulative total budget")
    require(peak_rss < limits["sampled_rss_bytes"], "RSS violation")
    disk = sum(p.stat().st_size for parent in (out, run / "returns") for p in parent.rglob("*") if p.is_file())
    require(disk < limits["all_outputs_and_return_zip_bytes"], "Disk violation")
    decision = decide(summaries, spec["training_seeds"], spec["acceptance"], baseline["mean_reward"])
    return {"status": "passed", "errors": [], "training_steps": trained, "evaluation_games": 900,
            "segments": 39, "controls": 48, "peak_sampled_rss_bytes": peak_rss,
            "output_and_return_bytes": disk, "elapsed_seconds": totals, "compatibility": signature,
            "resume_policy_and_optimizer_exact": True, "reports": summaries, "decision": decision}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run, read(args.baseline))
    (args.run / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"reports", "compatibility"}}))


if __name__ == "__main__":
    main()
