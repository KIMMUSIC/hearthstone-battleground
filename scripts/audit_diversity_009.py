"""Independent audit for opponent-diversity-009 returned result directories or ZIPs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import zipfile

from diversity_009 import (
    CAMPAIGN_ID,
    COHORTS,
    EVALUATION_SEEDS,
    EXPECTED_FINAL_CHECKPOINT,
    EXPECTED_FINAL_UPDATES,
    FULL_STEPS,
    MODES,
    POLICY_SEEDS,
    TRAINING_SEEDS,
    file_hash,
    json_hash,
    load_opponent_payload,
    load_project,
    read_json,
    write_json,
    zip_member_hash,
)
from selection_probe import selection_summary


ACTION_COUNT = 37


def safe_extract(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (root / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"Unsafe ZIP member path: {member.filename}")
        zf.extractall(root)
    children = [path for path in root.iterdir() if path.is_dir()]
    return children[0] if len(children) == 1 and (children[0] / "train").exists() else root


def close_enough(left: float, right: float, *, tolerance: float = 1e-12) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def checkpoint_from_summary(results: Path, label: str, checkpoint_value: object) -> Path:
    run_dir = (results / "train" / label).resolve()
    checkpoint = (run_dir / Path(str(checkpoint_value)).name).resolve()
    if run_dir != checkpoint and run_dir not in checkpoint.parents:
        raise ValueError(f"checkpoint escapes run directory: {label}")
    return checkpoint


def stable_baselines_data(model_zip: Path) -> dict[str, object]:
    with zipfile.ZipFile(model_zip) as zf:
        return json.loads(zf.read("data"))


def policy_tensor_digest(model_zip: Path) -> str:
    import io
    import hashlib
    import torch

    with zipfile.ZipFile(model_zip) as zf:
        state = torch.load(io.BytesIO(zf.read("policy.pth")), map_location="cpu", weights_only=True)
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not torch.is_tensor(tensor):
            continue
        array = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.numpy().tobytes())
    return digest.hexdigest()


def audit_training(results: Path) -> tuple[dict[str, object], list[str]]:
    errors = []
    summary = read_json(results / "train" / "summary.json")
    jobs = summary.get("jobs", [])
    expected_order = [(seed, mode) for seed in TRAINING_SEEDS for mode in MODES]
    actual_order = [(job.get("seed"), job.get("opponent_mode")) for job in jobs]
    if actual_order != expected_order:
        errors.append(f"training order mismatch: {actual_order}")
    if summary.get("initial_parameter_pair_matches") != {str(seed): True for seed in TRAINING_SEEDS}:
        errors.append("initial parameter pair digest check missing or failed")
    policy_hashes: dict[int, dict[str, str]] = {seed: {} for seed in TRAINING_SEEDS}
    for job in jobs:
        label = job.get("label", "")
        status = job.get("status", {})
        if status.get("status") != "completed" or status.get("stop_reason") != "steps":
            errors.append(f"training did not complete by steps: {label}")
        if status.get("num_timesteps") != FULL_STEPS:
            errors.append(f"wrong timestep count: {label}")
        if status.get("checkpoint") != EXPECTED_FINAL_CHECKPOINT:
            errors.append(f"wrong final checkpoint name: {label}")
        if job.get("optimizer_updates") != EXPECTED_FINAL_UPDATES:
            errors.append(f"wrong optimizer update count: {label}")
        try:
            final_checkpoint = checkpoint_from_summary(results, label, job.get("final_checkpoint", ""))
            initial_checkpoint = checkpoint_from_summary(results, label, job.get("initial_checkpoint", ""))
        except ValueError as error:
            errors.append(str(error))
            continue
        for stage, checkpoint in (("initial", initial_checkpoint), ("final", final_checkpoint)):
            model = checkpoint / "model.zip"
            metadata = checkpoint / "metadata.json"
            if not model.exists() or not metadata.exists():
                errors.append(f"missing {stage} checkpoint files: {checkpoint}")
                continue
            expected_timesteps = 0 if stage == "initial" else FULL_STEPS
            expected_updates = 0 if stage == "initial" else EXPECTED_FINAL_UPDATES
            expected_hash = job.get(f"{stage}_model_sha256")
            if file_hash(model) != expected_hash:
                errors.append(f"{stage} model hash mismatch: {checkpoint}")
            expected_payload_hash = job.get(f"{stage}_policy_payload_sha256")
            actual_payload_hash = zip_member_hash(model, "policy.pth")
            if expected_payload_hash and actual_payload_hash != expected_payload_hash:
                errors.append(f"{stage} policy payload hash mismatch: {checkpoint}")
            data = stable_baselines_data(model)
            if data.get("num_timesteps") != expected_timesteps:
                errors.append(f"{stage} model.zip timestep mismatch: {checkpoint}")
            if data.get("_n_updates") != expected_updates:
                errors.append(f"{stage} model.zip update mismatch: {checkpoint}")
            if data.get("n_steps") != 32 or data.get("batch_size") != 32 or data.get("n_epochs") != 4:
                errors.append(f"{stage} model.zip rollout config mismatch: {checkpoint}")
            meta = read_json(metadata)
            config = meta.get("config", {})
            if meta.get("num_timesteps") != expected_timesteps:
                errors.append(f"{stage} metadata timestep mismatch: {checkpoint}")
            if config.get("seed") != job.get("seed") or config.get("opponent_mode") != job.get("opponent_mode"):
                errors.append(f"{stage} metadata config seed/mode mismatch: {checkpoint}")
            if config.get("max_steps") != FULL_STEPS or config.get("max_turns") != 8:
                errors.append(f"{stage} metadata budget mismatch: {checkpoint}")
            tensor_digest = policy_tensor_digest(model)
            expected_tensor_digest = job.get(f"{stage}_parameter_digest")
            if expected_tensor_digest and tensor_digest != expected_tensor_digest:
                errors.append(f"{stage} policy tensor digest mismatch: {checkpoint}")
            if stage == "final":
                status_path = checkpoint.parents[0] / "status.json"
                status_file = read_json(status_path)
                if status_file.get("checkpoint") != checkpoint.name:
                    errors.append(f"status final checkpoint mismatch: {label}")
                if status_file.get("num_timesteps") != FULL_STEPS:
                    errors.append(f"status timestep mismatch: {label}")
                if status_file.get("additional_steps") != FULL_STEPS:
                    errors.append(f"status additional_steps mismatch: {label}")
            if stage == "initial":
                policy_hashes[job["seed"]][job["opponent_mode"]] = actual_payload_hash
        if final_checkpoint.name != EXPECTED_FINAL_CHECKPOINT:
            errors.append(f"final checkpoint path does not point at final status checkpoint: {label}")
    for seed, hashes in policy_hashes.items():
        if set(hashes) != set(MODES) or hashes.get("fixed-v1") != hashes.get("diverse-v1"):
            errors.append(f"initial policy payload pair mismatch for seed {seed}")
    return summary, errors


def normalized_trace_episodes(report: dict[str, object]) -> list[dict[str, object]]:
    trace = report.get("trace", [])
    if isinstance(trace, dict):
        episodes = trace.get("episodes", [])
        return episodes if isinstance(episodes, list) else []
    return trace if isinstance(trace, list) else []


def is_model_report(report: dict[str, object]) -> bool:
    return "selection" in report or report.get("policy") == "model"


def check_action_step(
    *,
    label: str,
    seed: object,
    index: int,
    step: dict[str, object],
    require_probabilities: bool,
    errors: list[str],
) -> None:
    action = step.get("action")
    if not isinstance(action, int) or not 0 <= action < ACTION_COUNT:
        errors.append(f"invalid action in {label} seed {seed} step {index}")
        return
    mask = step.get("action_mask")
    if not isinstance(mask, list) or len(mask) != ACTION_COUNT:
        errors.append(f"wrong action mask width in {label} seed {seed} step {index}")
        return
    if not all(isinstance(value, bool) for value in mask):
        errors.append(f"non-boolean action mask in {label} seed {seed} step {index}")
    if not mask[action]:
        errors.append(f"chosen illegal action in {label} seed {seed} step {index}")
    if "probabilities" not in step:
        if require_probabilities:
            errors.append(f"missing probabilities in model report {label} seed {seed} step {index}")
        return
    probabilities = step["probabilities"]
    if not isinstance(probabilities, list) or len(probabilities) != ACTION_COUNT:
        errors.append(f"wrong action probability width in {label} seed {seed} step {index}")
        return
    if not close_enough(sum(probabilities), 1.0, tolerance=1e-6):
        errors.append(f"probability sum mismatch in {label} seed {seed} step {index}")
    if any(prob != 0 for prob, allowed in zip(probabilities, mask, strict=True) if not allowed):
        errors.append(f"illegal action probability leak in {label} seed {seed} step {index}")
    ordered = sorted((float(prob) for prob in probabilities), reverse=True)
    expected_entropy = -sum(float(prob) * math.log(float(prob)) for prob in probabilities if prob > 0)
    checks = {
        "entropy_nats": expected_entropy,
        "top_probability": ordered[0],
        "top_margin": ordered[0] - ordered[1],
    }
    for key, expected in checks.items():
        if key not in step:
            errors.append(f"missing {key} in {label} seed {seed} step {index}")
        elif not close_enough(float(step[key]), expected, tolerance=1e-6):
            errors.append(f"{key} mismatch in {label} seed {seed} step {index}")


def audit_report_payload(label: str, report: dict[str, object], errors: list[str]) -> None:
    episodes = report.get("episodes", [])
    if not isinstance(episodes, list):
        errors.append(f"episodes is not a list in {label}")
        return
    if len(episodes) != len(EVALUATION_SEEDS):
        errors.append(f"wrong episode count in {label}")
    if report.get("episode_count") != len(episodes):
        errors.append(f"episode_count field mismatch in {label}")
    if [episode.get("seed") for episode in episodes] != list(EVALUATION_SEEDS):
        errors.append(f"episode seed order mismatch in {label}")
    reward_sum = sum(float(episode["reward"]) for episode in episodes)
    if episodes and not close_enough(report.get("mean_reward", 0), reward_sum / len(episodes)):
        errors.append(f"mean_reward mismatch in {label}")
    combat_count = sum(
        int(episode.get("wins", 0)) + int(episode.get("draws", 0)) + int(episode.get("losses", 0))
        for episode in episodes
    )
    if report.get("combat_count") != combat_count:
        errors.append(f"combat_count mismatch in {label}")
    if combat_count and not close_enough(
        report.get("combat_win_rate", 0),
        sum(int(episode.get("wins", 0)) for episode in episodes) / combat_count,
    ):
        errors.append(f"combat_win_rate mismatch in {label}")
    if episodes and not close_enough(
        report.get("survival_rate", 0),
        sum(bool(episode.get("survived")) for episode in episodes) / len(episodes),
    ):
        errors.append(f"survival_rate mismatch in {label}")
    trace = normalized_trace_episodes(report)
    if not trace:
        errors.append(f"missing trace in {label}")
        return
    if len(trace) != len(EVALUATION_SEEDS):
        errors.append(f"trace episode count mismatch in {label}")
    trace_by_seed = {row.get("seed"): row for row in trace}
    for episode in episodes:
        seed = episode.get("seed")
        if seed not in trace_by_seed:
            errors.append(f"missing trace seed {seed} in {label}")
            continue
        steps = trace_by_seed[seed].get("steps", [])
        if not steps:
            errors.append(f"empty trace seed {seed} in {label}")
            continue
        step_reward = sum(float(step.get("reward", 0.0)) for step in steps)
        if not close_enough(step_reward, episode["reward"]):
            errors.append(f"trace reward sum mismatch seed {seed} in {label}")
        forced = sum(bool(step.get("forced_end_turn", False)) for step in steps)
        if forced != episode.get("forced_end_turns", 0):
            errors.append(f"forced_end_turns mismatch seed {seed} in {label}")
        for index, step in enumerate(steps):
            check_action_step(
                label=label,
                seed=seed,
                index=index,
                step=step,
                require_probabilities=is_model_report(report),
                errors=errors,
            )


def report_summary(report: dict[str, object]) -> dict[str, float]:
    if report.get("policy") in {"heuristic", "random"}:
        return {
            key: report[key] for key in ("mean_reward", "combat_win_rate", "survival_rate")
        }
    return selection_summary(report)


def audit_eval_phase(
    results: Path,
    root: Path,
    cohort: str,
    train_summary: dict[str, object],
) -> tuple[dict[str, object], list[str]]:
    errors = []
    phase_dir = results / f"eval-{cohort}"
    summary = read_json(phase_dir / "summary.json")
    _, _, expected_opponent_hash = load_opponent_payload(root, cohort)
    if summary.get("opponent_set_sha256") != expected_opponent_hash:
        errors.append(f"{cohort} opponent hash mismatch")
    reported = summary.get("summary", {})
    if len(reported) != 50:
        errors.append(f"{cohort} expected 50 report summaries, got {len(reported)}")
    recalculated = {}
    expected_labels = set()
    for seed in TRAINING_SEEDS:
        for stage in ("initial", "final"):
            for mode in MODES:
                expected_labels.add(f"{cohort}-{stage}-{mode}-seed-{seed}-argmax")
                for policy_seed in POLICY_SEEDS:
                    expected_labels.add(f"{cohort}-{stage}-{mode}-seed-{seed}-sample-{policy_seed}")
    expected_labels.update({f"{cohort}-heuristic", f"{cohort}-random"})
    for path in sorted(phase_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        report = read_json(path)
        label = path.stem
        audit_report_payload(label, report, errors)
        if report.get("campaign_id") != CAMPAIGN_ID:
            errors.append(f"wrong campaign id: {path.name}")
        if report.get("seeds") != list(EVALUATION_SEEDS):
            errors.append(f"wrong evaluation seeds: {path.name}")
        if report.get("opponent_set_sha256") != expected_opponent_hash:
            errors.append(f"wrong opponent hash: {path.name}")
        if label not in expected_labels:
            errors.append(f"unexpected report label: {label}")
        parts = label.split("-")
        if is_model_report(report):
            if report.get("cohort") != cohort:
                errors.append(f"wrong cohort field: {path.name}")
            if report.get("stage") not in {"initial", "final"}:
                errors.append(f"wrong stage field: {path.name}")
            if report.get("opponent_mode") not in MODES:
                errors.append(f"wrong opponent_mode field: {path.name}")
            if report.get("training_seed") not in TRAINING_SEEDS:
                errors.append(f"wrong training_seed field: {path.name}")
            if str(report.get("training_seed")) not in parts:
                errors.append(f"label/training_seed mismatch: {path.name}")
        else:
            if label not in {f"{cohort}-heuristic", f"{cohort}-random"}:
                errors.append(f"unexpected baseline label: {label}")
        recalculated[label] = report_summary(report)
        if label not in reported:
            errors.append(f"report missing from summary: {label}")
            continue
        for key, value in recalculated[label].items():
            if key not in reported[label] or not close_enough(value, reported[label][key]):
                errors.append(f"summary mismatch for {label}.{key}")
    if len(recalculated) != 50:
        errors.append(f"{cohort} expected 50 JSON reports, got {len(recalculated)}")
    if set(recalculated) != expected_labels:
        errors.append(f"{cohort} report label set mismatch")
    train_jobs = train_summary.get("jobs", [])
    for job in train_jobs:
        seed = job["seed"]
        for policy_seed in (None, *POLICY_SEEDS):
            selection = "argmax" if policy_seed is None else f"sample-{policy_seed}"
            fixed = reported.get(f"{cohort}-initial-fixed-v1-seed-{seed}-{selection}")
            diverse = reported.get(f"{cohort}-initial-diverse-v1-seed-{seed}-{selection}")
            if fixed != diverse:
                errors.append(f"initial eval mismatch for seed {seed} {cohort} {selection}")
    return {"reported": reported, "recalculated": recalculated}, errors


def primary_decision(eval_summaries: dict[str, dict[str, object]]) -> dict[str, object]:
    per_seed = {}
    fixed_survival = []
    diverse_survival = []
    for seed in TRAINING_SEEDS:
        mode_scores = {}
        for mode in MODES:
            rewards = []
            survival = []
            for cohort, payload in eval_summaries.items():
                reported = payload["reported"]
                for policy_seed in POLICY_SEEDS:
                    label = f"{cohort}-final-{mode}-seed-{seed}-sample-{policy_seed}"
                    row = reported[label]
                    rewards.append(row["mean_reward"])
                    survival.append(row["survival_rate"])
            mode_scores[mode] = sum(rewards) / len(rewards)
            (fixed_survival if mode == "fixed-v1" else diverse_survival).extend(survival)
        per_seed[str(seed)] = {
            "fixed_mean_reward": mode_scores["fixed-v1"],
            "diverse_mean_reward": mode_scores["diverse-v1"],
            "delta": mode_scores["diverse-v1"] - mode_scores["fixed-v1"],
        }
    deltas = [row["delta"] for row in per_seed.values()]
    mean_delta = sum(deltas) / len(deltas)
    positive_seed_count = sum(delta > 0 for delta in deltas)
    survival_delta = (
        sum(diverse_survival) / len(diverse_survival)
        - sum(fixed_survival) / len(fixed_survival)
    )
    return {
        "per_training_seed": per_seed,
        "mean_delta": mean_delta,
        "positive_seed_count": positive_seed_count,
        "stochastic_survival_delta": survival_delta,
        "candidate_keep": mean_delta >= 0.25 and positive_seed_count >= 2 and survival_delta >= -0.05,
        "criterion": "mean_delta>=0.25 and at least two positive seeds and survival drop<=5pp",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--extract-to", type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.results) == bool(args.archive):
        raise ValueError("Provide exactly one of --results or --archive")
    root, signature = load_project(args.checkout, args.expected_source_sha256)
    results = (
        safe_extract(args.archive, args.extract_to or args.output.with_suffix(".extract"))
        if args.archive
        else args.results.resolve()
    )
    errors = []
    train_summary, train_errors = audit_training(results)
    errors.extend(train_errors)
    eval_summaries = {}
    for cohort in COHORTS:
        payload, eval_errors = audit_eval_phase(results, root, cohort, train_summary)
        eval_summaries[cohort] = payload
        errors.extend(eval_errors)
    opponent_hashes = {cohort: load_opponent_payload(root, cohort)[2] for cohort in COHORTS}
    report = {
        "campaign_id": CAMPAIGN_ID,
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "results_root": str(results),
        "compatibility": signature,
        "opponent_hashes": opponent_hashes,
        "opponent_hashes_sha256": json_hash(opponent_hashes),
        "training_summary_sha256": file_hash(results / "train" / "summary.json"),
        "primary_decision": primary_decision(eval_summaries) if not errors else None,
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()
