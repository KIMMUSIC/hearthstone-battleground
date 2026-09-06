"""Reusable entrypoint for opponent-diversity-009 smoke, training, and evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

from selection_probe import probe, selection_summary


CAMPAIGN_ID = "opponent-diversity-009"
MODES = ("fixed-v1", "diverse-v1")
TRAINING_SEEDS = (7, 17, 27)
TRAINING_ORDER = tuple((seed, mode) for seed in TRAINING_SEEDS for mode in MODES)
EVALUATION_SEEDS = tuple(range(401, 421))
SMOKE_EVALUATION_SEEDS = (501, 502)
POLICY_SEEDS = (11, 22, 33)
COHORTS = {
    "reference": "handoff/holdout-008/reference-opponents.json",
    "shifted": "handoff/holdout-008/shifted-opponents.json",
}
OPPONENT_SPEC = {
    "version": "opponent-diversity-009-v1",
    "turn_counts": [1, 1, 2, 2, 3, 3, 4, 4],
    "fixed-v1": {"candidate_ids": [72387], "sampling": "constant"},
    "diverse-v1": {"candidate_ids": [42467, 72387], "sampling": "iid-uniform-with-replacement"},
}
FULL_STEPS = 8192
SMOKE_STEPS = 64
EXPECTED_FINAL_CHECKPOINT = "checkpoint-000000008192-0002"
EXPECTED_FINAL_UPDATES = 1024


def json_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    else:
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


def configure_torch_threads() -> None:
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def require_empty(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {path}")


def load_project(checkout: Path, expected_source_sha256: str):
    root = checkout.resolve()
    sys.path.insert(0, str(root / "src"))
    from hearthstone_ai.artifacts import compatibility_signature
    from hearthstone_ai.cards import Catalog

    signature = compatibility_signature()
    if signature["source_sha256"] != expected_source_sha256:
        raise ValueError(
            "Source hash mismatch: "
            f"expected {expected_source_sha256}, got {signature['source_sha256']}"
        )
    catalog = Catalog(root / "data")
    for card_id in (42467, 72387):
        card = catalog.by_id[card_id]
        if card.tier != 1:
            raise ValueError(f"Campaign card must stay tier one: {card_id}")
    return root, signature


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def training_config(root: Path, *, seed: int, mode: str, smoke: bool) -> dict[str, Any]:
    filename = "smoke_train.json" if smoke else "research_train.json"
    config = read_json(root / "configs" / filename)
    config.update(
        seed=seed,
        opponent_mode=mode,
        max_steps=SMOKE_STEPS if smoke else FULL_STEPS,
        checkpoint_interval=SMOKE_STEPS if smoke else FULL_STEPS,
        max_turns=8,
        max_actions=24,
        threads=1,
    )
    if not smoke:
        config.update(n_steps=32, batch_size=32, n_epochs=4, max_seconds=60)
    return config


def parameter_digest(checkpoint: Path, env) -> str:
    import torch
    from hearthstone_ai.training import load_model

    model = load_model(checkpoint, env)
    digest = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        array = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.numpy().tobytes())
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return digest.hexdigest()


def zip_member_hash(archive: Path, member: str) -> str:
    import zipfile

    with zipfile.ZipFile(archive) as zf:
        return hashlib.sha256(zf.read(member)).hexdigest()


def load_opponent_payload(root: Path, cohort: str) -> tuple[Path, dict[str, Any], str]:
    path = root / COHORTS[cohort]
    payload = read_json(path)
    if not isinstance(payload.get("opponents"), list) or not payload["opponents"]:
        raise ValueError(f"Invalid opponent payload: {path}")
    return path, payload, json_hash(payload)


def manifest_base(root: Path, signature: dict[str, str]) -> dict[str, Any]:
    return {
        "campaign_id": CAMPAIGN_ID,
        "compatibility": signature,
        "entrypoint_sha256": file_hash(Path(__file__)),
        "selection_probe_sha256": file_hash(Path(__file__).with_name("selection_probe.py")),
        "opponent_spec": OPPONENT_SPEC,
        "training_order": [{"seed": seed, "opponent_mode": mode} for seed, mode in TRAINING_ORDER],
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "policy_seeds": list(POLICY_SEEDS),
        "cohorts": {name: str((root / rel).resolve()) for name, rel in COHORTS.items()},
    }


def phase_plan(root: Path, output: Path, signature: dict[str, str], python: str) -> dict[str, Any]:
    require_empty(output / "plan")
    commands = {
        "smoke": [
            python,
            "scripts/diversity_009.py",
            "--checkout",
            str(root),
            "--output",
            str(output),
            "--expected-source-sha256",
            signature["source_sha256"],
            "--phase",
            "smoke",
        ],
        "train": [
            python,
            "scripts/diversity_009.py",
            "--checkout",
            str(root),
            "--output",
            str(output),
            "--expected-source-sha256",
            signature["source_sha256"],
            "--phase",
            "train",
        ],
        "reference": [
            python,
            "scripts/diversity_009.py",
            "--checkout",
            str(root),
            "--output",
            str(output),
            "--expected-source-sha256",
            signature["source_sha256"],
            "--phase",
            "reference",
        ],
        "shifted": [
            python,
            "scripts/diversity_009.py",
            "--checkout",
            str(root),
            "--output",
            str(output),
            "--expected-source-sha256",
            signature["source_sha256"],
            "--phase",
            "shifted",
        ],
    }
    plan = manifest_base(root, signature) | {
        "phase": "plan",
        "commands": commands,
        "expected_outputs": {
            "smoke": "smoke/summary.json plus two <=64-step mode run directories",
            "train": "train/summary.json plus six fresh 8192-step run directories",
            "reference": "eval-reference/summary.json plus 50 JSON reports",
            "shifted": "eval-shifted/summary.json plus 50 JSON reports",
            "audit": "audit report from scripts/audit_diversity_009.py over the returned result root",
        },
        "supervisor_limits": {
            "timeout_seconds_per_phase": 180,
            "rss_limit_bytes": 2147483648,
            "threads": 1,
        },
    }
    write_json(output / "plan" / "plan.json", plan)
    return plan


def phase_smoke(root: Path, output: Path, signature: dict[str, str]) -> dict[str, Any]:
    from hearthstone_ai.env import BgEnv
    from hearthstone_ai.evaluation import evaluate
    from hearthstone_ai.training import train

    configure_torch_threads()
    phase_dir = output / "smoke"
    require_empty(phase_dir)
    opponent_path, opponent_payload, opponent_hash = load_opponent_payload(root, "reference")
    jobs = []
    for mode in MODES:
        run_dir = phase_dir / f"{mode}-seed-7-steps-64"
        started = time.monotonic()
        status = train(training_config(root, seed=7, mode=mode, smoke=True), run_dir)
        if (
            status["status"] != "completed"
            or status["stop_reason"] != "steps"
            or status["additional_steps"] != SMOKE_STEPS
            or status["num_timesteps"] != SMOKE_STEPS
        ):
            raise RuntimeError(f"Smoke training did not complete exactly 64 steps: {status}")
        checkpoint = run_dir / status["checkpoint"]
        env = BgEnv(opponents=opponent_payload["opponents"], max_turns=8, max_actions=24)
        try:
            digest = parameter_digest(checkpoint, env)
        finally:
            env.close()
        report = evaluate(
            policy="model",
            seeds=SMOKE_EVALUATION_SEEDS,
            opponent_path=opponent_path,
            checkpoint=checkpoint,
            max_turns=8,
            max_actions=24,
            trace=True,
        )
        write_json(run_dir / "fixed-eval-load.json", report, compact=True)
        metadata = read_json(checkpoint / "metadata.json")
        if metadata["config"].get("opponent_mode") != mode:
            raise RuntimeError("Checkpoint metadata did not preserve opponent_mode")
        jobs.append(
            {
                "opponent_mode": mode,
                "status": status,
                "train_seconds": time.monotonic() - started,
                "checkpoint": str(checkpoint),
                "model_sha256": file_hash(checkpoint / "model.zip"),
                "policy_payload_sha256": zip_member_hash(checkpoint / "model.zip", "policy.pth"),
                "parameter_digest": digest,
                "evaluation_mean_reward": report["mean_reward"],
            }
        )
    summary = manifest_base(root, signature) | {
        "phase": "smoke",
        "smoke_evaluation_seeds": list(SMOKE_EVALUATION_SEEDS),
        "opponent_set_sha256": opponent_hash,
        "jobs": jobs,
    }
    write_json(phase_dir / "summary.json", summary)
    return summary


def phase_train(root: Path, output: Path, signature: dict[str, str]) -> dict[str, Any]:
    from hearthstone_ai.env import BgEnv
    from hearthstone_ai.training import load_model, train

    configure_torch_threads()
    phase_dir = output / "train"
    require_empty(phase_dir)
    jobs = []
    initial_digests: dict[int, dict[str, str]] = {seed: {} for seed in TRAINING_SEEDS}
    for seed, mode in TRAINING_ORDER:
        label = f"{mode}-seed-{seed}-steps-8192"
        run_dir = phase_dir / label
        config = training_config(root, seed=seed, mode=mode, smoke=False)
        started = time.monotonic()
        status = train(config, run_dir)
        if status["status"] != "completed" or status["stop_reason"] != "steps":
            raise RuntimeError(f"Training did not finish by step budget: {label}")
        if status["num_timesteps"] != FULL_STEPS or status["checkpoint"] != EXPECTED_FINAL_CHECKPOINT:
            raise RuntimeError(f"Unexpected final checkpoint for {label}: {status}")
        final_checkpoint = run_dir / status["checkpoint"]
        initial_checkpoint = run_dir / status["initial_checkpoint"]
        env = BgEnv(max_turns=8, max_actions=24)
        try:
            model = load_model(final_checkpoint, env)
            if model._n_updates != EXPECTED_FINAL_UPDATES:
                raise RuntimeError(f"Unexpected optimizer update count for {label}")
            final_parameter_digest = parameter_digest(final_checkpoint, env)
            initial_parameter_digest = parameter_digest(initial_checkpoint, env)
        finally:
            env.close()
        metadata = read_json(final_checkpoint / "metadata.json")
        if metadata["config"].get("opponent_mode") != mode:
            raise RuntimeError(f"Final metadata lost opponent_mode for {label}")
        initial_digests[seed][mode] = initial_parameter_digest
        entry = {
            "label": label,
            "seed": seed,
            "opponent_mode": mode,
            "config": config,
            "status": status,
            "train_seconds": time.monotonic() - started,
            "initial_checkpoint": str(initial_checkpoint),
            "final_checkpoint": str(final_checkpoint),
            "initial_model_sha256": file_hash(initial_checkpoint / "model.zip"),
            "final_model_sha256": file_hash(final_checkpoint / "model.zip"),
            "initial_policy_payload_sha256": zip_member_hash(
                initial_checkpoint / "model.zip", "policy.pth"
            ),
            "final_policy_payload_sha256": zip_member_hash(final_checkpoint / "model.zip", "policy.pth"),
            "initial_parameter_digest": initial_parameter_digest,
            "final_parameter_digest": final_parameter_digest,
            "optimizer_updates": EXPECTED_FINAL_UPDATES,
        }
        jobs.append(entry)
        write_json(phase_dir / "summary.json", manifest_base(root, signature) | {"jobs": jobs})
    pair_matches = {
        str(seed): initial_digests[seed]["fixed-v1"] == initial_digests[seed]["diverse-v1"]
        for seed in TRAINING_SEEDS
    }
    if not all(pair_matches.values()):
        raise RuntimeError(f"Initial parameter digests differ by mode: {pair_matches}")
    summary = manifest_base(root, signature) | {
        "phase": "train",
        "jobs": jobs,
        "initial_parameter_pair_matches": pair_matches,
    }
    write_json(phase_dir / "summary.json", summary)
    return summary


def evaluation_cases(train_summary: dict[str, Any]) -> list[dict[str, Any]]:
    cases = []
    for job in train_summary["jobs"]:
        for stage in ("initial", "final"):
            checkpoint = Path(job[f"{stage}_checkpoint"])
            cases.append(
                {
                    "stage": stage,
                    "seed": job["seed"],
                    "opponent_mode": job["opponent_mode"],
                    "checkpoint": checkpoint,
                    "model_sha256": job[f"{stage}_model_sha256"],
                    "parameter_digest": job[f"{stage}_parameter_digest"],
                }
            )
    return cases


def phase_eval(root: Path, output: Path, signature: dict[str, str], cohort: str) -> dict[str, Any]:
    from hearthstone_ai.env import BgEnv
    from hearthstone_ai.evaluation import evaluate
    from hearthstone_ai.training import load_model

    configure_torch_threads()
    phase_dir = output / f"eval-{cohort}"
    require_empty(phase_dir)
    train_summary = read_json(output / "train" / "summary.json")
    opponent_path, opponent_payload, opponent_hash = load_opponent_payload(root, cohort)
    summary: dict[str, Any] = {}
    for case in evaluation_cases(train_summary):
        checkpoint = case["checkpoint"]
        if file_hash(checkpoint / "model.zip") != case["model_sha256"]:
            raise RuntimeError(f"Checkpoint changed before evaluation: {checkpoint}")
        for policy_seed in (None, *POLICY_SEEDS):
            env = BgEnv(opponents=opponent_payload["opponents"], max_turns=8, max_actions=24)
            try:
                model = load_model(checkpoint, env)
                report = probe(model, env, EVALUATION_SEEDS, policy_seed)
            finally:
                env.close()
            if file_hash(checkpoint / "model.zip") != case["model_sha256"]:
                raise RuntimeError(f"Evaluation mutated checkpoint: {checkpoint}")
            report.update(
                campaign_id=CAMPAIGN_ID,
                cohort=cohort,
                stage=case["stage"],
                training_seed=case["seed"],
                opponent_mode=case["opponent_mode"],
                model_sha256=case["model_sha256"],
                parameter_digest=case["parameter_digest"],
                compatibility=signature,
                opponent_set_sha256=opponent_hash,
            )
            selection = "argmax" if policy_seed is None else f"sample-{policy_seed}"
            label = (
                f"{cohort}-{case['stage']}-{case['opponent_mode']}-"
                f"seed-{case['seed']}-{selection}"
            )
            write_json(phase_dir / f"{label}.json", report, compact=True)
            summary[label] = selection_summary(report)
    for policy in ("heuristic", "random"):
        report = evaluate(
            policy=policy,
            seeds=EVALUATION_SEEDS,
            opponent_path=opponent_path,
            max_turns=8,
            max_actions=24,
            trace=True,
        )
        report.update(campaign_id=CAMPAIGN_ID, cohort=cohort, opponent_set_sha256=opponent_hash)
        label = f"{cohort}-{policy}"
        write_json(phase_dir / f"{label}.json", report, compact=True)
        summary[label] = {
            key: report[key] for key in ("mean_reward", "combat_win_rate", "survival_rate")
        }
    if len(summary) != 50:
        raise RuntimeError(f"Expected 50 reports for cohort {cohort}, got {len(summary)}")
    result = manifest_base(root, signature) | {
        "phase": f"eval-{cohort}",
        "cohort": cohort,
        "opponent_set_sha256": opponent_hash,
        "summary": summary,
    }
    write_json(phase_dir / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument(
        "--phase",
        choices=("plan", "smoke", "train", "eval-reference", "eval-shifted", "reference", "shifted"),
        required=True,
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable for plan output")
    args = parser.parse_args()

    root, signature = load_project(args.checkout, args.expected_source_sha256)
    output = args.output.resolve()
    if args.phase == "plan":
        result = phase_plan(root, output, signature, args.python)
    elif args.phase == "smoke":
        result = phase_smoke(root, output, signature)
    elif args.phase == "train":
        result = phase_train(root, output, signature)
    else:
        cohort = args.phase.removeprefix("eval-")
        result = phase_eval(root, output, signature, cohort)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
