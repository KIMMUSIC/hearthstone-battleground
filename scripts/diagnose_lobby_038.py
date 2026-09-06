"""Read-only direct terminal reward contribution from fixed training logs."""

import argparse
import json
from pathlib import Path
import zipfile

from lobby_032 import digest, read
from run_diversity_009 import verify_files


def credit(episodes, total_steps, rollout, gamma, lam):
    if total_steps < 1 or rollout < 1 or not 0 < gamma <= 1 or not 0 < lam <= 1:
        raise ValueError("Invalid credit parameters")
    previous, covered, nonzero, magnitude = 0, 0, 0, 0.0
    lengths = []
    for episode in episodes:
        end = episode["num_timesteps"]
        if not previous < end <= total_steps or episode["truncated"] or episode["rank"] is None:
            raise ValueError("Need ordered complete ranked episodes")
        reward = (4.5 - episode["rank"]) / 3.5
        if not -1 <= reward <= 1:
            raise ValueError("Invalid rank reward")
        start = max(previous + 1, ((end - 1) // rollout) * rollout + 1)
        span = end - start + 1
        covered += span
        if reward != 0:
            nonzero += span
        magnitude += sum(abs(reward) * (gamma * lam) ** distance for distance in range(span))
        lengths.append(end - previous)
        previous = end
    return {"steps": total_steps, "completed_episodes": len(lengths),
            "mean_completed_episode_steps": sum(lengths) / len(lengths) if lengths else None,
            "unfinished_tail_steps": total_steps - previous, "rollout": rollout, "lambda": lam,
            "same_rollout_terminal_coverage": covered / total_steps,
            "nonzero_direct_reward_coverage": nonzero / total_steps,
            "mean_absolute_direct_terminal_reward_component": magnitude / total_steps}


def diagnose(root, output):
    if output.exists():
        raise FileExistsError("Preserve previous diagnostic")
    verify_files(root)
    audit = read(root / "audit.json")
    if audit["status"] != "passed":
        raise ValueError("Need verified source run")
    rows, hashes = {}, {}
    for seed in (7, 17, 27):
        for arm in ("control", "candidate"):
            folder = root / f"outputs/train-s{seed:03d}-{arm}"
            status_path = folder / "status.json"
            status = read(status_path)
            model_path = folder / status["checkpoint"] / "model.zip"
            for p in (status_path, model_path):
                hashes[p.relative_to(root).as_posix()] = digest(p)
            with zipfile.ZipFile(model_path) as z:
                model = json.loads(z.read("data"))
            if model["num_timesteps"] != 8192 or model["n_steps"] != 32 or model["gamma"] != 1 or model["gae_lambda"] != .95:
                raise ValueError("Unexpected source model")
            episodes = status["episodes"]
            rows[f"s{seed:03d}-{arm}"] = {
                "actual32": credit(episodes, 8192, 32, 1, .95),
                "counterfactual128": credit(episodes, 8192, 128, 1, .95),
                "counterfactual256": credit(episodes, 8192, 256, 1, .95),
                "counterfactual_lambda1": credit(episodes, 8192, 32, 1, 1),
            }
    verify_files(root)
    if any(digest(root / path) != sha for path, sha in hashes.items()):
        raise ValueError("Input changed during diagnosis")
    result = {"status": "passed", "run": str(root), "rows": rows, "input_hashes": hashes,
              "inputs_unchanged": True, "limitation": "Direct terminal reward component only; excludes value TD terms and bootstrap. Counterfactual partitions use existing trajectories and are not measured learning outcomes."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.run.resolve(), args.output.resolve()), allow_nan=False))
