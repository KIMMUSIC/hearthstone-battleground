"""Supervised behavior diagnostic, separate from PPO reward optimization."""

import argparse
import json
from pathlib import Path
import sys

from lobby_032 import dump, evaluate, package, read
from lobby_035 import identity, run
from run_diversity_009 import require_runtime, verify_files


def jobs(spec):
    return ([("dataset", 7, "student")]
            + [(mode, seed, "student") for mode in ("train", "evaluate") for seed in spec["seeds"]]
            + [("baseline", 7, "student")])


def phase(root, index):
    spec = read(root / "configs/lobby041_imitation.json")
    require_runtime(spec)
    verify_files(root)
    sys.path.insert(0, str(root / "src"))
    import torch
    from hearthstone_ai.lobby_env import LobbyEnv
    from hearthstone_ai.lobby_imitation import build_dataset, evaluate_labels, train_imitation
    from hearthstone_ai.lobby_training import load_model
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print(json.dumps({"torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads()}), flush=True)
    job = jobs(spec)[index]
    mode, seed, _ = job
    output = root / "outputs" / identity(job)
    dataset = root / "outputs/dataset-s007-student"
    training = root / "outputs" / identity(("train", seed, "student"))
    if mode == "dataset":
        result = build_dataset(spec["dataset"], output)
    elif mode == "train":
        result = train_imitation(spec["train"] | {"seed": seed}, spec["imitation"], dataset, output)
        if result["optimizer_steps"] != spec["imitation"]["updates"] or result["stop_reason"] != "updates":
            raise RuntimeError("Incomplete supervised budget")
        env = LobbyEnv(observation_scale="fixed-v1")
        reports = {}
        for label, key in (("initial", "initial_checkpoint"), ("final", "checkpoint")):
            model = load_model(output / result[key], env)
            if model.num_timesteps != 0 or model._n_updates != 0:
                raise ValueError("Imitation mislabeled as PPO learning")
            reports[label] = evaluate_labels(model, dataset / "dev.npz")
        env.close()
        dump(output / "classification.json", reports)
    else:
        result = evaluate(root, spec, training=training, output=output,
                          labels=("initial", "final") if mode == "evaluate" else ("heuristic", "random"),
                          observation_scale="fixed-v1")
    verify_files(root)
    return result


def decide(classification, games):
    if sorted(classification) != [7, 17, 27] or sorted(games) != [7, 17, 27]:
        raise ValueError("All three seeds required")
    learned = [classification[s]["final"]["accuracy"] >= .8 and
               classification[s]["final"]["accuracy"] > classification[s]["initial"]["accuracy"] for s in classification]
    improvements = [games[s]["initial"]["mean_rank"] - games[s]["final"]["mean_rank"] for s in games]
    top4 = sum(games[s]["final"]["top4_fraction_ranked"] for s in games) / 3
    no_truncation = all(p["truncated"] == 0 for s in games.values() for p in s.values())
    return {"imitation_supported": sum(learned) >= 2, "successful_classification_seeds": sum(learned),
            "mean_rank_improvement": sum(improvements) / 3, "positive_seeds": sum(x > 0 for x in improvements),
            "mean_final_top4": top4, "no_truncation": no_truncation,
            "game_transfer_supported": sum(improvements) / 3 >= 1 and sum(x > 0 for x in improvements) >= 2 and top4 >= .1 and no_truncation}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["package", "run", "phase"])
    parser.add_argument("--name", default="lobby-041-local-r1")
    parser.add_argument("--index", type=int)
    parser.add_argument("--spec", choices=["lobby041_imitation.json"], default="lobby041_imitation.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    spec = read(root / "configs/lobby041_imitation.json")
    result = (package(root, args.name, [root / "experiments/LOBBY_041_IMITATION.md"]) if args.mode == "package" else
              phase(root, args.index) if args.mode == "phase" else
              run(root, args.spec, job_list=jobs(spec), phase_script="lobby_041.py"))
    print(json.dumps(result, allow_nan=False))
