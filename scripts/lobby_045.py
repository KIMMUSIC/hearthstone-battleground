"""Fixed-budget teacher-only versus teacher plus visited-state data comparison."""

import argparse
import json
from pathlib import Path
import sys

from lobby_032 import dump, evaluate, package, read
from lobby_035 import identity, run
from run_diversity_009 import require_runtime, verify_files


def jobs(spec):
    return ([("dataset", 7, "teacher")]
            + [("collect", seed, "candidate") for seed in spec["seeds"]]
            + [(mode, seed, arm) for mode in ("train", "evaluate") for seed in spec["seeds"] for arm in ("control", "candidate")]
            + [("baseline", 7, "control")])


def phase(root, index):
    spec = read(root / "configs/lobby045_comparison.json")
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
    mode, seed, arm = job
    output = root / "outputs" / identity(job)
    base = root / "outputs/dataset-s007-teacher"
    dataset = base if arm != "candidate" else root / "outputs" / identity(("collect", seed, arm))
    training = root / "outputs" / identity(("train", seed, arm))
    if mode == "dataset":
        result = build_dataset(spec["dataset"], output)
    elif mode == "collect":
        from collect_lobby_045 import collect
        parent = spec["collectors"][str(seed)]
        checkpoint = root / parent["checkpoint"]
        from lobby_032 import digest
        if digest(checkpoint / "model.zip") != parent["model_sha256"] or digest(checkpoint / "metadata.json") != parent["metadata_sha256"]:
            raise ValueError("Collector checkpoint hash mismatch")
        result = collect(base, checkpoint, output, model_seed=seed, seeds=spec["dataset"]["train_seeds"], max_rounds=spec["train"]["max_rounds"])
    elif mode == "train":
        result = train_imitation(spec["train"] | {"seed": seed}, spec["imitation"] | {"sampling": spec["arms"][arm]}, dataset, output)
        if result["optimizer_steps"] != spec["imitation"]["updates"] or result["stop_reason"] != "updates":
            raise RuntimeError("Incomplete supervised budget")
        env = LobbyEnv(observation_scale="fixed-v1")
        reports = {}
        for label, key in (("initial", "initial_checkpoint"), ("final", "checkpoint")):
            model = load_model(output / result[key], env)
            if model.num_timesteps != 0 or model._n_updates != 0:
                raise ValueError("Imitation mislabeled as PPO learning")
            reports[label] = {split: evaluate_labels(model, dataset / f"{split}.npz") for split in ("train", "dev")}
        env.close()
        dump(output / "classification.json", reports)
    else:
        result = evaluate(root, spec, training=training, output=output,
                          labels=("initial", "final") if mode == "evaluate" else ("heuristic", "random"),
                          observation_scale="fixed-v1")
        if mode == "evaluate":
            status = read(training / "status.json")
            dump(output / "visited.json", visited_metrics(training / status["checkpoint"], read(output / "final.json"), spec))
    verify_files(root)
    return result


def visited_metrics(checkpoint, report, spec):
    from hearthstone_ai.lobby_env import LobbyEnv
    from hearthstone_ai.lobby_policies import heuristic_policy
    from hearthstone_ai.lobby_training import load_model
    from diagnose_lobby_044 import empty_metrics, record_prediction, summarize_prediction_metrics
    env = LobbyEnv(learner_seat=spec["train"]["learner_seat"],
                   max_rounds=spec["train"]["max_rounds"], observation_scale="fixed-v1")
    model = load_model(checkpoint, env)
    metrics = empty_metrics()
    try:
        for episode in report["episodes"]:
            obs, _ = env.reset(seed=episode["seed"])
            done = False
            total_reward = 0.0
            for action in episode["decisions"]:
                if done:
                    raise ValueError("Visited-state actions continue after terminal")
                predicted = int(model.predict(obs, deterministic=True, action_masks=env.action_masks())[0])
                if predicted != action:
                    raise ValueError("Visited-state checkpoint action differs")
                record_prediction(metrics, heuristic_policy(env.game.view(0)), predicted)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
                if truncated:
                    raise ValueError("Visited-state replay truncated")
            if not done or total_reward != episode["reward"]:
                raise ValueError("Visited-state termination/reward differs")
            if env.game.players[0].rank != episode["rank"]:
                raise ValueError("Visited-state rank differs")
        return summarize_prediction_metrics(metrics) | {"episodes": len(report["episodes"]), "denominator_source": "final model visited states"}
    finally:
        env.close()


def package_inputs(root, name, spec):
    from lobby_032 import digest
    previous = root / "runs/lobby-043-local-r1/src/hearthstone_ai"
    current = root / "src/hearthstone_ai"
    if {p.name: digest(p) for p in current.glob("*.py")} != {p.name: digest(p) for p in previous.glob("*.py")}:
        raise ValueError("045 must preserve 043 source compatibility")
    extra = [root / "experiments/LOBBY_045_VISITED_TRAINING.md"]
    for parent in spec["collectors"].values():
        checkpoint = root / parent["checkpoint"]
        for filename, key in (("model.zip", "model_sha256"), ("metadata.json", "metadata_sha256")):
            path = checkpoint / filename
            if digest(path) != parent[key]:
                raise ValueError("Collector changed before packaging")
            extra.append(path)
    return package(root, name, extra)


def decide(classification, games):
    if sorted(classification) != [7, 17, 27] or sorted(games) != [7, 17, 27]:
        raise ValueError("All three seeds required")
    macro = [classification[s]["candidate"]["final"]["dev"]["macro_action_accuracy"] -
             classification[s]["control"]["final"]["dev"]["macro_action_accuracy"] for s in sorted(classification)]
    accuracy_floor = all(classification[s]["candidate"]["final"]["dev"]["accuracy"] >= .75 for s in classification)
    ranks = [games[s]["control"]["final"]["mean_rank"] - games[s]["candidate"]["final"]["mean_rank"] for s in sorted(games)]
    top4 = sum(games[s]["candidate"]["final"]["top4_fraction_ranked"] - games[s]["control"]["final"]["top4_fraction_ranked"] for s in games) / 3
    no_truncation = all(p["truncated"] == 0 for s in games.values() for arm in s.values() for p in arm.values())
    classification_supported = accuracy_floor
    game_supported = sum(ranks) / 3 >= .5 and sum(v > 0 for v in ranks) >= 2 and top4 >= 0 and no_truncation
    return {"macro_differences": macro, "mean_macro_difference": sum(macro)/3,
            "positive_classification_seeds": sum(v > 0 for v in macro), "candidate_accuracy_floor_met": accuracy_floor,
            "classification_supported": classification_supported, "rank_improvements": ranks,
            "mean_rank_improvement": sum(ranks)/3, "positive_game_seeds": sum(v > 0 for v in ranks),
            "top4_difference": top4, "no_truncation": no_truncation, "game_transfer_supported": game_supported,
            "candidate_keep": classification_supported and game_supported}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["package", "run", "phase"])
    parser.add_argument("--name", default="lobby-045-local-r1")
    parser.add_argument("--index", type=int)
    parser.add_argument("--spec", choices=["lobby045_comparison.json"], default="lobby045_comparison.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    spec = read(root / "configs/lobby045_comparison.json")
    result = (package_inputs(root, args.name, spec) if args.mode == "package" else
              phase(root, args.index) if args.mode == "phase" else
              run(root, args.spec, job_list=jobs(spec), phase_script="lobby_045.py"))
    print(json.dumps(result, allow_nan=False))
