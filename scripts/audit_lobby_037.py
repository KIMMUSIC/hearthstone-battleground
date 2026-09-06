"""Audit common game performance and separately measure initial Tanh saturation."""

import argparse
import json
from pathlib import Path
import sys

from audit_lobby_035 import audit
from diagnose_lobby_036 import activation_report, observation_hash_update
from lobby_032 import digest, dump, read
from run_diversity_009 import verify_files


def numerical_audit(root):
    import hashlib
    import numpy as np
    import torch

    verify_files(root)
    sys.path.insert(0, str(root / "src"))
    from hearthstone_ai import lobby_env
    from hearthstone_ai.lobby_policies import heuristic_policy
    from hearthstone_ai.lobby_training import load_model
    if not Path(lobby_env.__file__).resolve().is_relative_to(root / "src"):
        raise ValueError("Use frozen source")
    torch.set_num_threads(1)
    corpora = {scale: [] for scale in ("raw", "fixed-v1")}
    hashes = {scale: hashlib.sha256() for scale in corpora}
    envs = {scale: lobby_env.LobbyEnv(observation_scale=scale) for scale in corpora}
    for seed in range(20001, 20021):
        observations = {scale: env.reset(seed=seed)[0] for scale, env in envs.items()}
        done, step = False, 0
        while not done:
            for scale, obs in observations.items():
                corpora[scale].append({k: v.copy() for k, v in obs.items()})
                observation_hash_update(hashes[scale], seed, step, obs)
            if not np.array_equal(envs["raw"].action_masks(), envs["fixed-v1"].action_masks()):
                raise ValueError("Scaling changed actions")
            action = heuristic_policy(envs["raw"].game.view(0))
            outputs = {scale: env.step(action) for scale, env in envs.items()}
            if outputs["raw"][1:] != outputs["fixed-v1"][1:]:
                raise ValueError("Scaling changed reward or transition")
            observations = {scale: out[0] for scale, out in outputs.items()}
            done = outputs["raw"][2] or outputs["raw"][3]
            step += 1
        if envs["raw"].game.trace != envs["fixed-v1"].game.trace:
            raise ValueError("Scaling changed game trace")
    rows = {}
    for seed in (7, 17, 27):
        rows[seed] = {}
        for arm, scale in (("control", "raw"), ("candidate", "fixed-v1")):
            training = root / f"outputs/train-s{seed:03d}-{arm}"
            checkpoint = training / read(training / "status.json")["initial_checkpoint"]
            before = digest(checkpoint / "model.zip")
            model = load_model(checkpoint, envs[scale], allow_opponent_shift=True)
            report = activation_report(model, corpora[scale])
            if before != digest(checkpoint / "model.zip"):
                raise ValueError("Numerical audit changed model")
            rows[seed][arm] = {"model_sha256": before, "observation_scale": scale,
                               "policy_first_tanh": report["policy_first_tanh"]["saturation_abs_ge_0_95_fraction"],
                               "value_first_tanh": report["value_first_tanh"]["saturation_abs_ge_0_95_fraction"]}
    for env in envs.values():
        env.close()
    reductions = [rows[s]["control"]["policy_first_tanh"] - rows[s]["candidate"]["policy_first_tanh"] for s in rows]
    verify_files(root)
    return {"rows": rows, "mean_initial_policy_saturation_reduction": sum(reductions) / 3,
            "numerical_hypothesis_supported": sum(reductions) / 3 >= .20,
            "observation_count_per_scale": len(corpora["raw"]),
            "corpus_hashes": {scale: h.hexdigest() for scale, h in hashes.items()},
            "paired_heuristic_game_traces_equal": True,
            "limitation": "Numerical saturation reduction alone is not performance improvement."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    output = root / "scale-audit.json"
    if output.exists() or (root / "audit.json").exists():
        raise FileExistsError("Preserve prior audit")
    numeric = numerical_audit(root)
    performance = audit(root, "037")
    result = {"status": "passed", "numeric": numeric, "performance_decision": performance["decision"]}
    dump(output, result)
    print(json.dumps(result, allow_nan=False))
