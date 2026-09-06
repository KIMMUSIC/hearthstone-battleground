"""Apply frozen campaign gates; training seeds, not episodes, are repetitions."""

from itertools import product
import math
from statistics import mean


def decide(rows, seeds, gates, heuristic_reward):
    if len(seeds) != 3 or len(set(seeds)) != 3 or set(rows) != set(seeds):
        raise ValueError("All three predefined training seeds are required")
    deltas, survival, gaps = [], [], []
    for seed in seeds:
        if set(rows[seed]) != {0, 49152, 100000}:
            raise ValueError("All predefined milestones are required")
        initial, final = rows[seed][0], rows[seed][100000]
        values = [r[k] for r in rows[seed].values() for k in ("mean_reward", "survival_rate")]
        if not all(math.isfinite(v) for v in values + [heuristic_reward]):
            raise ValueError("Non-finite comparison metric")
        deltas.append(final["mean_reward"] - initial["mean_reward"])
        survival.append(final["survival_rate"] - initial["survival_rate"])
        gaps.append(final["mean_reward"] - heuristic_reward)
    progress = (mean(deltas) >= gates["mean_seed_reward_delta_min"]
                and sum(d > 0 for d in deltas) >= gates["positive_seeds_min"]
                and mean(survival) >= gates["mean_seed_survival_delta_min"])
    samples = sorted(mean(sample) for sample in product(deltas, repeat=3))

    def quantile(probability):
        position = probability * (len(samples) - 1)
        lower = int(position)
        upper = min(lower + 1, len(samples) - 1)
        return samples[lower] + (position - lower) * (samples[upper] - samples[lower])

    return {
        "independent_training_seeds": 3,
        "seed_reward_deltas": dict(zip(seeds, deltas, strict=True)),
        "mean_reward_delta": mean(deltas), "positive_seeds": sum(d > 0 for d in deltas),
        "mean_survival_delta": mean(survival), "mean_heuristic_reward_gap": mean(gaps),
        "learning_progress": progress,
        "million_step_candidate": progress and mean(gaps) >= gates["million_step_candidate_heuristic_reward_gap_min"],
        "reward_delta_descriptive_bootstrap_95": [quantile(0.025), quantile(0.975)],
        "interval_interpretation": "Exact resampling of three seed deltas; descriptive, low replication",
        "automatic_execution_authorized_by_result": False,
    }
