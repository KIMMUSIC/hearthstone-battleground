"""Frozen DISCOUNT_024 gates using three paired training seeds as repetitions."""

from itertools import product
import math
from statistics import mean


def decide(reports, heuristic_reward=3.4):
    seeds = ("7", "17", "27")
    if set(reports) != set(seeds):
        raise ValueError("Exactly the three predefined training seeds are required")
    if not math.isfinite(heuristic_reward):
        raise ValueError("Non-finite baseline")
    for row in reports.values():
        if set(row) != {"control", "candidate"}:
            raise ValueError("Both conditions are required")
        for arm in row.values():
            if set(arm) != {"initial", "final"}:
                raise ValueError("Initial and final evidence required")
            for summary in arm.values():
                for key in ("mean_reward", "survival_rate"):
                    value = summary.get(key)
                    if (isinstance(value, bool) or not isinstance(value, (float, int))
                            or not math.isfinite(value)):
                        raise ValueError("Missing or non-finite metric")
                if not 0 <= summary["survival_rate"] <= 1:
                    raise ValueError("Survival rate outside [0,1]")
    paired, survival, learning, learning_survival, gaps = [], [], [], [], []
    for seed in seeds:
        row = reports[seed]
        final, initial = row["candidate"]["final"], row["candidate"]["initial"]
        control = row["control"]["final"]
        paired.append(final["mean_reward"] - control["mean_reward"])
        survival.append(final["survival_rate"] - control["survival_rate"])
        learning.append(final["mean_reward"] - initial["mean_reward"])
        learning_survival.append(final["survival_rate"] - initial["survival_rate"])
        gaps.append(final["mean_reward"] - heuristic_reward)
    supported = (mean(paired) >= 0.5 and sum(d > 0 for d in paired) >= 2
                 and mean(survival) >= -0.05)
    progress = (mean(learning) >= 0.5 and sum(d > 0 for d in learning) >= 2
                and mean(learning_survival) >= -0.05)
    selected = supported and progress and mean(gaps) >= -0.5
    samples = sorted(mean(sample) for sample in product(paired, repeat=3))

    def quantile(p):
        position = p * (len(samples) - 1)
        lower = int(position)
        return samples[lower] + (position - lower) * (samples[lower + 1] - samples[lower])

    return {
        "independent_training_seeds": 3,
        "paired_reward_deltas": dict(zip(seeds, paired, strict=True)),
        "mean_paired_reward_delta": mean(paired),
        "positive_paired_seeds": sum(d > 0 for d in paired),
        "mean_paired_survival_delta": mean(survival),
        "candidate_learning_deltas": dict(zip(seeds, learning, strict=True)),
        "mean_candidate_learning_delta": mean(learning),
        "mean_candidate_survival_delta": mean(learning_survival),
        "mean_heuristic_reward_gap": mean(gaps),
        "hypothesis_supported": supported, "candidate_learning_progress": progress,
        "candidate_selected": selected,
        "selected_training_seeds": [7, 17, 27] if selected else [],
        "paired_reward_delta_descriptive_95": [quantile(0.025), quantile(0.975)],
        "interval_interpretation": "Exact resampling of three paired seed deltas; descriptive, low replication",
    }
