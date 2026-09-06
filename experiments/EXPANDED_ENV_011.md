# expanded-env-011 — archival-subset-v3 baseline and 100k campaign

> **Superseded diagnostic record, not an executable plan.** The v3 pool incorrectly included two summon tokens and counted an alternate Annoy-o-Tron ID as expansion. The 64-step diagnostic lacks the required frozen-package/supervisor evidence, and the 10,800-second budget was not adopted. Preserve its completed run files, but use [EXPANSION_019](EXPANSION_019.md) and [EXPANSION_020_CAMPAIGN](EXPANSION_020_CAMPAIGN.md) for the corrected work. The three former config files are preserved in `experiments/archive/expanded-011-diagnostic/`; do not dispatch them.

This experiment uses the local WSL Ubuntu Python 3.11 environment and the `archival-subset-v3` rules contract. It does not reuse 009 checkpoints because the card mapping, observation width, source hash, and rules hash changed.

## Environment

- Working tree: `D:\HearthStoneAI-Rebuild`
- WSL Python: `/home/hwa3060/hsai-local/venv311/bin/python`
- Training config template: `configs/expanded_100k_train.json`
- Development evaluation opponents: `configs/eval_opponents_extended.json`
- Final evaluation opponents: `configs/final_eval_opponents_extended.json`

## Preflight Evidence

The 9/19 smoke run `runs/expanded-011-smoke-r1` completed 64 steps and produced checkpoint `checkpoint-000000000064-0003`. The model loaded under `onehot-slots-v3` and evaluated on the expanded development opponent file.

Baseline development evaluation on seeds 501-520:

| Policy | Episodes | Mean reward | Combat win rate | Survival |
| --- | ---: | ---: | ---: | ---: |
| heuristic | 20 | 5.20 | 0.74375 | 1.00 |
| random | 20 | -6.55 | 0.00 | 0.00 |

These are development baselines only. They are not final skill claims.

## 100k Training Plan

Run three independent seeds with one output directory per seed:

| Run | Config seed | Output directory |
| --- | ---: | --- |
| expanded-011-fixed-seed-701 | 701 | `runs/expanded-011-100k-r1/seed-701` |
| expanded-011-fixed-seed-711 | 711 | `runs/expanded-011-100k-r1/seed-711` |
| expanded-011-fixed-seed-721 | 721 | `runs/expanded-011-100k-r1/seed-721` |

`max_steps=100000` is exactly 3,125 rollouts at `n_steps=32`. Use `threads=1`, `n_epochs=4`, and `checkpoint_interval=10000`. Keep `max_seconds=10800` per seed as the first local upper bound. If a seed stops by time instead of steps, preserve its checkpoint and status rather than increasing the budget silently.

Evaluate initial, intermediate if available, and final checkpoints against the development opponent file. Choose a candidate before using the final opponent file. Do not inspect or tune against `configs/final_eval_opponents_extended.json` results before candidate freeze.

## Pass Criteria

- All three seed directories are new and nonempty only after their run starts.
- Each status records actual steps, stop reason, checkpoint list, compatibility signature, and training metrics.
- Development evaluation records reward, combat results, survival, damage, forced turns, rerolls, freezes, swaps, and action counts.
- Compare against both heuristic and random baselines. A learned policy that beats random but remains far below heuristic is not a success claim; it is only a learning signal.
