"""Evaluate frozen epoch candidates on preregistered held-out scenarios; no training."""

import argparse
import json
from pathlib import Path
import sys

from selection_probe import probe, selection_summary


def validate_spec(spec):
    seeds = spec['seeds']
    if seeds != list(range(301, 321)):
        raise ValueError('This holdout requires fixed seeds 301-320')
    cases = spec['cases']
    if len(cases) != 6 or {(c['epochs'], c['seed']) for c in cases} != {
            (e, s) for e in (1, 4) for s in (7, 17, 27)}:
        raise ValueError('Requires all six epoch candidates')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    validate_spec(spec)
    sys.path.insert(0, str(args.checkout.resolve() / 'src'))
    import torch
    from hearthstone_ai.artifacts import compatibility_signature, content_hash, file_hash
    from hearthstone_ai.env import BgEnv
    from hearthstone_ai.evaluation import evaluate
    from hearthstone_ai.training import load_model

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    signature = compatibility_signature()
    if signature['source_sha256'] != '80b5ca7a81687165553578ef28f581025d754a6759a04f42ae8d00387c5b97a8':
        raise ValueError('Requires frozen snapshot-003')
    for case in spec['cases']:
        path = Path(case['checkpoint'])
        if file_hash(path / 'model.zip') != case['sha256']:
            raise ValueError('Model hash differs from audited baseline')
        meta = json.loads((path / 'metadata.json').read_text())
        if meta['config']['seed'] != case['seed'] or meta['config']['n_epochs'] != case['epochs']:
            raise ValueError('Wrong candidate configuration')
    cohorts = {}
    for name, filename in spec['cohorts'].items():
        path = args.spec.parent / filename
        payload = json.loads(path.read_text())
        env = BgEnv(opponents=payload['opponents'])
        env.close()
        cohorts[name] = (path, payload)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(spec=spec, compatibility=signature, additional_training_steps=0,
                    script_sha256=file_hash(Path(__file__)),
                    probe_sha256=file_hash(Path(__file__).with_name('selection_probe.py')),
                    torch_threads=1, torch_interop_threads=1,
                    opponent_hashes={n: content_hash(p) for n, (_, p) in cohorts.items()})
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    summary = {}
    for cohort, (path, payload) in cohorts.items():
        for case in spec['cases']:
            checkpoint = Path(case['checkpoint'])
            for policy_seed in (None, 11, 22, 33):
                env = BgEnv(opponents=payload['opponents'])
                try:
                    model = load_model(checkpoint, env)
                    if model.num_timesteps != 8192 or model._n_updates != 256 * case['epochs']:
                        raise ValueError('Candidate has wrong training budget')
                    report = probe(model, env, spec['seeds'], policy_seed)
                finally:
                    env.close()
                report.update(model_sha256=case['sha256'], compatibility=signature,
                              opponent_set_sha256=manifest['opponent_hashes'][cohort])
                mode = 'argmax' if policy_seed is None else f'sample-{policy_seed}'
                label = f'{cohort}-epoch-{case["epochs"]}-seed-{case["seed"]}-{mode}'
                (args.output / f'{label}.json').write_text(json.dumps(report, allow_nan=False))
                summary[label] = selection_summary(report)
        for policy in ('heuristic', 'random'):
            report = evaluate(policy=policy, seeds=spec['seeds'], opponent_path=path, trace=True)
            label = f'{cohort}-{policy}'
            (args.output / f'{label}.json').write_text(json.dumps(report, allow_nan=False))
            summary[label] = {k: report[k] for k in ('mean_reward', 'combat_win_rate', 'survival_rate')}
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    for case in spec['cases']:
        if file_hash(Path(case['checkpoint']) / 'model.zip') != case['sha256']:
            raise ValueError('Evaluation mutated checkpoint')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
