"""Fresh-run learning budgets, supervised externally; immutable snapshot-003 source."""

import argparse
import json
from pathlib import Path
import sys
import time

from selection_probe import probe, selection_summary


def validate_jobs(jobs):
    if not isinstance(jobs, list) or not jobs:
        raise ValueError('jobs must be a nonempty list')
    seen = set()
    for job in jobs:
        if set(job) != {'seed', 'steps'}:
            raise ValueError('Each job requires exactly seed and steps')
        if type(job['seed']) is not int or job['seed'] < 0:
            raise ValueError('Invalid seed')
        if type(job['steps']) is not int or job['steps'] not in (512, 2048, 8192):
            raise ValueError('Supported budgets are 512, 2048, 8192')
        pair = (job['seed'], job['steps'])
        if pair in seen:
            raise ValueError('Duplicate job')
        seen.add(pair)


def training_config(base, job, epochs):
    if type(epochs) is not int or epochs not in (1, 4):
        raise ValueError('Supported epoch counts are 1 and 4')
    return base | {'seed': job['seed'], 'max_steps': job['steps'],
                   'checkpoint_interval': job['steps'], 'n_epochs': epochs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    jobs = spec['jobs']
    validate_jobs(jobs)
    epochs = spec.get('n_epochs', 1)
    training_config({}, jobs[0], epochs)  # Reject unsupported settings before any work.
    root = args.checkout.resolve()
    sys.path.insert(0, str(root / 'src'))
    import torch
    from hearthstone_ai.artifacts import compatibility_signature, content_hash, file_hash
    from hearthstone_ai.env import BgEnv
    from hearthstone_ai.training import load_model, train

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    signature = compatibility_signature()
    if signature['source_sha256'] != '80b5ca7a81687165553578ef28f581025d754a6759a04f42ae8d00387c5b97a8':
        raise ValueError('Requires unchanged snapshot-003')
    opponents = json.loads((root / 'configs/eval_opponents.json').read_text())
    base = json.loads((root / 'configs/smoke_train.json').read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(jobs=jobs, n_epochs=epochs, compatibility=signature, script_sha256=file_hash(Path(__file__)),
                    probe_sha256=file_hash(Path(__file__).with_name('selection_probe.py')),
                    torch_threads=torch.get_num_threads(),
                    torch_interop_threads=torch.get_num_interop_threads(),
                    opponent_set_sha256=content_hash(opponents),
                    evaluation_seeds=list(range(201, 221)), policy_seeds=[11, 22, 33],
                    semantics='Each budget starts fresh; no resumed RNG trajectory')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    summary = {}
    for job in jobs:
        label = f'seed-{job["seed"]}-steps-{job["steps"]}'
        run = args.output / label
        config = training_config(base, job, epochs)
        started = time.monotonic()
        status = train(config, run)
        train_seconds = time.monotonic() - started
        if status['stop_reason'] != 'steps' or status['num_timesteps'] != job['steps']:
            raise RuntimeError('Training did not finish the budget')
        expected = job['steps'] // config['n_steps'] * config['n_epochs']
        check_env = BgEnv()
        try:
            model = load_model(run / status['checkpoint'], check_env)
            if model._n_updates != expected:
                raise RuntimeError('Unexpected optimizer update count')
            updates = model._n_updates
        finally:
            check_env.close()
        entry = dict(train_seconds=train_seconds, updates=updates,
                     training_metrics=status['training_metrics'], evaluations={})
        for stage, checkpoint in [('initial', run / status['initial_checkpoint']),
                                  ('final', run / status['checkpoint'])]:
            digest = file_hash(checkpoint / 'model.zip')
            for policy_seed in (None, 11, 22, 33):
                env = BgEnv(opponents=opponents['opponents'], max_turns=8, max_actions=24)
                try:
                    model = load_model(checkpoint, env)
                    report = probe(model, env, list(range(201, 221)), policy_seed)
                finally:
                    env.close()
                if file_hash(checkpoint / 'model.zip') != digest:
                    raise RuntimeError('Checkpoint mutated during evaluation')
                report.update(model_sha256=digest, compatibility=signature,
                              opponent_set_sha256=manifest['opponent_set_sha256'])
                selection = 'argmax' if policy_seed is None else f'sample-{policy_seed}'
                name = f'{stage}-{selection}'
                (run / f'{name}.json').write_text(json.dumps(report, allow_nan=False), encoding='utf-8')
                entry['evaluations'][name] = selection_summary(report)
        summary[label] = entry
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps({label: entry}), flush=True)


if __name__ == '__main__':
    main()
