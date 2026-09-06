"""Small diagnostic campaign; run only inside supervise.py from the checkout."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    launcher = [sys.executable, str(root / 'scripts/bgai_threaded.py')]

    def call(*argv):
        command = launcher + list(map(str, argv))
        print(json.dumps({'argv': command}), flush=True)
        subprocess.run(command, cwd=root, check=True)

    def status(directory):
        report = json.loads((directory / 'status.json').read_text(encoding='utf-8'))
        if report['status'] != 'completed' or report['stop_reason'] != 'steps':
            raise RuntimeError('Training did not finish the specified step budget')
        return report

    train64 = out / 'train64'
    train128 = out / 'train128'
    config = root / 'configs/smoke_train.json'
    call('train', '--config', config, '--output', train64)
    first = status(train64)
    if first['num_timesteps'] != 64:
        raise RuntimeError('Expected 64 steps')
    checkpoint64 = train64 / first['checkpoint']
    initial = train64 / 'checkpoint-000000000000'
    metadata = json.loads((initial / 'metadata.json').read_text(encoding='utf-8'))
    if metadata['num_timesteps'] != 0:
        raise RuntimeError('Expected an untrained checkpoint')
    call('train', '--config', config, '--resume', checkpoint64, '--output', train128)
    second = status(train128)
    if second['initial_steps'] != 64 or second['num_timesteps'] != 128:
        raise RuntimeError('Expected 64 to 128 resume')
    cases = [('untrained', initial), ('model64', checkpoint64),
             ('model128', train128 / second['checkpoint']),
             ('heuristic', None), ('random', None)]
    summary = {}
    for name, checkpoint in cases:
        result = out / f'{name}.json'
        argv = ['evaluate', '--policy', 'model' if checkpoint else name,
                '--seeds', *map(str, range(201, 221)), '--output', result, '--trace']
        if checkpoint:
            argv += ['--checkpoint', checkpoint]
        call(*argv)
        report = json.loads(result.read_text(encoding='utf-8'))
        summary[name] = {key: report[key] for key in
                         ('mean_reward', 'combat_win_rate', 'survival_rate', 'model')}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
