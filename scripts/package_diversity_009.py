"""Create a fresh immutable checkout bundle for the four supervised 009 phases."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[1]
    dest = args.destination.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    paths = []
    for pattern in ('src/hearthstone_ai/*.py', 'tests/*.py', 'tests/fixtures/*.json',
                    'data/*.json', 'configs/*.json'):
        paths.extend(checkout.glob(pattern))
    paths += [checkout / name for name in (
        'docs/RULES.md', 'pyproject.toml',
        'scripts/diversity_009.py', 'scripts/audit_diversity_009.py',
        'scripts/run_diversity_009.py', 'scripts/supervise.py', 'scripts/selection_probe.py',
        'scripts/bgai.py', 'scripts/bgai_threaded.py', 'scripts/learning_curve.py',
        'scripts/holdout_eval.py',
        'handoff/holdout-008/reference-opponents.json',
        'handoff/holdout-008/shifted-opponents.json',
        'handoff/snapshot-003/HearthStoneAI-Rebuild-003.zip',
        'handoff/holdout-008/grok-holdout-008.zip',
        'experiments/OPPONENT_DIVERSITY_009.md')]
    for path in paths:
        target = dest / path.relative_to(checkout)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    source = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((dest / 'src/hearthstone_ai').glob('*.py'))}
    signature = hashlib.sha256(json.dumps(source, sort_keys=True,
                                         separators=(',', ':')).encode()).hexdigest()
    phases = {}
    for phase in ('smoke', 'train', 'reference', 'shifted'):
        argv = ['{python}', '-B', '{root}/scripts/diversity_009.py',
                '--checkout', '{root}', '--output', '{root}/outputs',
                '--expected-source-sha256', signature, '--phase', phase]
        commands = [argv]
        if phase == 'smoke':
            commands.insert(0, ['{python}', '-B', '-m', 'pytest', '-q',
                               '-p', 'no:cacheprovider', '{root}/tests'])
        phases[phase] = dict(requires=[] if phase == 'smoke' else
                            ['smoke'] if phase == 'train' else ['train'], commands=commands)
    dump(dest / 'execution.json', dict(
        source_sha256=signature, phases=phases, minimum_free_bytes=2 * 1024**3,
        output_disk_limit_bytes=1024**3))
    entries = {p.relative_to(dest).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(dest.rglob('*')) if p.is_file()}
    dump(dest / 'FILES.json', entries)
    archive = dest.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(dest.rglob('*')):
            if path.is_file():
                z.write(path, path.relative_to(dest).as_posix())
    result = dict(archive=str(archive), sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                  source_sha256=signature, bytes=archive.stat().st_size, input_files=len(entries))
    dump(dest.with_suffix('.manifest.json'), result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
