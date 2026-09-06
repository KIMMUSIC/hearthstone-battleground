"""Execute one frozen 009 phase in its declared Linux venv; preserve all outputs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile


def verify_files(root):
    entries = json.loads((root / 'FILES.json').read_text())
    for name, digest in entries.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or (root / name).is_symlink():
            raise ValueError(f'Unsafe input path: {name}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Input hash mismatch: {name}')


def size(folder):
    total = 0
    for path in folder.rglob('*'):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            # Atomic checkpoint commits can rename temporary files between reads.
            continue
    return total


def require_fresh_phase(root, phase):
    data_phase = f'eval-{phase}' if phase in ('reference', 'shifted') else phase
    paths = [root / 'outputs' / ('control-' + phase), root / 'outputs' / data_phase,
             root / f'return-{phase}.zip']
    if any(path.exists() for path in paths):
        raise RuntimeError('Phase already attempted; preserve it, do not restart')


def require_runtime(plan):
    python = Path(plan['python'])
    # Keep the venv entry path: resolving symlinks loses its package environment.
    if (sys.platform != 'linux' or not python.is_absolute()
            or Path(sys.executable).absolute() != python
            or sys.prefix == sys.base_prefix):
        raise RuntimeError('Requires the declared Linux virtual environment')
    return python


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['smoke', 'train', 'reference', 'shifted'])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    verify_files(root)
    plan = json.loads((root / 'execution.json').read_text())
    python = require_runtime(plan)
    phase = plan['phases'][args.phase]
    output = root / 'outputs'
    report_dir = output / ('control-' + args.phase)
    archive = root / f'return-{args.phase}.zip'
    require_fresh_phase(root, args.phase)
    for dependency in phase['requires']:
        result = json.loads((output / ('control-' + dependency) / 'result.json').read_text())
        if result['status'] != 'completed':
            raise RuntimeError(f'Incomplete predecessor: {dependency}')
    if shutil.disk_usage(root).free < plan['minimum_free_bytes']:
        raise RuntimeError('Insufficient free disk; never clean old results')
    report_dir.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update({key: '1' for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                                    'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                                    'PYTHONDONTWRITEBYTECODE')})
    env['PYTHONPATH'] = str(root / 'src')
    commands = [[arg.replace('{root}', str(root)).replace('{python}', str(python))
                 for arg in argv] for argv in phase['commands']]
    spec = dict(timeout_seconds=180, rss_limit_bytes=2147483648, commands=commands)
    spec_path = report_dir / 'supervisor-spec.json'
    spec_path.write_text(json.dumps(spec, indent=2))
    argv = [str(python), '-B', str(root / 'scripts/supervise.py'), '--spec',
            str(spec_path), '--output', str(report_dir / 'supervisor')]
    result = dict(phase=args.phase, started_unix=time.time(), status='running', argv=argv,
                  disk_limit_bytes=plan['output_disk_limit_bytes'],
                  disk_measurement='sampled package outputs, plus returned ZIP; not whole filesystem')
    child = None
    try:
        with (report_dir / 'entry.log').open('wb') as log:
            child = subprocess.Popen(argv, cwd=root, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            result['supervisor_pid'] = child.pid
            (report_dir / 'live.json').write_text(json.dumps(result, indent=2))
            peak = 0
            while child.poll() is None:
                peak = max(peak, size(output))
                previous_archives = sum(p.stat().st_size for p in root.glob('return-*.zip'))
                if peak * 2 + previous_archives > plan['output_disk_limit_bytes']:
                    result['status'] = 'disk_exceeded'
                    # Only this phase supervisor is signalled; it cleans its own child groups.
                    child.send_signal(signal.SIGTERM)
                    break
                time.sleep(0.5)
            result['returncode'] = child.wait(timeout=10)
            result['peak_sampled_output_bytes'] = max(peak, size(output))
        supervision = json.loads((report_dir / 'supervisor/result.json').read_text())
        if result['status'] == 'running':
            result['status'] = supervision['status']
        verify_files(root)
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            child.wait(timeout=10)
        raise
    finally:
        result['ended_unix'] = time.time()
        (report_dir / 'result.json').write_text(json.dumps(result, indent=2))
        # No deletion, no overwrite; include fixed inputs and available phase evidence.
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
            for name in json.loads((root / 'FILES.json').read_text()):
                z.write(root / name, name)
            z.write(root / 'FILES.json', 'FILES.json')
            for path in sorted(output.rglob('*')):
                if path.is_file():
                    z.write(path, path.relative_to(root).as_posix())
        print(json.dumps(dict(result=result, archive=str(archive),
                              archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())),
              flush=True)
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
