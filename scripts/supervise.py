"""Linux process-group campaign supervisor; sampled RSS, not a hard memory limit."""

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def group_usage(pgid):
    """Include live group members and grandchildren; group/session escape is unsupported."""
    members = []
    rss = 0
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == pgid and fields[0] != 'Z':
                members.append(int(path.parent.name))
                rss += int(fields[21]) * os.sysconf('SC_PAGE_SIZE')
        except (FileNotFoundError, ProcessLookupError):
            continue
    return members, rss


def stop_group(pgid, grace=0.3):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if not group_usage(pgid)[0]:
                return []
            time.sleep(0.02)
    return group_usage(pgid)[0]


def validate(spec):
    for key in ('timeout_seconds', 'rss_limit_bytes'):
        value = spec.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'{key} must be positive and finite')
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be positive and finite')
    commands = spec.get('commands')
    if not isinstance(commands, list) or not commands:
        raise ValueError('commands must be a nonempty list of argv lists')
    for argv in commands:
        if not isinstance(argv, list) or not argv:
            raise ValueError('each command must be an argv list')
        if any(not isinstance(arg, str) or '\0' in arg for arg in argv) or not argv[0]:
            raise ValueError('invalid argv')


def run(spec, output):
    if not sys.platform.startswith('linux'):
        raise RuntimeError('This supervisor requires Linux /proc and process groups')
    validate(spec)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'spec.json').write_text(json.dumps(spec, indent=2), encoding='utf-8')
    started = time.monotonic()
    result = {'started_unix': time.time(), 'commands': [], 'status': 'running',
              'memory_enforcement': 'sampled process-group RSS; not a hard limit'}
    active = None
    old_handlers = {}

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.signal(sig, interrupted)
        env = os.environ.copy()
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS'):
            env[name] = '1'
        for index, argv in enumerate(spec['commands']):
            if time.monotonic() - started >= spec['timeout_seconds']:
                result['status'] = 'timeout'
                break
            row = {'argv': argv, 'started_unix': time.time(), 'peak_sampled_rss': 0}
            result['commands'].append(row)
            with (output / f'{index:03d}.log').open('wb') as log:
                active = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                          stdin=subprocess.DEVNULL, env=env,
                                          start_new_session=True)
                row['pid'] = active.pid
                while True:
                    members, rss = group_usage(active.pid)
                    row['peak_sampled_rss'] = max(row['peak_sampled_rss'], rss)
                    code = active.poll()
                    if time.monotonic() - started >= spec['timeout_seconds']:
                        result['status'] = 'timeout'
                        break
                    if rss > spec['rss_limit_bytes']:
                        result['status'] = 'rss_exceeded'
                        break
                    if code is not None:
                        result['status'] = ('child_left_running' if members else
                                            'completed' if code == 0 else 'command_failed')
                        break
                    time.sleep(0.02)
                row['remaining_pids'] = stop_group(active.pid)
                row['returncode'] = active.wait(timeout=2)
                row['ended_unix'] = time.time()
                active = None
                if row['remaining_pids']:
                    result['status'] = 'cleanup_failed'
                if result['status'] != 'completed':
                    break
    except KeyboardInterrupt as error:
        result.update(status='interrupted', error=str(error))
    except Exception as error:
        result.update(status='supervisor_error', error=f'{type(error).__name__}: {error}')
    finally:
        # Repeated cancellation must not interrupt cleanup.
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        if active is not None:
            row = result['commands'][-1]
            row['remaining_pids'] = stop_group(active.pid)
            try:
                row['returncode'] = active.wait(timeout=2)
            except subprocess.TimeoutExpired:
                result['status'] = 'cleanup_failed'
            if row['remaining_pids']:
                result['status'] = 'cleanup_failed'
        result['elapsed_seconds'] = time.monotonic() - started
        result['ended_unix'] = time.time()
        (output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(json.loads(args.spec.read_text(encoding='utf-8')), args.output)
    print(json.dumps(report))
    sys.exit(0 if report['status'] == 'completed' else 1)
