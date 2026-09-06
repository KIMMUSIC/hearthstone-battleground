"""Real Linux subprocess regression tests; no RL dependencies required."""

import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/supervise.py'
SPEC = importlib.util.spec_from_file_location('supervise', SCRIPT)
supervise = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supervise)


@unittest.skipUnless(sys.platform.startswith('linux'), 'requires real Linux processes')
class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_case(self, code, timeout=3, rss=256 * 1024 * 1024):
        marker = self.root / 'should-not-run'
        spec = {'timeout_seconds': timeout, 'rss_limit_bytes': rss, 'commands': [
            [sys.executable, '-c', code],
            [sys.executable, '-c', f'from pathlib import Path; Path({str(marker)!r}).touch()']]}
        result = supervise.run(spec, self.root / 'out')
        for row in result['commands']:
            self.assertFalse(row.get('remaining_pids'))
        return result, marker

    def test_success_and_no_overwrite(self):
        result, marker = self.run_case('print("ok")')
        self.assertEqual(result['status'], 'completed')
        self.assertTrue(marker.exists())
        with self.assertRaises(FileExistsError):
            supervise.run({'timeout_seconds': 1, 'rss_limit_bytes': 1,
                           'commands': [[sys.executable, '-c', 'pass']]}, self.root / 'out')

    def test_nonzero_stops_campaign(self):
        result, marker = self.run_case('raise SystemExit(7)')
        self.assertEqual(result['status'], 'command_failed')
        self.assertEqual(result['commands'][0]['returncode'], 7)
        self.assertFalse(marker.exists())

    def test_timeout_kills_term_ignoring_descendants(self):
        leaf = 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)'
        child = ('import subprocess,sys,signal,time; '
                 'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                 f'subprocess.Popen([sys.executable,"-c",{leaf!r}]); time.sleep(30)')
        code = ('import subprocess,sys,signal,time; '
                'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                f'subprocess.Popen([sys.executable,"-c",{child!r}]); time.sleep(30)')
        result, marker = self.run_case(code, timeout=1)
        self.assertEqual(result['status'], 'timeout')
        self.assertEqual(result['commands'][0]['returncode'], -signal.SIGKILL)
        self.assertFalse(supervise.group_usage(result['commands'][0]['pid'])[0])
        self.assertFalse(marker.exists())

    def test_rss_includes_grandchild(self):
        leaf = 'import time; a=bytearray(80*1024*1024); time.sleep(30)'
        child = f'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",{leaf!r}]); time.sleep(30)'
        code = f'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",{child!r}]); time.sleep(30)'
        result, marker = self.run_case(code, rss=64 * 1024 * 1024)
        self.assertEqual(result['status'], 'rss_exceeded')
        self.assertFalse(marker.exists())

    def test_parent_exit_with_live_child_is_failure(self):
        code = 'import subprocess,sys; subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"])'
        result, marker = self.run_case(code)
        self.assertEqual(result['status'], 'child_left_running')
        self.assertFalse(marker.exists())

    def test_cancel_cleans_child_and_returns_failure(self):
        spec = self.root / 'spec.json'
        out = self.root / 'out'
        ready = self.root / 'ready'
        code = f'from pathlib import Path; import time; Path({str(ready)!r}).touch(); time.sleep(30)'
        spec.write_text(json.dumps({'timeout_seconds': 5, 'rss_limit_bytes': 100000000,
                                   'commands': [[sys.executable, '-c', code]]}))
        proc = subprocess.Popen([sys.executable, str(SCRIPT), '--spec', str(spec),
                                 '--output', str(out)], stdout=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 3
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertTrue(ready.exists())
            proc.terminate()
            self.assertNotEqual(proc.wait(timeout=3), 0)
            result = json.loads((out / 'result.json').read_text())
            self.assertEqual(result['status'], 'interrupted')
            self.assertFalse(result['commands'][0]['remaining_pids'])
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


if __name__ == '__main__':
    unittest.main()
