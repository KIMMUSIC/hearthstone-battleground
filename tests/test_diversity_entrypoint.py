import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'run_diversity_009', Path(__file__).resolve().parents[1] / 'scripts/run_diversity_009.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def test_runtime_requires_declared_venv_entry_path(monkeypatch, tmp_path):
    venv = str(tmp_path / 'venv')
    base = str(tmp_path / 'base')
    python = str(tmp_path / 'venv' / 'bin' / 'python')
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr(sys, 'executable', python)
    monkeypatch.setattr(sys, 'prefix', venv)
    monkeypatch.setattr(sys, 'base_prefix', base)
    assert entry.require_runtime({'python': python}) == Path(python)
    with pytest.raises(RuntimeError, match='declared'):
        entry.require_runtime({'python': str(tmp_path / 'base' / 'bin' / 'python')})
    monkeypatch.setattr(sys, 'prefix', base)
    with pytest.raises(RuntimeError, match='declared'):
        entry.require_runtime({'python': python})


def test_frozen_input_mutation_rejected(tmp_path):
    source = tmp_path / 'source.py'
    source.write_text('original')
    (tmp_path / 'FILES.json').write_text(json.dumps({
        'source.py': hashlib.sha256(source.read_bytes()).hexdigest()}))
    entry.verify_files(tmp_path)
    source.write_text('changed')
    with pytest.raises(ValueError, match='Input hash mismatch'):
        entry.verify_files(tmp_path)


def test_manifest_cannot_escape_package(tmp_path):
    (tmp_path / 'FILES.json').write_text(json.dumps({'../outside.py': 'irrelevant'}))
    with pytest.raises(ValueError, match='Unsafe input path'):
        entry.verify_files(tmp_path)


@pytest.mark.parametrize('existing', ['return-train.zip', 'outputs/control-train', 'outputs/train'])
def test_any_prior_phase_evidence_blocks_before_execution(tmp_path, existing):
    prior = tmp_path / existing
    prior.parent.mkdir(parents=True, exist_ok=True)
    prior.write_bytes(b'original evidence')
    with pytest.raises(RuntimeError, match='already attempted'):
        entry.require_fresh_phase(tmp_path, 'train')
    assert prior.read_bytes() == b'original evidence'
