import importlib.util
from pathlib import Path
import sys

import pytest

scripts = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('learning_curve', scripts / 'learning_curve.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('jobs', [[], [{'seed': 7, 'steps': 513}],
                                [{'seed': True, 'steps': 512}],
                                [{'seed': 7, 'steps': 512}] * 2])
def test_rejects_invalid_or_duplicate_budgets(jobs):
    with pytest.raises(ValueError):
        module.validate_jobs(jobs)


def test_accepts_planned_curve():
    module.validate_jobs([{'seed': seed, 'steps': steps}
                          for seed in (7, 17, 27) for steps in (512, 2048, 8192)])


def test_epoch_comparison_changes_only_epochs():
    base = {'n_steps': 32, 'batch_size': 32, 'n_epochs': 1, 'max_seconds': 60}
    job = {'seed': 7, 'steps': 8192}
    baseline = module.training_config(base, job, 1)
    candidate = module.training_config(base, job, 4)
    assert candidate == baseline | {'n_epochs': 4}
    assert base['n_epochs'] == 1
    assert candidate['max_steps'] // candidate['n_steps'] * candidate['n_epochs'] == 1024


@pytest.mark.parametrize('epochs', [True, 0, 2, 4.0, '4'])
def test_rejects_unsupported_epoch_counts(epochs):
    with pytest.raises(ValueError):
        module.training_config({}, {'seed': 7, 'steps': 8192}, epochs)
