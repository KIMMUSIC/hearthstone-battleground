import importlib.util
from pathlib import Path
import sys

import pytest

scripts = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('holdout_eval', scripts / 'holdout_eval.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixed_spec():
    return {'seeds': list(range(301, 321)), 'cases': [
        {'epochs': e, 'seed': s} for e in (1, 4) for s in (7, 17, 27)]}


def test_all_candidates_and_fresh_seeds():
    module.validate_spec(fixed_spec())


@pytest.mark.parametrize('change', ['old_seeds', 'missing', 'duplicate'])
def test_rejects_contaminated_or_selected_holdout(change):
    value = fixed_spec()
    if change == 'old_seeds':
        value['seeds'] = list(range(201, 221))
    elif change == 'missing':
        value['cases'].pop()
    else:
        value['cases'][-1] = value['cases'][0]
    with pytest.raises(ValueError):
        module.validate_spec(value)
