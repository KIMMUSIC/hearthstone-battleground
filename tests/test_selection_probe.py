import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
import torch
import pytest

from hearthstone_ai.env import BgEnv

spec = importlib.util.spec_from_file_location(
    'selection_probe', Path(__file__).resolve().parents[1] / 'scripts/selection_probe.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_sampling_is_reproducible_independent_and_masked():
    env = BgEnv(max_turns=2, max_actions=8)
    try:
        model = MaskablePPO('MultiInputPolicy', env, seed=7, device='cpu',
                            n_steps=8, batch_size=8, policy_kwargs={'net_arch': [32, 32]})
        first = module.probe(model, env, [201, 202], 11)
        torch.rand(50)
        assert first == module.probe(model, env, [201, 202], 11)
        reordered = module.probe(model, env, [202, 201], 11)
        assert first['trace'] == list(reversed(reordered['trace']))
        different = module.probe(model, env, [201, 202], 22)
        assert first['trace'] != different['trace']
        for episode in first['trace']:
            for step in episode['steps']:
                assert step['action_mask'][step['action']]
                assert np.isclose(sum(step['probabilities']), 1)
                assert step['entropy_nats'] >= 0
    finally:
        env.close()


def test_argmax_matches_predict():
    env = BgEnv(max_turns=2, max_actions=8)
    try:
        model = MaskablePPO('MultiInputPolicy', env, seed=7, device='cpu',
                            n_steps=8, batch_size=8, policy_kwargs={'net_arch': [32, 32]})
        report = module.probe(model, env, [201])
        obs, _ = env.reset(seed=201)
        for step in report['trace'][0]['steps']:
            action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
            assert int(action) == step['action']
            obs, _, _, _, _ = env.step(int(action))
    finally:
        env.close()


def test_preflight_resolves_relocated_checkpoints_and_rejects_hash_change(tmp_path):
    for name in ('untrained', 'model64', 'model128'):
        folder = 'train128' if name == 'model128' else 'train64'
        checkpoint = tmp_path / folder / name / 'model.zip'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(name.encode())
        (tmp_path / f'{name}.json').write_text(json.dumps({'model': {
            'path': f'/old/host/{folder}/{name}/model.zip',
            'sha256': hashlib.sha256(name.encode()).hexdigest()}}))
    cases = module.preflight(tmp_path)
    assert all(checkpoint.is_relative_to(tmp_path) for _, _, checkpoint, _ in cases)
    cases[0][2].write_bytes(b'changed')
    with pytest.raises(ValueError, match='Checkpoint changed'):
        module.preflight(tmp_path)
