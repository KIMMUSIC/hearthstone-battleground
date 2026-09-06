"""Replay existing 009 final-policy actions without model inference or training."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.snapshot.resolve()
    sys.path.insert(0, str(root / 'src'))
    from hearthstone_ai.actions import decode_action
    from hearthstone_ai.env import BgEnv

    report = {'episodes_replayed': 0, 'steps_replayed': 0, 'errors': [], 'cases': []}
    for cohort in ('reference', 'shifted'):
        opponents = root / f'handoff/holdout-008/{cohort}-opponents.json'
        for path in sorted((root / f'outputs/eval-{cohort}').glob('*-final-*.json')):
            data = json.loads(path.read_text(encoding='utf-8'))
            counts = Counter()
            for trace, expected in zip(data['trace'], data['episodes'], strict=True):
                env = BgEnv(opponents=json.loads(opponents.read_text())['opponents'],
                            max_turns=8, max_actions=24)
                env.reset(seed=trace['seed'])
                reward_sum = 0.0
                done = False
                for index, step in enumerate(trace['steps']):
                    assert not done, (path.name, index, 'action after terminal')
                    for key in ('turn', 'gold', 'frozen', 'actions_remaining'):
                        assert getattr(env.game, key) == step[key], (path.name, index, key)
                    assert env.action_masks().tolist() == step['action_mask']
                    counts[decode_action(step['action']).kind] += 1
                    _, reward, terminated, truncated, info = env.step(step['action'])
                    done = terminated or truncated
                    assert reward == step['reward']
                    assert bool(info.get('forced_end_turn', False)) == step['forced_end_turn']
                    reward_sum += reward
                    report['steps_replayed'] += 1
                assert done and reward_sum == expected['reward']
                assert env.game.hp == expected['hp']
                env.close()
                report['episodes_replayed'] += 1
            report['cases'].append({'report': path.name,
                                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                    'action_counts': dict(counts)})
    report['status'] = 'passed'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}))


if __name__ == '__main__':
    main()
