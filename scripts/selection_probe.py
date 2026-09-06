"""Compare selection modes without changing checkpoint-compatible training source."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


def episode_seed(policy_seed, environment_seed):
    value = f'selection-probe-v1:{policy_seed}:{environment_seed}'.encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], 'big') % (2**63)


def preflight(previous):
    cases = []
    for name in ('untrained', 'model64', 'model128'):
        reference = json.loads((previous / f'{name}.json').read_text())
        folder = 'train128' if name == 'model128' else 'train64'
        checkpoint = (previous / folder / Path(reference['model']['path']).parent.name
                      / 'model.zip').resolve()
        if not checkpoint.is_relative_to(previous.resolve()):
            raise ValueError('Checkpoint escaped the intended run root')
        before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if before != reference['model']['sha256']:
            raise ValueError('Checkpoint changed since prior evaluation')
        cases.append((name, reference, checkpoint, before))
    return cases


def selection_summary(report):
    steps = [s for t in report['trace'] for s in t['steps']]
    result = {k: report[k] for k in ('mean_reward', 'combat_win_rate', 'survival_rate')}
    result.update(
        freeze_fraction=sum(s['action'] == 3 for s in steps) / len(steps),
        end_fraction=sum(s['action'] == 0 for s in steps) / len(steps),
        mean_freeze_probability=sum(s['probabilities'][3] for s in steps) / len(steps),
        mean_end_probability=sum(s['probabilities'][0] for s in steps) / len(steps),
        forced_turns=sum(e['forced_end_turns'] for e in report['episodes']))
    for key in ('entropy_nats', 'top_probability', 'top_margin'):
        result['mean_' + key] = sum(s[key] for s in steps) / len(steps)
    return result


def probe(model, env, seeds, policy_seed=None):
    import torch

    model.policy.set_training_mode(False)
    episodes = []
    traces = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        generator = torch.Generator(device='cpu')
        derived = episode_seed(policy_seed, seed) if policy_seed is not None else None
        generator.manual_seed(derived if derived is not None else 0)
        row = dict(seed=seed, reward=0.0, wins=0, draws=0, losses=0, damage_taken=0,
                   damage_dealt=0, forced_end_turns=0, rerolls=0, swaps=0, actions=0)
        steps = []
        for _ in range(env.max_turns * (env.max_actions + 2)):
            mask = env.action_masks()
            with torch.no_grad():
                tensor, _ = model.policy.obs_to_tensor(obs)
                distribution = model.policy.get_distribution(tensor, action_masks=mask)
                probabilities = distribution.distribution.probs[0].cpu()
                if not torch.isfinite(probabilities).all():
                    raise ValueError('Nonfinite probabilities')
                if policy_seed is None:
                    action = int(probabilities.argmax())
                else:
                    action = int(torch.multinomial(probabilities, 1, generator=generator))
                entropy = float(distribution.entropy()[0])
                ordered = torch.sort(probabilities, descending=True).values
            if not mask[action] or any(float(p) != 0 for p, legal in zip(
                    probabilities, mask, strict=True) if not legal):
                raise ValueError('Policy distribution violated action mask')
            step = dict(turn=env.game.turn, action=action, gold=env.game.gold,
                        frozen=env.game.frozen, actions_remaining=env.game.actions_remaining,
                        action_mask=mask.tolist(), probabilities=probabilities.tolist(),
                        entropy_nats=entropy, top_probability=float(ordered[0]),
                        top_margin=float(ordered[0] - ordered[1]))
            obs, reward, terminated, truncated, info = env.step(action)
            step.update(reward=reward, forced_end_turn=bool(info.get('forced_end_turn', False)))
            steps.append(step)
            row['actions'] += 1
            row['reward'] += reward
            row['rerolls'] += action == 1
            row['swaps'] += 28 <= action < 34
            if 'combat_result' in info:
                row[{1: 'wins', 0: 'draws', -1: 'losses'}[info['combat_result']]] += 1
                for key in ('damage_taken', 'damage_dealt'):
                    row[key] += info[key]
                row['forced_end_turns'] += bool(info['forced_end_turn'])
            if terminated or truncated:
                row.update(hp=env.game.hp, last_turn=env.game.turn, survived=env.game.hp > 0,
                           termination_reason=env.game.termination_reason)
                episodes.append(row)
                traces.append(dict(seed=seed, derived_policy_seed=derived, steps=steps))
                break
        else:
            raise RuntimeError('Episode exceeded action bound')
    combat_count = sum(e['wins'] + e['draws'] + e['losses'] for e in episodes)
    return dict(episodes=episodes, trace=traces, episode_count=len(episodes),
                mean_reward=sum(e['reward'] for e in episodes) / len(episodes),
                combat_count=combat_count,
                combat_win_rate=sum(e['wins'] for e in episodes) / combat_count,
                survival_rate=sum(e['survived'] for e in episodes) / len(episodes),
                policy_seed=policy_seed, seeds=list(seeds),
                selection='argmax' if policy_seed is None else 'categorical-local-generator')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.checkout.resolve()
    sys.path.insert(0, str(root / 'src'))
    import torch
    from hearthstone_ai.artifacts import compatibility_signature, content_hash, file_hash
    from hearthstone_ai.env import BgEnv
    from hearthstone_ai.training import load_model

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    signature = compatibility_signature()
    if signature['source_sha256'] != '80b5ca7a81687165553578ef28f581025d754a6759a04f42ae8d00387c5b97a8':
        raise ValueError('This experiment requires the unchanged snapshot-003 source')
    previous = root / 'runs/diagnostic-003'
    cases = preflight(previous)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(compatibility=signature, script_sha256=file_hash(Path(__file__)),
                    torch_threads=torch.get_num_threads(),
                    torch_interop_threads=torch.get_num_interop_threads(),
                    environment_seeds=list(range(201, 221)), policy_seeds=[11, 22, 33],
                    additional_training_steps=0, checkpoint_hashes={})
    opponents = json.loads((root / 'configs/eval_opponents.json').read_text())
    manifest['opponent_set_sha256'] = content_hash(opponents)
    summary = {}
    for name, reference, checkpoint, before in cases:
        manifest['checkpoint_hashes'][name] = before
        for policy_seed in (None, 11, 22, 33):
            env = BgEnv(opponents=opponents['opponents'], max_turns=8, max_actions=24)
            try:
                model = load_model(checkpoint, env)
                report = probe(model, env, list(range(201, 221)), policy_seed)
            finally:
                env.close()
            if policy_seed is None and report['episodes'] != reference['episodes']:
                raise ValueError('Argmax regression against original evaluation')
            if file_hash(checkpoint) != before:
                raise ValueError('Evaluation changed checkpoint')
            report.update(model_sha256=before, compatibility=signature,
                          opponent_set_sha256=manifest['opponent_set_sha256'])
            label = f'{name}-' + ('argmax' if policy_seed is None else f'sample-{policy_seed}')
            (args.output / f'{label}.json').write_text(
                json.dumps(report, allow_nan=False), encoding='utf-8')
            summary[label] = selection_summary(report)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
