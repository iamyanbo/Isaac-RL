"""Read-only learner verification and immutable evidence; never controls games."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import psutil
import torch

from isaac_rl.lifecycle import identity, process_liveness
from isaac_rl.storage import atomic_json, load_snapshot, source_fingerprint


def equal(a, b):
    if torch.is_tensor(a): return torch.equal(a, b)
    if isinstance(a, np.ndarray): return np.array_equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)): return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parent', type=Path)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--phase', choices=['initial', 'optimized'], required=True)
    args = parser.parse_args()
    parent, parent_hash, _ = load_snapshot(args.parent)
    deadline = time.monotonic() + 50
    while True:
        try:
            saved, digest, payload = load_snapshot(args.run/'latest.pt')
            status = json.loads((args.run/'status.json').read_text())
            assert process_liveness(identity(status)) == 'alive', status
            if args.phase == 'initial' or saved['updates'] > parent['updates']: break
        except FileNotFoundError:
            pass
        if time.monotonic() >= deadline: raise TimeoutError('Requested checkpoint not available; no process action taken')
        time.sleep(.25)
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = args.output/f'{args.phase}.pt'
    with frozen.open('xb') as stream: stream.write(payload)
    exact_keys = ['frames', 'timing_profile', 'entropy_coef', 'training_schedule',
                  'control_timing', 'observation_profile', 'observation_layout',
                  'architecture', 'native_frames_origin_steps', 'ports']
    if args.phase == 'initial':
        exact_keys += ['model', 'optimizer', 'torch_rng', 'numpy_rng', 'cuda_rng',
                       'steps', 'updates', 'episodes', 'native_frames']
        assert saved['recent'] == [] and saved['losses'] == {}
    for key in exact_keys: assert equal(saved[key], parent[key]), key
    assert set(parent['training_seeds']).issubset(saved['training_seeds'])
    assert parent['reward_profile'] == 'completion_v4'
    assert saved['reward_profile'] == 'native_clear_v5'
    for key, value in parent['reward'].items():
        if key not in ('name', 'require_uncleared_combat'): assert equal(value, saved['reward'][key]), key
    assert saved['reward']['require_uncleared_combat'] is True
    assert saved['optimizer']['param_groups'] == parent['optimizer']['param_groups']
    assert saved['origin']['parent_sha256'] == parent_hash
    assert saved['origin']['parent_steps'] == parent['steps']
    assert saved['source_hashes'] == source_fingerprint(Path(__file__).resolve().parents[1])
    assert len(saved['ports']) == 6
    assert all(torch.isfinite(t).all() for t in saved['model'].values())
    changed = [k for k, t in saved['model'].items() if not torch.equal(t, parent['model'][k])]
    if args.phase == 'optimized':
        assert changed and saved['steps'] > parent['steps']
        assert saved['native_frames'] > parent['native_frames']
    config = json.loads((args.run/'config.json').read_text())
    old_config = json.loads((args.output/'config.json').read_text())
    config_constants = ['steps', 'ports', 'rollout', 'batch_size', 'frames', 'timing_profile',
                        'max_episode_steps', 'idle_limit', 'seed', 'device', 'entropy_coef',
                        'observation_profile', 'architecture', 'observation_layout',
                        'algorithm', 'mode', 'control_timing', 'training_schedule']
    for key in config_constants: assert equal(config[key], old_config[key]), key
    sockets = [c.laddr.port for c in psutil.net_connections(kind='tcp')
               if c.pid == status['pid'] and c.status == psutil.CONN_ESTABLISHED
               and c.laddr.port in saved['ports']]
    assert sorted(sockets) == sorted(saved['ports']), sockets
    report = dict(passed=True, phase=args.phase, time=time.time(), pid=status['pid'],
        process_started=status['process_started'], session_id=status['session_id'],
        parent_sha256=parent_hash, checkpoint_sha256=digest, steps=saved['steps'],
        updates=saved['updates'], native_frames=saved['native_frames'],
        exact_inheritance=exact_keys, config_constants=config_constants,
        connected_ports=sorted(sockets), changed_parameter_tensors=changed,
        reward_change='require_uncleared_combat false -> true', all_weights_finite=True)
    atomic_json(args.output/f'{args.phase}-verification.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
