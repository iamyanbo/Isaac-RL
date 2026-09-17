"""Archive and verify a live GRU fork without changing learner/game state."""
import argparse
import json
from pathlib import Path
import time

import psutil
import torch

from isaac_rl.lifecycle import identity, process_liveness
from isaac_rl.observation import observation_manifest
from isaac_rl.ppo import load_policy
from isaac_rl.storage import atomic_json, load_snapshot, source_fingerprint
from scripts.verify_reward_handover import equal


def verify(parent, saved, config, parent_config, initial=None):
    assert parent['observation_profile']=='combat_history_v3' and parent['architecture']==2
    assert saved['observation_profile']=='combat_gru_v4' and saved['architecture']==3
    assert saved['observation_layout']==observation_manifest('combat_gru_v4')
    exact=['reward_profile','reward','frames','timing_profile','entropy_coef',
           'control_timing','training_schedule','ports','native_frames_origin_steps']
    constants=['steps','ports','rollout','batch_size','frames','timing_profile',
               'max_episode_steps','idle_limit','seed','device','entropy_coef',
               'algorithm','mode','control_timing','training_schedule','reward_profile','reward']
    for key in exact: assert equal(saved[key],parent[key]),key
    for key in constants: assert equal(config[key],parent_config[key]),key
    assert len(saved['ports'])==6
    assert set(parent['training_seeds']).issubset(saved['training_seeds'])
    if initial is None:
        # Same construction seed/order as the real learner, BEFORE RNG restore.
        torch.set_num_threads(2);torch.manual_seed(config['seed'])
        model,optimizer=load_policy(parent,observation_profile='combat_gru_v4',with_optimizer=True)
        assert equal(saved['model'],model.state_dict()),'model projection or new GRU initialization'
        assert equal(saved['optimizer'],optimizer.state_dict()),'Adam projection'
        exact+=['steps','updates','episodes','native_frames','torch_rng','numpy_rng','cuda_rng']
        for key in exact: assert equal(saved[key],parent[key]),key
        assert saved['recent']==[] and saved['losses']=={}
        assert not saved['model']['memory_readout.weight'].any()
        changed=[]
    else:
        assert saved['steps']>parent['steps'] and saved['updates']>parent['updates']
        assert saved['native_frames']>parent['native_frames']
        assert equal(saved['optimizer']['param_groups'],initial['optimizer']['param_groups'])
        changed=[key for key,value in saved['model'].items() if not torch.equal(value,initial['model'][key])]
        assert any(k.startswith('gru.') for k in changed),'No actual GRU weight update'
        assert 'memory_readout.weight' in changed
        for key in ('reward_profile','reward','observation_layout','control_timing','training_schedule','entropy_coef'):
            assert equal(saved[key],initial[key]),key
    assert all(torch.isfinite(t).all() for t in saved['model'].values())
    return dict(exact_parent_fields=exact,unchanged_config_fields=constants,
        declared_input_and_adam_projection_verified=initial is None,
        changed_parameter_tensors=changed,all_weights_finite=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parent',type=Path)
    parser.add_argument('run',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--phase',choices=['initial','optimized'],required=True)
    args=parser.parse_args()
    parent,parent_hash,_=load_snapshot(args.parent)
    deadline=time.monotonic()+50
    while True:
        try:
            saved,digest,payload=load_snapshot(args.run/'latest.pt')
            status=json.loads((args.run/'status.json').read_text())
            assert process_liveness(identity(status))=='alive',status
            if args.phase=='initial' or saved['updates']>parent['updates']: break
        except FileNotFoundError: pass
        if time.monotonic()>=deadline: raise TimeoutError('Requested checkpoint not available; no process action taken')
        time.sleep(.25)
    # Archive the very bytes verified. Never overwrite a previous activation.
    with (args.output/f'{args.phase}.pt').open('xb') as stream:stream.write(payload)
    config=json.loads((args.run/'config.json').read_text())
    old=json.loads((args.output/'parent-final-config.json').read_text())
    initial=load_snapshot(args.output/'initial.pt')[0] if args.phase=='optimized' else None
    checks=verify(parent,saved,config,old,initial)
    assert saved['origin']['parent_sha256']==parent_hash
    assert saved['origin']['parent_steps']==parent['steps']
    assert saved['source_hashes']==source_fingerprint(Path(__file__).resolve().parents[1])
    sockets=[c.laddr.port for c in psutil.net_connections(kind='tcp')
             if c.pid==status['pid'] and c.status==psutil.CONN_ESTABLISHED and c.laddr.port in saved['ports']]
    assert sorted(sockets)==sorted(saved['ports'])
    result=dict(passed=True,phase=args.phase,time=time.time(),pid=status['pid'],
        process_started=status['process_started'],session_id=status['session_id'],
        parent_sha256=parent_hash,checkpoint_sha256=digest,steps=saved['steps'],updates=saved['updates'],
        native_frames=saved['native_frames'],connected_ports=sorted(sockets),**checks)
    atomic_json(args.output/f'{args.phase}-verification.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
