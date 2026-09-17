"""Bounded native GRU wiring smoke; disposable learning, never a competence eval."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import psutil
import torch

from isaac_rl.bridge import Bridge
from isaac_rl.env import IsaacEnv
from isaac_rl.evaluate import park_after_evaluation
from isaac_rl.ppo import as_tensor, load_policy, optimize
from isaac_rl.recurrent import PolicyMemory
from isaac_rl.storage import atomic_json, load_snapshot, sha256
from isaac_rl.timing import checkpoint_timing


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--native-pid',type=int,required=True)
    parser.add_argument('--native-created',type=float,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    native=psutil.Process(args.native_pid)
    assert abs(native.create_time()-args.native_created)<.01
    assert Path(native.exe()).resolve()==(root/'runtime/worker-03/game/isaac-ng.exe').resolve()
    mods=root/'runtime/worker-03/game/mods'
    assert {p.name for p in mods.iterdir() if p.is_dir() and not (p/'disable.it').exists()}=={'isaac_rl_bridge'}
    assert sha256(mods/'isaac_rl_bridge/main.lua')==sha256(root/'mod/main.lua')
    assert not any(c.laddr.port==10002 and c.status=='LISTEN' for c in psutil.net_connections(kind='tcp'))
    args.output.mkdir(parents=True,exist_ok=False)
    saved,digest,_=load_snapshot(args.checkpoint)
    torch.set_num_threads(2);torch.manual_seed(20260912);np.random.seed(20260912)
    model,adam=load_policy(saved,observation_profile='combat_gru_v4',with_optimizer=True)
    control=checkpoint_timing(saved);schedule=saved['training_schedule']
    before={k:v.clone() for k,v in model.state_dict().items()}
    atomic_json(args.output/'manifest.json',dict(purpose=__doc__,checkpoint_sha256=digest,
        native_pid=native.pid,native_created=native.create_time(),port=10002,
        requested_actions=64,scored=False,production_training_transitions=False))
    env=IsaacEnv(Bridge(port=10002,timeout=15,trace=args.output/'trace.jsonl'),frames=saved['frames'],
        max_steps=schedule['max_episode_steps'],idle_limit=schedule['idle_limit'],
        observation_profile='combat_gru_v4',reward_profile=saved['reward_profile'],timing_profile=control.profile)
    memory=PolicyMemory(model,1)
    rollout={k:[] for k in ('obs','actions','log_probs','values','rewards','next_values','terminated','ended','hidden_states','episode_starts')}
    timings=[];frames=[];completed=False
    try:
        obs,info=env.reset();initial=env.state
        for _ in range(64):
            assert env.observation_space.contains(obs)
            state=memory.rollout_state()
            with torch.no_grad():a,lp,_,v=memory.act(as_tensor(obs))
            start=time.perf_counter()
            following,reward,terminated,truncated,info=env.step(a[0].numpy())
            timings.append(time.perf_counter()-start);frames.append(info['frame_delta'])
            with torch.no_grad():nv=0. if terminated else memory.value(as_tensor(following)).item()
            row=dict(obs=obs,actions=a[0].numpy(),log_probs=lp.item(),values=v.item(),rewards=reward,
                next_values=nv,terminated=terminated,ended=terminated or truncated,**state)
            for k,value in row.items():rollout[k].append(value)
            obs=following;memory.reset_done([terminated or truncated])
            if terminated or truncated:obs,info=env.reset()
        final=env.state
        start=time.perf_counter()
        metrics=optimize(model,adam,rollout,batch_size=schedule['batch_size'],
            entropy_coef=saved['entropy_coef'],gamma=control.gamma,lam=control.lam)
        update_s=time.perf_counter()-start
        assert any(not torch.equal(before[k],v) for k,v in model.state_dict().items() if k.startswith('gru.'))
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert sha256(args.checkpoint)==digest
        result=dict(passed=True,actions=64,scored=False,production_training_transitions=False,
            checkpoint_unchanged=True,grads_into_gru=True,all_weights_finite=True,
            median_native_step_s=float(np.median(timings)),native_frame_deltas=frames,
            update_s=update_s,metrics=metrics,initial=initial,final=final)
        atomic_json(args.output/'result.json',result);completed=True
        print(json.dumps({k:v for k,v in result.items() if k not in ('initial','final','native_frame_deltas')}),flush=True)
    except Exception:
        atomic_json(args.output/'failure-state.json',env.state)
        raise
    finally:
        if completed:park_after_evaluation(env,args.output)
        env.close()


if __name__=='__main__':main()
