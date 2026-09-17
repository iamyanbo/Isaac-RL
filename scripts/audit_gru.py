"""Disposable GRU migration/cost audit; no game connections or production updates."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from isaac_rl.history import ObservationCurrent, ObservationHistory
from isaac_rl.ppo import as_tensor, load_policy, optimize
from isaac_rl.recurrent import PolicyMemory, project_history_input
from isaac_rl.storage import atomic_json, load_snapshot, sha256, source_fingerprint
from isaac_rl.timing import checkpoint_timing


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('states',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True,exist_ok=False)
    saved,digest,payload=load_snapshot(args.checkpoint)
    with (args.output/'preparation-baseline.pt').open('xb') as f:f.write(payload)
    states=json.loads(args.states.read_text())
    atomic_json(args.output/'native-states.json',states)
    atomic_json(args.output/'preparation.json',dict(checkpoint_sha256=digest,steps=saved['steps'],updates=saved['updates'],
        activation_parent=False,source_hashes=source_fingerprint(Path(__file__).resolve().parents[1]),
        purpose='migration correctness and synthetic repetition cost only; not gameplay competence'))
    torch.manual_seed(20260912)
    model,adam=load_policy(saved,observation_profile='combat_gru_v4',with_optimizer=True)
    initial=deepcopy(model.state_dict())
    for name,value in saved['model'].items():
        assert torch.equal(initial[name],project_history_input(value,name,initial[name])),name
    control=checkpoint_timing(saved)
    history,current=ObservationHistory(),ObservationCurrent()
    rows=[];input_bytes=[];prefix_error=0.
    for i,s in enumerate(states):
        action=None if i==0 else [0,2,0]
        old=history.append(s,Counter(),control.scale,action)
        new=current.append(s,Counter(),control.scale,action)
        prefix_error=max(prefix_error,float(np.max(np.abs(old['vector'][:2132]-new['vector'][:2132]))))
        assert np.array_equal(old['grid'][:10],new['grid'])
        rows.append(new)
    assert prefix_error==0
    # Repeat native input patterns solely to measure six-env/full-rollout cost.
    # These are NOT native transitions, evaluated policies, or saved learning.
    length=saved['training_schedule']['rollout'];count=len(saved['ports'])
    rollout={k:[] for k in ('obs','actions','log_probs','values','rewards','next_values','terminated','ended','hidden_states','episode_starts')}
    memory=PolicyMemory(model,count);inference=[]
    for i in range(length):
        obs={k:np.stack([rows[(i+e)%len(rows)][k] for e in range(count)]) for k in rows[0]}
        memory_state=memory.rollout_state()
        start=time.perf_counter()
        with torch.no_grad():
            tensors={k:torch.as_tensor(v) for k,v in obs.items()}
            a,lp,_,v=memory.act(tensors);nv=memory.value(tensors)
        inference.append(time.perf_counter()-start)
        done=np.zeros(count,dtype=bool)
        row=dict(obs=obs,actions=a.numpy(),log_probs=lp.numpy(),values=v.numpy(),next_values=nv.numpy(),
            rewards=np.full(count,-.005,np.float32),terminated=done,ended=done,**memory_state)
        for k,v in row.items():rollout[k].append(v)
        memory.reset_done(done)
    times=[];results=[]
    for _ in range(2):
        torch.manual_seed(20260912)
        probe,probe_adam=load_policy(saved,observation_profile='combat_gru_v4',with_optimizer=True)
        np.random.seed(20260912)
        start=time.perf_counter()
        result=optimize(probe,probe_adam,rollout,batch_size=saved['training_schedule']['batch_size'],
            entropy_coef=saved['entropy_coef'],gamma=control.gamma,lam=control.lam,target_kl=float('inf'))
        times.append(time.perf_counter()-start);results.append(result)
        assert all(torch.isfinite(p).all() for p in probe.parameters())
        assert any(not torch.equal(initial[k],v) for k,v in probe.state_dict().items() if k.startswith('gru.'))
    assert sha256(args.output/'preparation-baseline.pt')==digest
    original,_=load_policy(saved)
    result=dict(passed=True,baseline_sha256=digest,baseline_steps=saved['steps'],native_states=len(states),
        exact_current_native_prefix=True,history_removed=True,training_transitions=False,
        production_checkpoint_written=False,grads_into_gru=True,
        parameters_before=sum(p.numel() for p in original.parameters()),parameters_after=sum(p.numel() for p in model.parameters()),
        rollout_observation_mib=sum(v.nbytes for o in rollout['obs'] for v in o.values())/2**20,
        hidden_state_rollout_mib=length*count*256*4/2**20,
        six_env_inference_and_value_median_ms=statistics.median(inference)*1000,
        cpu_full_four_epoch_update_seconds=times,optimizer_reports=results,
        limits='repeated recorded native observations with synthetic rewards; NOT native collection throughput or competence')
    atomic_json(args.output/'audit.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
