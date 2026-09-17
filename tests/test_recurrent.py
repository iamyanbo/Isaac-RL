from collections import Counter
from copy import deepcopy
import json
import sys

import numpy as np
import pytest
import torch

from isaac_rl import train, train_vector, vector
from isaac_rl.env import IsaacEnv
from isaac_rl.history import ObservationHistory, ObservationCurrent, FRAME_VECTOR_SIZE, HISTORY
from isaac_rl.observation import observation_manifest, resolve_observation_profile
from isaac_rl.ppo import load_policy, as_tensor, optimize, action_statistics
from isaac_rl.recurrent import RecurrentActorCritic, PolicyMemory, sequence_batches, project_history_input
from isaac_rl.storage import checkpoint, sha256, atomic_json
from isaac_rl.timing import ControlTiming
from isaac_rl.rewards import profile_manifest
from test_core import state
from test_observation_history import combat_state, entity
from test_completion_reward import AdvancingBridge, assert_exact


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def observation(t=1,b=1):
    return dict(grid=torch.randn(t,b,10,16,28),vector=torch.randn(t,b,FRAME_VECTOR_SIZE+19))


def test_single_snapshot_exact_native_enrichment_and_executed_action(combat_state):
    entity(combat_state,1,200)
    current=ObservationCurrent();past=ObservationHistory()
    old=past.append(combat_state,Counter(),.5)
    obs=current.append(combat_state,Counter(),.5)
    np.testing.assert_array_equal(obs['grid'],old['grid'][:10])
    np.testing.assert_array_equal(obs['vector'][:FRAME_VECTOR_SIZE],old['vector'][:FRAME_VECTOR_SIZE])
    assert obs['vector'].shape==(2151,) and not obs['vector'][-19:].any()
    combat_state['episode_frame']+=4;combat_state['room']['id']+=1
    after=current.append(combat_state,Counter(),.5,[2,3,1])
    assert after['vector'][-19:].sum()==4 and after['vector'][-1]==1
    assert after['vector'][FRAME_VECTOR_SIZE+2]==after['vector'][FRAME_VECTOR_SIZE+12]==after['vector'][FRAME_VECTOR_SIZE+15]==1
    assert len(current.encoder.frames)==1 and current.stats['history_span_ticks']==0
    current.reset();reset=current.append(combat_state,Counter(),.5)
    assert not reset['vector'][-19:].any()
    assert obs['vector'][-19:].sum()==0  # No alias into future observations.


def test_temporal_substitution_keeps_rewards_idle_and_end_conditions(combat_state):
    combat_state['damage_signal']='hp_delta_v1'
    envs=[IsaacEnv(AdvancingBridge(deepcopy(combat_state)),frames=4,timing_profile='physical_v1',
        observation_profile=p,reward_profile='native_clear_v5',max_steps=8) for p in ('combat_history_v3','combat_gru_v4')]
    try:
        for env in envs:env.reset()
        for i in range(8):
            for env in envs:
                s=env.bridge.state
                s['room'].update(id=90 if i<4 else 91,enemies=2 if i<2 else 0,clear=i>=2)
                s['events']['damage_taken']=int(i>=5)
            a,b=[env.step([2,1,0]) for env in envs]
            assert a[1:4]==b[1:4] and a[4]['reward_components']==b[4]['reward_components']
            assert envs[0].idle==envs[1].idle
    finally:
        for env in envs:env.close()


def test_gru_online_and_batched_sequence_agree_resets_are_per_worker():
    model=RecurrentActorCritic();torch.nn.init.normal_(model.memory_readout.weight,std=.02)
    obs=observation(9,3);starts=torch.zeros(9,3,dtype=torch.bool);starts[0]=True;starts[4,1]=True
    with torch.no_grad():
        logits,values,final=model.forward_sequence(obs,model.initial_state(3),starts)
        hidden=model.initial_state(3);online=[]
        for i in range(9):
            result=model.forward_sequence({k:v[i:i+1] for k,v in obs.items()},hidden,starts[i:i+1])
            online.append(result[:2]);hidden=result[2]
    torch.testing.assert_close(logits,torch.cat([x[0] for x in online]),rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(values,torch.cat([x[1] for x in online]),rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(final,hidden)
    other={k:v.clone() for k,v in obs.items()}
    for v in other.values():v[:4,1]+=10
    with torch.no_grad():changed=model.forward_sequence(other,model.initial_state(3),starts)[0]
    torch.testing.assert_close(changed[4:,1],logits[4:,1])
    torch.testing.assert_close(changed[:,0],logits[:,0])


def test_memory_carries_past_four_frames_and_gradients_do_not_cross_episode_boundary():
    model=RecurrentActorCritic();torch.nn.init.eye_(model.memory_readout.weight)
    # Force a slow update gate to make the causal memory test numerically robust.
    with torch.no_grad():model.gru.bias_ih_l0[256:512].fill_(3)
    obs=observation(12,1);obs['vector'].requires_grad_()
    starts=torch.zeros(12,1,dtype=torch.bool);starts[0]=True
    _,values,_=model.forward_sequence(obs,model.initial_state(1),starts)
    values[-1].sum().backward()
    assert obs['vector'].grad[0].abs().sum()>0
    obs['vector'].grad.zero_();starts[7]=True
    _,values,_=model.forward_sequence(obs,model.initial_state(1),starts)
    values[-1].sum().backward()
    assert obs['vector'].grad[:7].count_nonzero()==0
    assert obs['vector'].grad[7].abs().sum()>0


def test_bootstrap_probe_does_not_double_advance_and_truncation_reset_is_explicit():
    model=RecurrentActorCritic();torch.nn.init.eye_(model.memory_readout.weight)
    memory=PolicyMemory(model,2);obs={k:v[0] for k,v in observation(1,2).items()}
    assert memory.rollout_state()['episode_starts'].all()
    with torch.no_grad():memory.act(obs)
    hidden=memory.hidden.clone();starts=memory.starts.copy()
    with torch.no_grad():
        expected=model.forward_sequence({k:v[None] for k,v in obs.items()},hidden,torch.zeros(1,2,dtype=torch.bool))[1][0]
        torch.testing.assert_close(memory.value(obs),expected)
    torch.testing.assert_close(memory.hidden,hidden);np.testing.assert_array_equal(memory.starts,starts)
    memory.reset_done([True,False])
    assert not memory.hidden[:,0].any()
    torch.testing.assert_close(memory.hidden[:,1],hidden[:,1])
    with pytest.raises(ValueError,match='explicit hidden state'):model.act(obs)


@pytest.mark.parametrize('t,b,batch',[(256,6,256),(11,3,4),(3,6,256)])
def test_sequence_sampler_never_shuffles_time_or_mixes_games(t,b,batch):
    seen=[]
    for group in sequence_batches(t,b,batch):
        lengths={end-start for env,start,end in group}
        assert len(lengths)==1 and sum(end-start for env,start,end in group)<=batch
        for env,start,end in group:seen.extend((i,env) for i in range(start,end))
    assert len(seen)==len(set(seen))==t*b
    assert set(seen)=={(i,e) for i in range(t) for e in range(b)}


def parent_checkpoint():
    model,adam=load_policy(observation_profile='combat_history_v3',with_optimizer=True)
    sum(p.square().sum() for p in model.parameters()).backward();adam.step()
    return dict(architecture=2,observation_profile='combat_history_v3',observation_layout=model.observation_layout,
        model=deepcopy(model.state_dict()),optimizer=deepcopy(adam.state_dict()))


def test_checkpoint_projection_optimizer_mapping_and_exact_recurrent_resume():
    saved=parent_checkpoint();model,adam=load_policy(saved,observation_profile='combat_gru_v4',with_optimizer=True)
    for name,value in saved['model'].items():
        assert_exact(model.state_dict()[name],project_history_input(value,name,model.state_dict()[name]))
    old_ids=saved['optimizer']['param_groups'][0]['params'];new=adam.state_dict()
    for i,name in enumerate(saved['model']):
        for slot,value in saved['optimizer']['state'][old_ids[i]].items():
            expected=project_history_input(value,name,model.state_dict()[name]) if value.ndim else value
            assert_exact(new['state'][i][slot],expected)
    assert all(i not in new['state'] for i in new['param_groups'][0]['params'][16:])
    assert not model.memory_readout.weight.any()
    source=dict(architecture=3,observation_profile='combat_gru_v4',observation_layout=model.observation_layout,
        model=model.state_dict(),optimizer=adam.state_dict())
    resumed,resumed_adam=load_policy(source,with_optimizer=True)
    assert_exact(resumed.state_dict(),model.state_dict());assert_exact(resumed_adam.state_dict(),adam.state_dict())
    with pytest.raises(ValueError,match='new run'):resolve_observation_profile('combat_gru_v4',saved,True)
    with pytest.raises(ValueError):load_policy(source,observation_profile='combat_history_v3')
    bad=deepcopy(source);bad['observation_layout']['memory']['hidden_size']=999
    with pytest.raises(ValueError,match='layout'):load_policy(bad)


def synthetic_rollout(model,t=9,b=2):
    memory=PolicyMemory(model,b);rollout={k:[] for k in ('obs','actions','log_probs','values','rewards','next_values','terminated','ended','hidden_states','episode_starts')}
    for i in range(t):
        obs={k:v[0] for k,v in observation(1,b).items()};state=memory.rollout_state()
        with torch.no_grad():
            a,lp,_,v=memory.act(obs);nv=memory.value(obs).numpy()
        ends=np.zeros(b,dtype=bool);ends[0]=i==3
        row=dict(obs={k:v.numpy() for k,v in obs.items()},actions=a.numpy(),log_probs=lp.numpy(),values=v.numpy(),
            rewards=np.ones(b,dtype=np.float32),next_values=nv,terminated=np.zeros(b,dtype=bool),ended=ends,**state)
        for k,v in row.items():rollout[k].append(v)
        memory.reset_done(ends)
    return rollout


def test_actual_sequence_ppo_ratio_reconstruction_and_gradients_into_gru():
    model,adam=load_policy(observation_profile='combat_gru_v4',with_optimizer=True)
    rollout=synthetic_rollout(model)
    # Recompute old probabilities as whole ordered sequences from saved states.
    obs={k:torch.as_tensor(np.stack([o[k] for o in rollout['obs']])) for k in ('grid','vector')}
    with torch.no_grad():
        logits,values,_=model.forward_sequence(obs,torch.as_tensor(rollout['hidden_states'][0])[None],torch.as_tensor(np.stack(rollout['episode_starts'])))
        _,logs,_,_=action_statistics(logits,values,torch.as_tensor(np.stack(rollout['actions'])))
    torch.testing.assert_close(logs,torch.as_tensor(np.stack(rollout['log_probs'])),atol=1e-6,rtol=1e-5)
    before={k:v.clone() for k,v in model.state_dict().items()}
    result=optimize(model,adam,rollout,batch_size=8,epochs=2,target_kl=float('inf'))
    assert result['recurrent_sequence_length']==8 and result['recurrent_tokens']==36
    assert all(np.isfinite(v) for v in result.values())
    assert any(not torch.equal(v,before[k]) for k,v in model.state_dict().items() if k.startswith('gru.'))
    assert model.memory_readout.weight.count_nonzero()>0
    assert all(torch.isfinite(p).all() for p in model.parameters())
    bad=deepcopy(rollout);bad['episode_starts'][4][0]=False
    with pytest.raises(ValueError,match='masks'):optimize(model,adam,bad)


def test_frozen_evaluation_main_carries_memory_and_resets_each_episode(tmp_path,monkeypatch,combat_state):
    from isaac_rl import evaluate
    model,adam=load_policy(observation_profile='combat_gru_v4',with_optimizer=True)
    source=tmp_path/'model.pt';control=ControlTiming(4,'physical_v1')
    checkpoint(source,model,adam,dict(architecture=3,observation_profile='combat_gru_v4',observation_layout=model.observation_layout,
        steps=100,frames=4,timing_profile='physical_v1',training_seeds=['OLD'],reward_profile='native_clear_v5'))
    combat_state['damage_signal']='hp_delta_v1'
    class Game(AdvancingBridge):
        def request(self,op,**fields):
            if op=='reset':
                self.state['episode']+=1;self.state['episode_frame']=1
                self.state['seed']=f"TEST {self.state['episode']:04d}"
            return super().request(op,**fields)
    monkeypatch.setattr(evaluate,'Bridge',lambda **kw:Game(deepcopy(combat_state)))
    observations=[]
    class SpyMemory(PolicyMemory):
        def act(self,*a,**kw):
            observations.append((self.starts.copy(),self.hidden.clone()))
            return super().act(*a,**kw)
    monkeypatch.setattr(evaluate,'PolicyMemory',SpyMemory)
    def fail(*a,**kw):raise AssertionError('Frozen evaluation cannot train')
    monkeypatch.setattr(torch.optim.Adam,'step',fail)
    output=tmp_path/'evaluation';digest=sha256(source)
    monkeypatch.setattr(sys,'argv',['evaluate',str(source),'--output',str(output),'--episodes','2','--max-episode-steps','3','--idle-limit','900'])
    evaluate.main()
    assert len(observations)==6
    for i,(starts,hidden) in enumerate(observations):
        assert bool(starts[0])==(i%3==0)
        assert bool(hidden.count_nonzero())==(i%3!=0)
    assert sha256(source)==digest
    assert json.loads((output/'result.json').read_text())['complete']


@pytest.mark.parametrize('module',[train,train_vector])
def test_real_trainers_gru_fork_ordered_resets_bootstrap_and_resume(tmp_path,monkeypatch,combat_state,module):
    saved=parent_checkpoint();parent=tmp_path/'parent.pt'
    model,adam=load_policy(saved,with_optimizer=True);control=ControlTiming(4,'physical_v1')
    checkpoint(parent,model,adam,dict(architecture=2,observation_profile='combat_history_v3',observation_layout=model.observation_layout,
        steps=100,updates=1,episodes=2,recent=[],losses={},training_seeds=['OLD'],numpy_rng=np.random.get_state(),cuda_rng=None,
        frames=4,timing_profile='physical_v1',entropy_coef=.002,control_timing=control.manifest(),
        training_schedule=dict(rollout=4,batch_size=4,max_episode_steps=2,idle_limit=900),native_frames=400,native_frames_origin_steps=0,
        reward_profile='native_clear_v5',reward=profile_manifest('native_clear_v5',control)))
    digest=sha256(parent);combat_state['damage_signal']='hp_delta_v1'
    factory=lambda *a,**kw:AdvancingBridge(deepcopy(combat_state))
    monkeypatch.setattr(train,'Bridge',factory);monkeypatch.setattr(vector,'Bridge',factory)
    snapshots=[];rollouts=[];real_optimize=module.optimize
    def save(*a,**kw):
        checkpoint(*a,**kw);snapshots.append(torch.load(a[0],weights_only=False))
    def update(model,adam,rollout,**kw):
        rollouts.append(deepcopy(rollout))
        return real_optimize(model,adam,rollout,**kw)
    monkeypatch.setattr(module,'checkpoint',save);monkeypatch.setattr(module,'optimize',update)
    run=tmp_path/'new';ports=['--ports','18000','18001','18002','18003','18004','18005'] if module is train_vector else []
    count=6 if ports else 1;target=100+8*count
    monkeypatch.setattr(sys,'argv',['trainer','--run',str(run),'--resume',str(parent),'--observation-profile','combat_gru_v4','--steps',str(target),*ports]);module.main()
    initial=snapshots[0];old=torch.load(parent,weights_only=False)
    for key in ('torch_rng','numpy_rng','steps','updates','episodes','native_frames','frames','entropy_coef','control_timing','training_schedule','reward_profile','reward'):
        assert_exact(initial[key],old[key])
    assert initial['architecture']==3 and snapshots[-1]['updates']==3
    for r in rollouts:
        starts=np.stack(r['episode_starts']);np.testing.assert_array_equal(starts[2],True)
        assert np.stack(r['hidden_states'])[2].sum()==0
        assert np.asarray(r['next_values'])[1].any()  # Truncation bootstraps terminal observation.
    trained=snapshots[-1];rollouts.clear();snapshots.clear()
    monkeypatch.setattr(sys,'argv',['trainer','--run',str(run),'--resume',str(run/'latest.pt'),'--steps',str(target+4*count),*ports]);module.main()
    assert_exact(snapshots[0]['model'],trained['model']);assert_exact(snapshots[0]['optimizer'],trained['optimizer'])
    assert np.stack(rollouts[0]['hidden_states'])[0].sum()==0  # Restart creates fresh native episodes.
    assert sha256(parent)==digest
