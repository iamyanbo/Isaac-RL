from copy import deepcopy
from dataclasses import asdict
import json
import sys

import numpy as np
import pytest
import torch

from isaac_rl import train,train_vector,vector
from isaac_rl.env import IsaacEnv
from isaac_rl.ppo import load_policy
from isaac_rl.rewards import PROFILES,profile_manifest
from isaac_rl.storage import checkpoint,sha256,atomic_json
from isaac_rl.timing import ControlTiming
from test_core import state
from test_observation_history import combat_state,entity
from test_completion_reward import AdvancingBridge,assert_exact


def test_one_event_qualification_change_no_coefficient_or_unit_changes():
    old,new=[asdict(PROFILES[p]) for p in ('completion_v4','native_clear_v5')]
    assert {k for k in old if old[k]!=new[k]}=={'require_uncleared_combat'}
    assert not old['require_uncleared_combat'] and new['require_uncleared_combat']


def make_envs(s):
    s['damage_signal']='hp_delta_v1'
    return [IsaacEnv(AdvancingBridge(deepcopy(s)),frames=4,timing_profile='physical_v1',
        observation_profile='combat_history_v3',reward_profile=profile) for profile in ('completion_v4','native_clear_v5')]


def test_already_cleared_npc_room_no_longer_pays_combat_bonus(combat_state):
    envs=make_envs(combat_state)
    try:
        for env in envs:
            env.reset()
            s=env.bridge.state
            s['room'].update(id=97,clear=True,enemies=1)
            e,_=entity(s,123,300,kind=1)
            e[5]=218  # Native wall-hugger counterexample.
        old,new=[env.step([2,0,0]) for env in envs]
        assert old[4]['reward_components']['room_clear']==10
        assert new[4]['reward_components']['room_clear']==0
        assert old[1]-new[1]==pytest.approx(10)
        for key in old[4]['reward_components']:
            if key!='room_clear': assert old[4]['reward_components'][key]==new[4]['reward_components'][key]
        for key in old[0]: np.testing.assert_array_equal(old[0][key],new[0][key])
        assert old[2:4]==new[2:4] and envs[0].idle==envs[1].idle
        assert old[4]['combat_clears']==1 and new[4]['combat_clears']==0
        assert all(env.step([0,0,0])[4]['reward_components']['room_clear']==0 for env in envs)
    finally:
        for env in envs: env.close()


@pytest.mark.parametrize('remaining',[0,1,8])
@pytest.mark.parametrize('dead',[False,True])
def test_real_native_clear_transition_is_preserved_even_with_residual_npcs(combat_state,remaining,dead):
    combat_state['room'].update(clear=False,enemies=8)
    combat_state['cleared']={}
    envs=make_envs(combat_state)
    try:
        for env in envs:
            env.reset()
            env.bridge.state['room'].update(clear=True,enemies=remaining)
            env.bridge.state['player']['dead']=dead
            env.bridge.state['terminal']=dead
        old,new=[env.step([0,2,0]) for env in envs]
        assert old[1:4]==new[1:4]
        assert old[4]['reward_components']==new[4]['reward_components']
        assert new[4]['reward_components']['room_clear']==10
        for key in old[0]: np.testing.assert_array_equal(old[0][key],new[0][key])
        if not dead:
            assert all(env.step([0,0,0])[4]['reward_components']['room_clear']==0 for env in envs)
    finally:
        for env in envs: env.close()


@pytest.mark.parametrize('module',[train,train_vector])
def test_actual_enriched_trainers_fork_and_resume_gate_with_exact_model_adam_rng(tmp_path,monkeypatch,combat_state,module):
    torch.set_num_threads(2)
    model,adam=load_policy(observation_profile='combat_history_v3',with_optimizer=True)
    sum(p.square().sum() for p in model.parameters()).backward();adam.step()
    parent=tmp_path/'parent.pt';control=ControlTiming(4,'physical_v1')
    schedule=dict(rollout=256,batch_size=256,max_episode_steps=6750,idle_limit=900)
    checkpoint(parent,model,adam,dict(architecture=2,observation_profile='combat_history_v3',
        observation_layout=model.observation_layout,steps=100,updates=1,episodes=2,recent=[dict(r=2.,success=False)],
        losses=dict(loss=2.),training_seeds=['OLD'],numpy_rng=np.random.get_state(),
        frames=4,timing_profile='physical_v1',entropy_coef=.002,control_timing=control.manifest(),
        training_schedule=schedule,native_frames=400,native_frames_origin_steps=0,
        reward_profile='completion_v4',reward=profile_manifest('completion_v4',control)))
    original=torch.load(parent,weights_only=False);digest=sha256(parent)
    combat_state['damage_signal']='hp_delta_v1'
    factory=lambda *a,**kw:AdvancingBridge(deepcopy(combat_state))
    monkeypatch.setattr(train,'Bridge',factory);monkeypatch.setattr(vector,'Bridge',factory)
    snapshots=[]
    def capture(*a,**kw):
        checkpoint(*a,**kw);snapshots.append(torch.load(a[0],weights_only=False))
    monkeypatch.setattr(module,'checkpoint',capture)
    run=tmp_path/'new'
    run.mkdir()
    atomic_json(run/'snapshot_steps.json',[103])
    ports=['--ports','18000','18001','18002','18003','18004','18005'] if module is train_vector else []
    monkeypatch.setattr(sys,'argv',['trainer','--run',str(run),'--resume',str(parent),'--reward-profile','native_clear_v5','--steps','106',*ports])
    module.main()
    initial,trained=snapshots[0],snapshots[-1]
    for key in ['model','optimizer','torch_rng','numpy_rng','steps','updates','episodes','native_frames','native_frames_origin_steps',
                'frames','timing_profile','entropy_coef','training_schedule','control_timing','observation_layout','observation_profile','architecture']:
        assert_exact(initial[key],original[key])
    assert initial['recent']==[] and initial['losses']=={}
    assert trained['reward_profile']=='native_clear_v5' and trained['steps']==106 and trained['updates']==2
    assert trained['optimizer']['param_groups']==original['optimizer']['param_groups']
    assert all(torch.isfinite(t).all() for t in trained['model'].values())
    config=json.loads((run/'config.json').read_text())
    assert config['reward']==profile_manifest('native_clear_v5',control)
    retained=torch.load(run/'milestone-103.pt',weights_only=False)
    assert retained['steps']==106 and retained['updates']==2
    assert_exact(retained['model'],trained['model'])
    retained_digest=sha256(run/'milestone-103.pt')
    if module is train_vector: assert len(config['ports'])==6
    args=['trainer','--run',str(run),'--resume',str(run/'latest.pt'),'--steps','112',*ports]
    monkeypatch.setattr(sys,'argv',args);module.main()
    assert snapshots[-1]['steps']==112 and snapshots[-1]['reward_profile']=='native_clear_v5'
    assert sha256(run/'milestone-103.pt')==retained_digest
    monkeypatch.setattr(sys,'argv',args+['--reward-profile','completion_v4'])
    with pytest.raises(RuntimeError,match='new run directory'):module.main()
    assert sha256(parent)==digest
