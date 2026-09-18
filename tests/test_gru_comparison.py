from copy import deepcopy
import json
import sys

import pytest
import torch

from isaac_rl import compare_observability as comparison
from isaac_rl.ppo import load_policy
from isaac_rl.storage import atomic_json,checkpoint,sha256
from isaac_rl.rewards import profile_manifest
from isaac_rl.timing import ControlTiming
from test_core import state,FakeBridge
from test_observation_history import combat_state
from test_completion_reward import AdvancingBridge
from test_observability_comparison import record


def test_gru_primary_is_native_boss_not_room_metrics_and_100_seed_guard():
    seeds=[str(i) for i in range(100)]
    rows=[dict(record(s,a,clears=5),native_outcome_verified=True) for s in seeds for a in comparison.ARMS]
    result=comparison.summarize_gru(rows,seeds,100,True)
    assert result['complete'] and result['comparison']['fixed_budget_hypothesis_rejected']
    assert result['comparison']['delta']==0 and result['comparison']['conservative_95_upper']<.10
    assert not result['arms']['treatment']['boss_success_gate']['passed']
    rows[0]['native_counter_agrees']=False
    assert comparison.summarize_gru(rows,seeds,100,True)['comparison']['secondary_counter_disagreements']==1
    rows[0]['native_outcome_verified']=False
    assert comparison.summarize_gru(rows,seeds,100,True)['comparison'] is None
    assert comparison.summarize_gru(rows[2:],seeds,100,True)['comparison'] is None
    assert comparison.summarize_gru(rows[2:],seeds[1:],99,True)['comparison']['fixed_budget_hypothesis_rejected'] is None


def test_native_boss_verification_detects_false_success_and_missed_success(combat_state):
    initial=deepcopy(combat_state);initial['seed']='ABCD 1234'
    final=deepcopy(initial);result=dict(success=False,reason='death')
    assert not comparison.verify_boss_outcome(initial,final,result,'ABCD1234')
    with pytest.raises(ValueError):comparison.verify_boss_outcome(initial,final,dict(success=True,reason='boss_clear'),'ABCD1234')
    final.update(boss_seen=True,boss_defeated=True);final['room'].update(type=5,clear=True,enemies=0)
    with pytest.raises(ValueError):comparison.verify_boss_outcome(initial,final,result,'ABCD1234')
    assert comparison.verify_boss_outcome(initial,final,dict(success=True,reason='boss_clear'),'ABCD1234')
    final['player']['dead']=True
    with pytest.raises(ValueError):comparison.verify_boss_outcome(initial,final,dict(success=True,reason='boss_clear'),'ABCD1234')


def test_gru_secondary_native_clear_allows_residual_npc_but_not_entry_credit(combat_state):
    combat_state['damage_signal']='hp_delta_v1'
    env=comparison.NativeGRUEvidenceEnv(FakeBridge(combat_state),reward_profile='native_clear_v5')
    env.reset();env.state['room'].update(enemies=1,clear=True)
    env._record_native(None);assert not env.native_clears
    env.state=deepcopy(env.state);env.state['room']['clear']=False;env._record_native(None)
    before=deepcopy(env.state);env.state=deepcopy(env.state);env.state['room']['clear']=True
    env._record_native(before);assert len(env.native_clears)==1


def test_gru_budget_rejects_late_checkpoint_false_origin_and_metadata():
    saved=dict(parent=dict(steps=100),treatment=dict(steps=1201252,origin=dict(parent_sha256='abc')))
    review=dict(parent_sha256='abc',target_steps=1200100,budget_compliant=True,budget_overrun_decisions=1152)
    assert comparison.validate_gru_budget(saved,review)==1152
    for change in (dict(steps=1300100),dict(origin=dict(parent_sha256='different'))):
        bad=deepcopy(saved);bad['treatment'].update(change)
        with pytest.raises(ValueError):comparison.validate_gru_budget(bad,review)
    with pytest.raises(ValueError):comparison.validate_gru_budget(saved,dict(review,budget_overrun_decisions=0))


def test_real_gru_comparison_main_is_frozen_paired_and_resets_hidden_state(tmp_path,monkeypatch,combat_state):
    torch.set_num_threads(2)
    review=tmp_path/'review';review.mkdir();control=ControlTiming(4,'physical_v1')
    for arm,profile,steps in [('parent','combat_history_v3',100),('treatment','combat_gru_v4',1201252)]:
        model,adam=load_policy(observation_profile=profile,with_optimizer=True)
        checkpoint(review/f'{arm}.pt',model,adam,dict(architecture=model.observation_layout['architecture'],
            observation_layout=model.observation_layout,observation_profile=profile,frames=4,timing_profile='physical_v1',
            entropy_coef=.002,control_timing=control.manifest(),ports=[9999,10000,10001,10003,10004,10005],
            training_schedule=dict(rollout=256,batch_size=256,max_episode_steps=6750,idle_limit=900),
            reward_profile='native_clear_v5',reward=profile_manifest('native_clear_v5',control),steps=steps,
            origin=dict(parent_sha256=sha256(review/'parent.pt')) if arm=='treatment' else {},training_seeds=['OLDD1234']))
    atomic_json(review/'review.json',dict(parent_sha256=sha256(review/'parent.pt'),treatment_sha256=sha256(review/'treatment.pt'),
        target_steps=1200100,budget_compliant=True,budget_overrun_decisions=1152))
    atomic_json(review/'excluded-seeds.json',['OLDD1234'])
    combat_state.update(damage_signal='hp_delta_v1',bridge_port=10002,seed='NEWW 5678')
    class Game(AdvancingBridge):
        def request(self,op,**fields):
            if op=='reset':self.state['episode_frame']=1
            dead=op=='step' and self.state['episode_frame']>=9
            self.state['terminal']=dead;self.state['player']['dead']=dead
            return super().request(op,**fields)
    bridge=Game(deepcopy(combat_state));calls=[]
    def connect(**kw):calls.append(kw);return bridge
    monkeypatch.setattr(comparison,'Bridge',connect)
    monkeypatch.setattr(comparison,'park_after_evaluation',lambda *a:dict(parked=True))
    def fail(*a,**kw):raise AssertionError('No evaluation optimizer steps')
    monkeypatch.setattr(torch.optim.Adam,'step',fail)
    import isaac_rl.behavior_eval as behavior
    original=behavior.PolicyMemory;memories=[]
    class SpyMemory(original):
        def __init__(self,*a):super().__init__(*a);memories.append(self)
    monkeypatch.setattr(behavior,'PolicyMemory',SpyMemory)
    output=tmp_path/'output'
    monkeypatch.setattr(sys,'argv',['compare','--review',str(review),'--output',str(output),'--experiment','gru','--seeds','1'])
    comparison.main();bridge.trace.close()
    assert calls==[{'port':10002}] and len(memories)==2
    assert memories[1].recurrent and memories[1].hidden.abs().sum()>0
    rows=[json.loads(x) for x in (output/'episodes.jsonl').read_text().splitlines()]
    assert [r['arm'] for r in rows]==['parent','treatment']
    assert all(r['rng_seed']==946513 and r['seed']=='NEWW5678' and r['native_outcome_verified'] for r in rows)
    result=json.loads((output/'result.json').read_text())
    assert result['complete'] and result['native_outcomes_valid']
    assert result['comparison']['fixed_budget_hypothesis_rejected'] is None  # Diagnostic1 seed != predeclared100.
