from copy import deepcopy
import io
import json
import sys

import pytest
import torch

from isaac_rl import compare_observability as comparison
from isaac_rl.ppo import load_policy
from isaac_rl.storage import checkpoint,atomic_json,sha256
from isaac_rl.timing import ControlTiming
from test_core import state,FakeBridge
from test_observation_history import combat_state
from test_completion_reward import AdvancingBridge


def record(seed,arm,clears=0,**extra):
    return dict(seed=seed,arm=arm,native_combat_clears_alive=clears,
        native_counter_agrees=True,boss_seen=False,boss_defeated=False,
        success=False,reason="idle",**extra)


def test_comparison_uses_all_seeds_and_never_calls_a_late_result_fixed_budget():
    seeds=[f"seed{i}" for i in range(100)]
    rows=[record(seed,arm,int(i<30)) for i,seed in enumerate(seeds) for arm in comparison.ARMS]
    result=comparison.summarize(rows,seeds,100,False)
    assert result["complete"] and result["arms"]["treatment"]["living_combat_clear_rate"]==.3
    assert result["comparison"]["original_fixed_budget_hypothesis_rejected"] is None
    assert result["comparison"]["delta"]==0
    assert not result["arms"]["treatment"]["boss_success_gate"]["passed"]
    assert comparison.summarize(rows[:-1],seeds,100,False)["comparison"] is None
    rows[0]["native_counter_agrees"]=False
    assert comparison.summarize(rows,seeds,100,False)["comparison"] is None
    with pytest.raises(ValueError,match="Duplicate"):
        comparison.summarize(rows+[rows[0]],seeds,100,False)


def test_predeclared_margin_and_budget_guard():
    seeds=[str(i) for i in range(100)]
    rows=[record(s,arm) for s in seeds for arm in comparison.ARMS]
    actual=comparison.summarize(rows,seeds,100,True)
    assert actual["comparison"]["conservative_95_upper"]<.10
    assert actual["comparison"]["original_fixed_budget_hypothesis_rejected"]
    assert comparison.summarize(rows,seeds,100,False)["comparison"]["original_fixed_budget_hypothesis_rejected"] is None


def test_native_room_clear_evidence_requires_previous_combat_no_enemies_and_alive(combat_state):
    env=comparison.NativeEvidenceEnv(FakeBridge(combat_state),reward_profile="completion_v4")
    combat_state['damage_signal']='hp_delta_v1'
    env.reset()
    assert not env.native_clears  # clear starting room is not a combat win
    env.state=deepcopy(combat_state)
    env.state['room'].update(enemies=1,clear=False)
    env._record_native(None)
    before=deepcopy(env.state)
    env.state=deepcopy(before); env.state['room']['clear']=True
    env._record_native(before)
    assert not env.native_clears  # remaining enemies contradict clear evidence
    env.state=deepcopy(before); env.state['room'].update(clear=True,enemies=0)
    env.state['player']['dead']=True
    env._record_native(before)
    assert not env.native_clears
    env.state=deepcopy(env.state); env.state['player']['dead']=False
    env._record_native(before)
    assert len(env.native_clears)==1
    env._record_native(before)
    assert len(env.native_clears)==1


def test_seed_selection_excludes_canonical_training_and_duplicate_seeds(tmp_path):
    class Seeds:
        def __init__(self): self.values=iter(['ABCD 1234','NEWW 5678','NEWW5678','LAST 1234'])
        def reset(self): return None,{'game_seed':next(self.values)}
    selected=comparison.choose_seeds(Seeds(),2,{'abcd1234'},tmp_path,lambda **kw:None)
    assert selected==['NEWW5678','LAST1234']
    assert json.loads((tmp_path/'seeds.json').read_text())==selected


def test_transition_tags_distinguish_policies_without_changing_actions():
    stream=io.StringIO()
    comparison.TaggedTransitions(stream,'parent').write(json.dumps({'arm':'stochastic','executed':[1,2,0]}))
    assert json.loads(stream.getvalue())==dict(arm='stochastic',comparison_arm='parent',executed=[1,2,0])


def make_checkpoint(path,profile):
    model,adam=load_policy(observation_profile=profile,with_optimizer=True)
    checkpoint(path,model,adam,dict(architecture=model.observation_layout['architecture'],
        observation_layout=model.observation_layout,observation_profile=profile,
        frames=4,timing_profile='physical_v1',control_timing=ControlTiming(4,'physical_v1').manifest(),
        training_schedule=dict(rollout=256,batch_size=256,max_episode_steps=6750,idle_limit=900),
        reward_profile='completion_v4',steps=100,training_seeds=['OLDD 1234']))


def test_real_comparison_entrypoint_shared_bridge_pairing_and_no_optimizer(tmp_path,monkeypatch,combat_state):
    torch.set_num_threads(2)
    review=tmp_path/'review'; review.mkdir()
    for arm,profile in [('parent','terrain_v2'),('treatment','combat_history_v3')]:
        make_checkpoint(review/f'{arm}.pt',profile)
    atomic_json(review/'review.json',dict(parent_sha256=sha256(review/'parent.pt'),
        treatment_sha256=sha256(review/'treatment.pt'),budget_compliant=False,budget_overrun_decisions=10))
    atomic_json(review/'excluded-seeds.json',['OLDD1234'])
    combat_state.update(damage_signal='hp_delta_v1',bridge_port=10002,seed='NEWW 5678')
    class OneStepBridge(AdvancingBridge):
        def connect(self): return self.state
        def request(self,op,**fields):
            self.state['terminal']=op=='step'
            self.state['player']['dead']=op=='step'
            return super().request(op,**fields)
    bridge=OneStepBridge(deepcopy(combat_state))
    bridge_calls=[]
    def factory(**kw): bridge_calls.append(kw); return bridge
    monkeypatch.setattr(comparison,'Bridge',factory)
    monkeypatch.setattr(comparison,'park_after_evaluation',lambda *a:dict(parked=True))
    def no_step(*a,**kw): raise AssertionError('Evaluation may not optimize')
    monkeypatch.setattr(torch.optim.Adam,'step',no_step)
    seen=[]
    original=comparison.run_case
    def capture(*a,**kw):
        seen.append((a[3],a[4],a[0].observation_profile))
        return original(*a,**kw)
    monkeypatch.setattr(comparison,'run_case',capture)
    output=tmp_path/'output'
    monkeypatch.setattr(sys,'argv',['compare','--review',str(review),'--output',str(output),'--seeds','1'])
    comparison.main()
    bridge.trace.close()  # fake bridge has no actual socket/trace cleanup
    assert bridge_calls==[{'port':10002}]
    assert seen==[('NEWW5678',946513,'terrain_v2'),('NEWW5678',946513,'combat_history_v3')]
    result=json.loads((output/'result.json').read_text())
    assert result['complete'] and result['native_counters_valid']
    assert result['comparison']['original_fixed_budget_hypothesis_rejected'] is None
    status=json.loads((output/'status.json').read_text())
    assert status['cleanup_complete'] and status['status']=='finished' and status['completed_episodes']==2


def test_different_control_or_budget_rejected():
    base=dict(frames=4,timing_profile='physical_v1',reward_profile='completion_v4',
        training_schedule=dict(rollout=256,batch_size=256,max_episode_steps=6750,idle_limit=900))
    a=dict(base,observation_profile='terrain_v2')
    b=dict(base,observation_profile='combat_history_v3')
    resolved=comparison.validate_comparison(dict(parent=a,treatment=b))
    assert resolved['frames']==4 and resolved['idle_limit']==900 and resolved['max_steps']==6750
    b=deepcopy(b); b['training_schedule']['idle_limit']=901
    with pytest.raises(ValueError,match='limits'):
        comparison.validate_comparison(dict(parent=a,treatment=b))
