from collections import Counter
from copy import deepcopy
import json
import sys

import numpy as np
import pytest
import torch

from isaac_rl.bridge import BridgeError
from isaac_rl.env import IsaacEnv
from isaac_rl.history import (ObservationHistory, HISTORY, FRAME_VECTOR_SIZE, HISTORY_VECTOR_SIZE,
                              PLAYER_SIZE, ENEMY_SIZE, HAZARD_SIZE, ENEMIES, HAZARDS)
from isaac_rl.observation import VECTOR_SIZE, encode, observation_manifest, resolve_observation_profile
from isaac_rl.ppo import ActorCritic, as_tensor, load_policy, optimize
from isaac_rl import train, train_vector, vector
from isaac_rl.storage import checkpoint, sha256
from isaac_rl.timing import ControlTiming
from test_completion_reward import AdvancingBridge, assert_exact
from test_core import state, FakeBridge


@pytest.fixture
def combat_state(state):
    state["combat_schema"] = "combat_v1"
    state["combat_player"] = dict(size=10,size_multi=[1,1],fire_cooldown=3,
        damage_cooldown_render_frames=0,invincible=False,can_shoot=True,can_fly=False,collision=4)
    state["combat_entities"] = []
    return state


def entity(s, identity, x, kind=2, **changes):
    e = [x,200,4,0,kind,9 if kind == 2 else 7 if kind == 7 else 10,0,0,5,10,12,identity]
    extra = dict(id=identity,size_multi=[1,1],collision=4,collision_damage=1,
        height=-10,falling_speed=0,falling_accel=0,npc_state=3,state_frame=10,animation_frame=2,
        endpoint=[500,200],angle=0,radius=30,circle=False,sample=False,timeout=60,
        samples=[[x,200],[500,200]],sample_count=2,countdown=10,geometry_valid=True)
    extra.update(changes)
    s["entities"].append(e)
    s["combat_entities"].append(extra)
    return e,extra


def append(history,s,action=None):
    return history.append(s,Counter(),.5,action)


def rows(obs):
    return obs["vector"][:HISTORY*FRAME_VECTOR_SIZE].reshape(HISTORY,FRAME_VECTOR_SIZE)


def hazards(obs,age=0):
    return rows(obs)[age,-HAZARDS*HAZARD_SIZE:].reshape(HAZARDS,HAZARD_SIZE)


def test_exact_current_prefix_and_shapes(combat_state):
    entity(combat_state,1,200)
    obs = append(ObservationHistory(),combat_state)
    old = encode(combat_state,Counter(),"terrain_v2",.5)
    np.testing.assert_array_equal(obs["grid"][:10],old["grid"])
    np.testing.assert_array_equal(obs["vector"][:VECTOR_SIZE],old["vector"])
    assert obs["grid"].shape == (40,16,28)
    assert obs["vector"].shape == (HISTORY_VECTOR_SIZE,)
    assert not rows(obs)[1:].any()
    assert not obs["vector"][-54:].any()
    assert all(np.isfinite(v).all() and np.abs(v).max() <= 1 for v in obs.values())


@pytest.mark.parametrize("field,value",[("fire_cooldown",9),("damage_cooldown_render_frames",60),
    ("invincible",True),("size",20),("size_multi",[2,1]),("can_shoot",False)])
def test_player_aliases_are_distinguished(combat_state,field,value):
    other = deepcopy(combat_state)
    other["combat_player"][field] = value
    np.testing.assert_array_equal(encode(combat_state,Counter())["vector"],encode(other,Counter())["vector"])
    a,b = [append(ObservationHistory(),s)["vector"] for s in (combat_state,other)]
    assert not np.array_equal(a,b)


@pytest.mark.parametrize("field,value",[("angle",90),("endpoint",[200,400]),("circle",True),
    ("radius",80),("samples",[[200,200],[500,400]]),("collision",0)])
def test_laser_geometry_and_collision_survive_encoding(combat_state,field,value):
    _,extra = entity(combat_state,1,200,7)
    a = append(ObservationHistory(),combat_state)
    extra[field] = value
    b = append(ObservationHistory(),combat_state)
    np.testing.assert_array_equal(a["vector"][:VECTOR_SIZE],b["vector"][:VECTOR_SIZE])
    assert not np.array_equal(a["vector"],b["vector"])


def test_projectile_category_and_fast_velocities_are_not_aliased(combat_state):
    e,_ = entity(combat_state,1,200)
    a = append(ObservationHistory(),combat_state)
    e[6] = 5
    b = append(ObservationHistory(),combat_state)
    assert not np.array_equal(hazards(a),hazards(b))
    e[2] = 20
    a = append(ObservationHistory(),combat_state)
    e[2] = 40
    b = append(ObservationHistory(),combat_state)
    assert hazards(a)[0,3] != hazards(b)[0,3]


def test_identity_tracks_distance_crossings_and_disappearances(combat_state):
    s = combat_state
    a,_ = entity(s,101,310)
    b,_ = entity(s,202,400)
    h = ObservationHistory()
    append(h,s)
    a[0],b[0] = 500,330
    s["episode_frame"] += 4
    obs = append(h,s,[2,1,0])
    # Current nearest is B. Historical B must follow it in slot zero, not A.
    assert hazards(obs,0)[0,1] == pytest.approx(10/560)
    assert hazards(obs,1)[0,1] == pytest.approx(80/560)
    s["entities"].pop(1)
    s["combat_entities"].pop(1)
    s["episode_frame"] += 4
    obs = append(h,s,[1,0,0])
    assert hazards(obs,0)[1,0] == 0  # disappeared B retained as a history track
    assert hazards(obs,1)[1,0] == 1
    # Numeric IDs themselves are never supplied to the network.
    renamed = deepcopy(s)
    renamed["entities"][0][11] = renamed["combat_entities"][0]["id"] = 123456789
    x,y = [append(ObservationHistory(),v) for v in (s,renamed)]
    np.testing.assert_array_equal(x["vector"],y["vector"])


def test_actions_are_intervening_executed_actions_and_history_does_not_alias(combat_state):
    s,h = combat_state,ObservationHistory()
    first = append(h,s)
    first_copy = deepcopy(first)
    for action in ([1,2,0],[2,3,1],[3,4,2],[4,0,3]):
        s["episode_frame"] += 4
        obs = append(h,s,action)
    assert h.stats["history_span_ticks"] == 12
    assert h.stats["history_valid"] == 4
    action_rows = obs["vector"][-54:].reshape(3,18)
    for row,action in zip(action_rows,([4,0,3],[3,4,2],[2,3,1])):
        assert np.flatnonzero(row).tolist() == [action[0],9+action[1],14+action[2]]
    np.testing.assert_array_equal(first["vector"],first_copy["vector"])
    twin = ObservationHistory()
    append(twin,s)
    assert twin.stats["history_valid"] == 1


def test_identical_current_snapshots_retain_distinct_previous_state_and_action(combat_state):
    a,b = ObservationHistory(),ObservationHistory()
    before = deepcopy(combat_state)
    before["player"]["x"] -= 20
    append(a,before)
    append(b,combat_state)
    combat_state["episode_frame"] += 4
    x,y = append(a,combat_state,[2,1,0]),append(b,combat_state,[0,1,0])
    np.testing.assert_array_equal(rows(x)[0],rows(y)[0])
    assert not np.array_equal(rows(x)[1],rows(y)[1])
    assert not np.array_equal(x["vector"][-54:],y["vector"][-54:])


def test_unknown_native_curve_geometry_is_masked_and_reported(combat_state):
    entity(combat_state,1,100,7,geometry_valid=False,sample=True,samples=[],sample_count=1)
    history = ObservationHistory()
    obs = append(history,combat_state)
    assert hazards(obs)[0,-1] == 0
    assert history.stats["invalid_laser_geometry"] == 1


@pytest.mark.parametrize("boundary",["room","episode","stage"])
def test_boundaries_clear_history(combat_state,boundary):
    s,h = combat_state,ObservationHistory()
    append(h,s)
    s["episode_frame"] += 4
    append(h,s,[2,1,0])
    if boundary == "room":
        s["room"]["id"] += 1
    else:
        s[boundary] += 1
    s["episode_frame"] += 4
    obs = append(h,s,[1,2,0])
    assert h.stats["history_valid"] == 1
    assert not obs["vector"][-54:].any()
    assert not rows(obs)[1:].any()


def test_terminal_history_survives_reset_and_observation_space_matches(combat_state):
    fake = FakeBridge(combat_state)
    env = IsaacEnv(fake,frames=4,max_steps=1,observation_profile="combat_history_v3",timing_profile="physical_v1")
    initial,_ = env.reset()
    fake.state["episode_frame"] += 4
    final,_,terminated,truncated,info = env.step([2,1,0])
    assert truncated and not terminated
    assert info["history_valid"] == 2
    frozen = deepcopy(final)
    reset,_ = env.reset()
    assert env.observation_space.contains(initial) and env.observation_space.contains(final)
    assert not rows(reset)[1:].any()
    np.testing.assert_array_equal(final["vector"],frozen["vector"])


def test_schema_failure_and_bounded_overflow_are_explicit(combat_state):
    bad = deepcopy(combat_state)
    del bad["combat_schema"]
    with pytest.raises(BridgeError,match="requires native"):
        append(ObservationHistory(),bad)
    for i in range(40):
        entity(combat_state,i,350+i)
    # A beam touching the player outranks point hazards despite its far origin.
    entity(combat_state,100,0,7,endpoint=[600,280],samples=[[0,280],[600,280]])
    h = ObservationHistory()
    obs = append(h,combat_state)
    assert h.stats["omitted_hazards"] == 9
    assert hazards(obs)[0,10] == 1
    combat_state["combat_entities"][0]["id"] = 999
    with pytest.raises(BridgeError,match="identity"):
        append(h,combat_state)


def test_migration_preserves_policy_and_adam_then_learns_new_inputs(combat_state):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    old = ActorCritic()
    adam = torch.optim.Adam(old.parameters(),lr=3e-4,eps=1e-5)
    obs = encode(combat_state,Counter(),"terrain_v2",.5)
    logits,value = old(as_tensor(obs))
    (logits.square().sum()+value.square().sum()).backward()
    adam.step()
    saved = dict(architecture=1,observation_profile="terrain_v2",model=old.state_dict(),optimizer=adam.state_dict())
    model,optimizer = load_policy(saved,observation_profile="combat_history_v3",with_optimizer=True)
    rich = append(ObservationHistory(),combat_state)
    expected = old(as_tensor(obs))
    actual = model(as_tensor(rich))
    for a,b in zip(expected,actual):
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
    ids = saved["optimizer"]["param_groups"][0]["params"]
    migrated = optimizer.state_dict()
    for identity,(name,param) in zip(ids,model.named_parameters()):
        for key,original in saved["optimizer"]["state"][identity].items():
            expanded = migrated["state"][identity][key]
            if key != "step" and name in ("spatial.0.weight","features.0.weight"):
                torch.testing.assert_close(expanded[:,:original.shape[1]],original,rtol=0,atol=0)
                assert not expanded[:,original.shape[1]:].count_nonzero()
            else:
                torch.testing.assert_close(expanded,original,rtol=0,atol=0)
    rollout = {k:[] for k in ("obs","actions","log_probs","values","next_values","rewards","terminated","ended")}
    for i in range(4):
        combat_state["episode_frame"] += 4
        rich = append(ObservationHistory(),combat_state)
        with torch.no_grad():
            action,logp,_,value = model.act(as_tensor(rich))
        transition = dict(obs=rich,actions=action[0].numpy(),log_probs=float(logp[0]),values=float(value[0]),
            next_values=0.,rewards=float(i+1),terminated=i == 3,ended=i == 3)
        for key,val in transition.items():
            rollout[key].append(val)
    report = optimize(model,optimizer,rollout,epochs=1,batch_size=4,entropy_coef=.002)
    assert all(np.isfinite(v) for v in report.values())
    assert model.features[0].weight[:,VECTOR_SIZE:].count_nonzero() > 0
    rich_saved = dict(architecture=2,observation_profile="combat_history_v3",observation_layout=model.observation_layout,
        model=model.state_dict(),optimizer=optimizer.state_dict())
    resumed,_ = load_policy(rich_saved,with_optimizer=True)
    for a,b in zip(model(as_tensor(rich)),resumed(as_tensor(rich))):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    with pytest.raises(ValueError,match="new run"):
        resolve_observation_profile("combat_history_v3",saved,True)
    with pytest.raises(ValueError,match="migration"):
        load_policy(rich_saved,observation_profile="terrain_v2")
    assert observation_manifest("combat_history_v3")["architecture"] == 2


@pytest.mark.parametrize("module",[train,train_vector])
def test_real_trainers_fork_and_resume_history_without_other_changes(tmp_path,monkeypatch,combat_state,module):
    torch.set_num_threads(2)
    model,optimizer = load_policy(with_optimizer=True)
    sum(p.square().sum() for p in model.parameters()).backward()
    optimizer.step()
    parent = tmp_path/"baseline.pt"
    control = ControlTiming(4,"physical_v1")
    schedule = dict(rollout=256,batch_size=256,max_episode_steps=6750,idle_limit=900)
    checkpoint(parent,model,optimizer,dict(architecture=1,steps=100,episodes=2,updates=1,
        recent=[dict(r=99.,success=False)],losses=dict(loss=99.),training_seeds=["OLD"],
        numpy_rng=np.random.get_state(),frames=4,timing_profile="physical_v1",entropy_coef=.002,
        control_timing=control.manifest(),training_schedule=schedule,native_frames=400,
        native_frames_origin_steps=0,reward_profile="completion_v4",observation_profile="terrain_v2"))
    digest = sha256(parent)
    saved = torch.load(parent,weights_only=False)
    combat_state["damage_signal"] = "hp_delta_v1"
    factory = lambda *a,**kw: AdvancingBridge(deepcopy(combat_state))
    monkeypatch.setattr(vector,"Bridge",factory)
    monkeypatch.setattr(train,"Bridge",factory)
    snapshots = []
    def capture(*a,**kw):
        checkpoint(*a,**kw)
        snapshots.append(torch.load(a[0],weights_only=False))
    monkeypatch.setattr(module,"checkpoint",capture)
    run = tmp_path/"new"
    ports = ["--ports","18000","18001","18002","18003","18004","18005"] if module is train_vector else []
    args = ["trainer","--run",str(run),"--resume",str(parent),"--steps","106",
            "--observation-profile","combat_history_v3",*ports]
    monkeypatch.setattr(sys,"argv",args)
    module.main()
    initial,trained = snapshots[0],snapshots[-1]
    expected_model,expected_adam = load_policy(saved,observation_profile="combat_history_v3",with_optimizer=True)
    assert_exact(initial["model"],expected_model.state_dict())
    assert_exact(initial["optimizer"],expected_adam.state_dict())
    for key in ("steps","updates","episodes","native_frames","torch_rng","numpy_rng","frames","entropy_coef",
                "control_timing","training_schedule","reward_profile","timing_profile"):
        assert_exact(initial[key],saved[key])
    assert initial["architecture"] == 2
    assert initial["losses"] == {} and initial["recent"] == []
    assert trained["steps"] == 106 and trained["updates"] == 2
    assert trained["model"]["features.0.weight"][:,VECTOR_SIZE:].count_nonzero() > 0
    config = json.loads((run/"config.json").read_text())
    assert config["observation_layout"] == observation_manifest("combat_history_v3")
    if module is train_vector:
        assert len(config["ports"]) == 6
    monkeypatch.setattr(sys,"argv",["trainer","--run",str(run),"--resume",str(run/"latest.pt"),"--steps","112",*ports])
    module.main()
    assert snapshots[-1]["observation_profile"] == "combat_history_v3"
    assert snapshots[-1]["steps"] == 112
    assert sha256(parent) == digest
