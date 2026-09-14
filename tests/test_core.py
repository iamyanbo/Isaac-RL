from collections import Counter
from copy import deepcopy
import socket
import threading
import json

import numpy as np
import pytest
import torch

from isaac_rl.bridge import Bridge, BridgeError
from isaac_rl.env import IsaacEnv
from isaac_rl.evaluate import success_gate
from isaac_rl.observation import encode, VECTOR_SIZE, resolve_observation_profile
from isaac_rl.ppo import ActorCritic, compute_gae, optimize
from isaac_rl.rewards import GAMMA, door_potential


@pytest.fixture
def state():
    return {
        "type":"state","protocol":1,"id":1,"sequence":1,"episode":1,"frame":1,"episode_frame":1,
        "seed":"TEST SEED","stage":1,"stage_type":0,"character":0,"difficulty":0,
        "player":dict(x=320,y=280,vx=0,vy=0,hearts=6,soul=0,max_hearts=6,coins=0,keys=0,bombs=1,
            damage=3.5,speed=1,fire_delay=10,shot_speed=1,tear_range=260,charge=0,dead=False,items=0),
        "room":dict(id=84,type=1,clear=True,enemies=0,bounds=[40,120,600,400],visits=1),
        "entities":[],"doors":[],"grid":[],"visited":{"84":1},"cleared":{"84":True},
        "events":dict(damage_taken=0,damage_dealt=0,kills=0),"boss_seen":False,"boss_defeated":False,
        "success":False,"terminal":False,
    }


def test_gae_bootstraps_time_limit_but_not_death_or_next_episode():
    adv,ret = compute_gae([1,100],[2,3],[5,7],[False,True],[True,True],gamma=0.9,lam=1)
    np.testing.assert_allclose(ret,[5.5,100])
    np.testing.assert_allclose(adv,[3.5,97])


def test_success_requires_large_unique_boss_verified_sample():
    def record(i,win):
        return dict(seed=str(i),success=win,boss_seen=win,boss_defeated=win,reason="boss_clear" if win else "death")
    assert not success_gate([record(i,True) for i in range(20)])["passed"]
    assert success_gate([record(i,i<90) for i in range(100)])["passed"]
    assert not success_gate([record(i,i<89) for i in range(100)])["passed"]
    assert not success_gate([record(1,True)]*100)["passed"]
    bad = [record(i,True) for i in range(100)]
    bad[0]["boss_seen"] = False
    assert not success_gate(bad)["passed"]


def test_observation_handles_empty_and_crowded_rooms(state):
    state["entities"] = [[320+i,200,0,0,1,10,0,0,5,10,15,i] for i in range(60)]
    obs = encode(state,Counter({(84,8,7):100}))
    assert obs["grid"].shape == (10,16,28)
    assert obs["vector"].shape == (VECTOR_SIZE,)
    assert all(np.isfinite(value).all() for value in obs.values())
    assert all(value.max() <= 1 and value.min() >= -1 for value in obs.values())


def test_corrected_terrain_distinguishes_passable_doors_walls_and_real_hazards(state):
    state["grid"] = [[120,200,16,5,0],[200,200,15,4,0],[280,200,14,2,0],
                     [360,200,8,0,0],[440,200,9,0,0],[520,200,12,2,0]]
    modern = encode(state,Counter(),"terrain_v2")
    legacy = encode(state,Counter(),"legacy_v1")
    assert modern["grid"][4].sum() == 3  # wall, poop, TNT; not open doorway
    assert legacy["grid"][4].sum() == 4  # old mistake preserved for old checkpoints
    assert modern["grid"][6].sum() == 3  # spikes, toggling spikes, TNT
    assert legacy["grid"][6].sum() == 2  # old wall/poop labels
    np.testing.assert_array_equal(modern["vector"],legacy["vector"])
    np.testing.assert_array_equal(modern["grid"][[0,1,2,3,5,7,8,9]],legacy["grid"][[0,1,2,3,5,7,8,9]])


def test_checkpoint_observation_profile_is_explicit_and_legacy_compatible():
    assert resolve_observation_profile() == "terrain_v2"
    assert resolve_observation_profile(checkpoint={}) == "legacy_v1"
    assert resolve_observation_profile(checkpoint={"observation_profile":"terrain_v2"}) == "terrain_v2"
    assert resolve_observation_profile("terrain_v2",{},False) == "terrain_v2"
    with pytest.raises(ValueError,match="new run"):
        resolve_observation_profile("terrain_v2",{},True)
    with pytest.raises(ValueError,match="Unsupported"):
        resolve_observation_profile(checkpoint={"observation_profile":"unknown"})


class FakeBridge:
    def __init__(self,state):
        self.state = state
    def connect(self):
        return self.state
    def request(self,*_,**__):
        return deepcopy(self.state)
    def close(self):
        pass


def test_rewards_cannot_be_farmed_by_room_reentry(state):
    fake = FakeBridge(state)
    env = IsaacEnv(fake,idle_limit=100)
    env.reset()
    fake.state["room"]["id"] = 85
    _,reward1,_,_,info = env.step([0,0,0])
    assert info["reward_components"]["room_clear"] == 5
    fake.state["room"]["id"] = 84
    env.step([0,0,0])
    fake.state["room"]["id"] = 85
    _,reward2,_,_,info = env.step([0,0,0])
    assert reward1 > 7 and reward2 < 0
    assert info["reward_components"]["room_clear"] == 0


def test_environment_uses_selected_observation_profile_on_reset_and_step(state):
    state["grid"] = [[120,200,16,5,0]]
    fake = FakeBridge(state)
    env = IsaacEnv(fake,observation_profile="terrain_v2")
    obs,info = env.reset()
    assert info["observation_profile"] == "terrain_v2"
    assert obs["grid"][4].sum() == 0
    fake.state["grid"][0][3] = 4
    obs,_,_,_,info = env.step([0,0,0])
    assert obs["grid"][4].sum() == 1
    assert info["observation_profile"] == "terrain_v2"


def test_death_is_terminal_and_idle_is_truncation(state):
    fake = FakeBridge(state)
    env = IsaacEnv(fake,idle_limit=1)
    env.reset()
    _,_,terminated,truncated,_ = env.step([0,0,0])
    assert truncated and not terminated
    env.reset()
    fake.state["terminal"] = True
    fake.state["player"]["dead"] = True
    _,reward,terminated,truncated,info = env.step([0,0,0])
    assert terminated and not truncated and reward < -15
    assert not info["episode"]["success"]
    with pytest.raises(RuntimeError):
        env.step([0,0,0])


def test_kill_bonus_uses_counter_delta_once_and_is_reported_on_episode_end(state):
    fake = FakeBridge(state)
    env = IsaacEnv(fake,idle_limit=100)
    env.reset()
    fake.state["events"]["kills"] = 3
    _,reward,_,_,info = env.step([0,0,0])
    assert info["reward_components"]["kills"] == 0.75
    assert reward == pytest.approx(0.747)
    _,reward,_,_,info = env.step([0,0,0])
    assert info["reward_components"]["kills"] == 0
    assert reward == pytest.approx(-0.003)
    fake.state["terminal"] = True
    fake.state["player"]["dead"] = True
    _,_,terminated,_,info = env.step([0,0,0])
    assert terminated and info["episode"]["kills"] == 3


def test_balanced_reward_pays_only_observed_combat_clear_once(state):
    fake = FakeBridge(state)
    env = IsaacEnv(fake,reward_profile="balanced_v2")
    env.reset()
    fake.state["room"]["id"] = 85
    _,_,_,_,info = env.step([0,0,0])
    assert info["reward_components"]["explore_room"] == 2
    assert info["reward_components"]["room_clear"] == 0
    fake.state["room"].update(id=86,clear=False,enemies=2)
    env.step([0,0,0])
    fake.state["room"].update(clear=True,enemies=0)
    fake.state["events"].update(kills=2,damage_dealt=10)
    _,_,_,_,info = env.step([0,0,0])
    assert info["reward_components"]["room_clear"] == 10
    assert info["reward_components"]["damage"] == 2
    assert info["reward_components"]["kills"] == 1
    assert info["combat_clears"] == 1
    fake.state["room"]["id"] = 85
    env.step([0,0,0])
    fake.state["room"]["id"] = 86
    _,_,_,_,info = env.step([0,0,0])
    assert info["reward_components"]["room_clear"] == 0


def test_confirmed_damage_profile_requires_native_signal_and_retains_coefficients(state):
    from dataclasses import asdict
    from isaac_rl.rewards import PROFILES
    old,new = asdict(PROFILES["balanced_v2"]),asdict(PROFILES["confirmed_v3"])
    assert old.pop("damage_signal") == "attempted_v1"
    assert new.pop("damage_signal") == "hp_delta_v1"
    assert old == new
    fake = FakeBridge(state)
    env = IsaacEnv(fake,reward_profile="confirmed_v3")
    with pytest.raises(BridgeError,match="resolved HP-delta"):
        env.reset()
    fake.state["damage_signal"] = "hp_delta_v1"
    env.reset()
    fake.state["events"]["damage_dealt"] = 3.5
    _,_,_,_,info = env.step([0,0,0])
    assert info["reward_components"]["damage"] == pytest.approx(0.7)


def test_balanced_idle_clock_is_not_reset_by_wandering_to_new_cells(state):
    fake = FakeBridge(state)
    env = IsaacEnv(fake,idle_limit=2,reward_profile="balanced_v2")
    env.reset()
    fake.state["player"]["x"] += 40
    env.step([2,0,0])
    fake.state["player"]["x"] += 40
    _,_,terminated,truncated,info = env.step([2,0,0])
    assert truncated and not terminated
    assert info["episode"]["reason"] == "idle"
    assert info["idle_steps"] == 2


def test_navigation_potential_rewards_progress_but_not_roundtrip_or_death_farming(state):
    state["doors"] = [dict(x=600,y=280,open=True,locked=False,visits=0,slot=2,type=1,target=85)]
    near = deepcopy(state)
    near["player"]["x"] += 80
    start,closer = door_potential(state,2),door_potential(near,2)
    assert closer > start > 0
    forward = GAMMA*closer-start
    backward = GAMMA*start-closer
    assert forward > 0
    assert forward+GAMMA*backward <= 0
    dead = deepcopy(near)
    dead["terminal"] = True
    terminal = door_potential(dead,2)
    assert terminal == 0
    assert forward+GAMMA*(GAMMA*terminal-closer) == pytest.approx(-start)
    near["room"]["clear"] = False
    assert door_potential(near,2) == 0


def test_navigation_only_uses_open_unlocked_least_visited_destinations(state):
    state["doors"] = [dict(x=320,y=280,open=True,locked=False,visits=10),
                      dict(x=600,y=280,open=True,locked=False,visits=0)]
    actual = door_potential(state,2)
    state["doors"] = [state["doors"][1]]
    assert door_potential(state,2) == actual
    state["doors"][0]["locked"] = True
    assert door_potential(state,2) == 0


def test_bridge_fragmentation_and_stale_response(state):
    bridge = Bridge(port=0,timeout=2)
    port = bridge.listener.getsockname()[1]
    failures = []
    def game():
        try:
            with socket.create_connection(("127.0.0.1",port)) as client:
                raw = (json.dumps(state)+"\n").encode()
                client.sendall(raw[:10])
                client.sendall(raw[10:])
                command = json.loads(client.makefile("rb").readline())
                reply = dict(state,id=command["id"])
                client.sendall((json.dumps(reply)+"\n").encode())
        except BaseException as exc:
            failures.append(exc)
    thread = threading.Thread(target=game)
    thread.start()
    assert bridge.connect()["seed"] == state["seed"]
    with pytest.raises(BridgeError,match="Stale"):
        bridge.request("step",action=[0,0,0],frames=8)
    bridge.client.close()
    bridge.client = None
    bridge.close()
    thread.join(timeout=3)
    assert not failures and not thread.is_alive()


def test_ppo_update_changes_weights_with_finite_losses(state):
    torch.set_num_threads(1)
    torch.manual_seed(5)
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)
    obs = encode(state,Counter())
    from isaac_rl.ppo import as_tensor
    rollout = {key:[] for key in ["obs","actions","log_probs","values","rewards","next_values","terminated","ended"]}
    for i in range(16):
        with torch.no_grad():
            action,logprob,_,value = model.act(as_tensor(obs))
        record = dict(obs=obs,actions=action[0].numpy(),log_probs=logprob.item(),values=value.item(),
            rewards=float(i%4),next_values=0.,terminated=i==15,ended=i==15)
        for key,val in record.items():
            rollout[key].append(val)
    before = model.actor.weight.detach().clone()
    result = optimize(model,optimizer,rollout,epochs=2,batch_size=8)
    assert all(np.isfinite(v) for v in result.values())
    assert not torch.equal(before,model.actor.weight)
