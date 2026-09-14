from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from isaac_rl.bridge import BridgeError
from isaac_rl.env import IsaacEnv
from isaac_rl.observation import encode
from isaac_rl.ppo import compute_gae
from isaac_rl.rewards import door_potential, profile_manifest
from isaac_rl.timing import ControlTiming, checkpoint_timing, configure_training, configure_evaluation
from test_core import state, FakeBridge
from test_lua_events import event_runtime


def arguments(**changes):
    values = dict(frames=None,timing_profile=None,rollout=None,batch_size=None,max_episode_steps=None,idle_limit=None)
    values.update(changes)
    return SimpleNamespace(**values)


def test_physical_conversion_preserves_time_quantities_and_update_exposure():
    args = arguments(frames=4,timing_profile="physical_v1")
    c,schedule,changed = configure_training(args,{"frames":8})
    assert changed
    assert schedule == dict(rollout=256,batch_size=256,max_episode_steps=6750,idle_limit=900)
    assert c.gamma**2 == pytest.approx(.995)
    assert c.lam**2 == pytest.approx(.95)
    assert (c.gamma*c.lam)**2 == pytest.approx(.995*.95)
    assert schedule["rollout"]*c.frames == 128*8
    assert schedule["rollout"]*6/schedule["batch_size"] == 128*6/128
    assert schedule["max_episode_steps"]*c.frames/30 == 900
    assert schedule["idle_limit"]*c.frames/30 == 120
    before,after = profile_manifest("confirmed_v3"),profile_manifest("confirmed_v3",c)
    assert 2*after["time"] == before["time"]
    for key in before.keys()-{"time","gamma"}:
        assert before[key] == after[key]


def test_legacy_semantics_and_checkpoint_inheritance_are_explicit():
    assert checkpoint_timing({}) == ControlTiming()
    assert ControlTiming(4).gamma == .995  # Historical four-frame checkpoints remain legacy.
    args = arguments()
    c,schedule,changed = configure_training(args)
    assert not changed and c.frames == 8 and args.rollout == 128 and args.batch_size == 128
    saved = dict(frames=4,timing_profile="physical_v1",training_schedule=dict(
        rollout=300,batch_size=200,max_episode_steps=5000,idle_limit=800))
    c,schedule,changed = configure_training(arguments(),saved,existing=True)
    assert c == ControlTiming(4,"physical_v1") and not changed
    assert schedule == saved["training_schedule"]
    _,s,_ = configure_training(arguments(),{"frames":8},existing=True,previous_config={"frames":8,"rollout":64})
    assert s["rollout"] == 64


def test_timing_change_requires_fork_and_custom_parent_schedule_converts():
    saved = dict(frames=8,training_schedule=dict(rollout=64,batch_size=32,max_episode_steps=1000,idle_limit=100))
    with pytest.raises(ValueError,match="new run"):
        configure_training(arguments(frames=4,timing_profile="physical_v1"),saved,existing=True)
    _,s,_ = configure_training(arguments(frames=4,timing_profile="physical_v1"),saved)
    assert s == dict(rollout=128,batch_size=64,max_episode_steps=2000,idle_limit=200)
    with pytest.raises(ValueError,match="existing run"):
        configure_training(arguments(),{"frames":8},existing=True,previous_config={"frames":4})


@pytest.mark.parametrize("frames",[0,-1,31,2.5,True])
def test_invalid_control_hold_rejected_before_bridge(frames):
    with pytest.raises(ValueError):
        ControlTiming(frames,"physical_v1")


def test_unknown_profile_and_invalid_schedule_rejected():
    with pytest.raises(ValueError):
        ControlTiming(4,"unknown")
    with pytest.raises(ValueError):
        configure_training(arguments(idle_limit=0))


def test_evaluation_preserves_physical_episode_and_diagnostic_idle_budgets():
    args = arguments()
    control = configure_evaluation(args,{"frames":4,"timing_profile":"physical_v1"})
    assert control.frames == 4
    assert args.max_episode_steps == 6750 and args.idle_limit == 1800
    explicit = arguments(idle_limit=123)
    configure_evaluation(explicit,{"frames":4,"timing_profile":"physical_v1"})
    assert explicit.idle_limit == 123
    legacy = arguments()
    configure_evaluation(legacy,{"frames":8})
    assert legacy.max_episode_steps == 3375 and legacy.idle_limit == 900


def test_visit_map_intensity_uses_time_units_without_altering_other_features(state):
    old = encode(state,Counter({(84,8,7):4}))
    new = encode(state,Counter({(84,8,7):8}),visit_scale=.5)
    np.testing.assert_array_equal(old["grid"],new["grid"])
    np.testing.assert_array_equal(old["vector"],new["vector"])


def test_physical_env_sends_four_ticks_and_uses_matching_potential_discount(state):
    class AdvancingBridge(FakeBridge):
        def request(self,op,**fields):
            if op == "step":
                assert fields["frames"] == 4
                self.state["episode_frame"] += 4
                self.state["frame"] += 4
            return deepcopy(self.state)
    state["damage_signal"] = "hp_delta_v1"
    state["doors"] = [dict(slot=0,x=40,y=280,open=True,locked=False,type=1,visits=0)]
    env = IsaacEnv(AdvancingBridge(state),frames=4,reward_profile="confirmed_v3",timing_profile="physical_v1",idle_limit=2)
    env.reset()
    phi = door_potential(state,2)
    _,_,terminated,truncated,info = env.step([0,0,0])
    assert not terminated and not truncated
    assert info["frame_delta"] == 4
    assert info["reward_components"]["time"] == -.005
    assert info["reward_components"]["navigation"] == pytest.approx((env.control.gamma-1)*phi)
    assert info["reward_components"]["explore_cell"] == 0
    _,_,_,truncated,info = env.step([0,0,0])
    assert truncated and info["episode"]["native_frames"] == 8
    assert info["episode"]["simulated_seconds"] == pytest.approx(8/30)
    env.close()


def test_physical_env_rejects_unexpected_native_advancement(state):
    env = IsaacEnv(FakeBridge(state),frames=4,timing_profile="physical_v1")
    env.reset()
    with pytest.raises(BridgeError,match="advanced 0 ticks"):
        env.step([0,0,0])
    env.close()


def test_early_death_charges_only_elapsed_native_time(state):
    class DyingBridge(FakeBridge):
        def request(self,op,**fields):
            if op == "step":
                self.state["episode_frame"] += 1
                self.state["frame"] += 1
                self.state["player"]["dead"] = True
                self.state["terminal"] = True
            return deepcopy(self.state)
    env = IsaacEnv(DyingBridge(state),frames=4,timing_profile="physical_v1",reward_profile="balanced_v2")
    env.reset()
    _,_,terminated,_,info = env.step([0,0,0])
    assert terminated and info["frame_delta"] == 1
    assert info["reward_components"]["time"] == pytest.approx(-.01/8)
    env.close()


@pytest.mark.parametrize("frames",[1,2,4,8,30])
def test_actual_lua_step_replies_after_requested_update_count(frames):
    lua = event_runtime()
    lua.globals().requested_frames = frames
    lua.execute('''
        local tick = callbacks.MC_POST_UPDATE
        local receive = upvalue(tick, 'receiveCommand')
        local client = {settimeout=function() end, receive=function() return 'step' end}
        upvalue(receive, 'client', client)
        upvalue(receive, 'json', {decode=function() return {op='step',frames=requested_frames,action={0,0,0}} end})
        ButtonAction = setmetatable({}, {__index=function(_,key) return key end})
        Isaac.GetPlayer=function() return {IsDead=function() return false end} end
        upvalue(tick, 'observeDamage', function() end)
        local replies = 0
        upvalue(tick, 'reply', function() replies=replies+1; return false end)
        receive()
        for i=1,requested_frames-1 do
            tick()
            assert(replies == 0, 'Must not reply before the hold ends')
        end
        tick()
        assert(replies == 1, 'Exactly one response after requested native updates')
        tick()
        assert(replies == 1, 'No unsolicited second response')
    ''')


def test_gae_discount_is_consistent_with_shaping_and_preserves_boundaries():
    c = ControlTiming(4,"physical_v1")
    _,returns = compute_gae([1.,2.],[0.,0.],[5.,7.],[False,True],[True,True],gamma=c.gamma,lam=c.lam)
    np.testing.assert_allclose(returns,[1+c.gamma*5,2])
