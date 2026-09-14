"""Reward ablation and real PPO fork/resume tests; no native game connections."""
from copy import deepcopy
from dataclasses import asdict
import json
import sys

import numpy as np
import pytest
import torch

from isaac_rl import train, train_vector, vector
from isaac_rl.bridge import BridgeError
from isaac_rl.env import IsaacEnv
from isaac_rl.ppo import ActorCritic
from isaac_rl.rewards import PROFILES, profile_manifest
from isaac_rl.storage import checkpoint, sha256
from isaac_rl.timing import ControlTiming
from test_core import state, FakeBridge


class AdvancingBridge(FakeBridge):
    def request(self, op, **fields):
        if op == "step":
            self.state["episode_frame"] += fields["frames"]
            self.state["frame"] += fields["frames"]
        return deepcopy(self.state)


def test_only_partial_combat_payments_are_removed():
    old, new = map(lambda name: asdict(PROFILES[name]), ("confirmed_v3", "completion_v4"))
    assert {k for k in old if old[k] != new[k]} == {"damage", "kill"}
    assert (old["damage"], old["kill"]) == (.2, .5)
    assert (new["damage"], new["kill"]) == (0., 0.)
    control = ControlTiming(4, "physical_v1")
    manifest = profile_manifest("completion_v4", control)
    assert manifest["time"] == -.005 and manifest["gamma"] == control.gamma
    assert manifest["clear"] == 10 and manifest["boss"] == 100


def test_same_transitions_keep_observations_endings_idle_and_all_other_terms(state):
    state["damage_signal"] = "hp_delta_v1"
    state["room"].update(clear=False, enemies=3)
    state["cleared"] = {}
    bridges = [AdvancingBridge(deepcopy(state)) for _ in range(2)]
    envs = [IsaacEnv(bridge, frames=4, timing_profile="physical_v1", idle_limit=2,
                    observation_profile="terrain_v2", reward_profile=profile)
            for bridge, profile in zip(bridges, ["confirmed_v3", "completion_v4"])]
    for env in envs:
        env.reset()
    try:
        for step in range(5):
            for bridge in bridges:
                if step == 0:  # A 3.5-damage / one-health-unit exchange, no kill/clear.
                    bridge.state["events"].update(damage_dealt=3.5, damage_taken=1)
                    bridge.state["player"]["hearts"] -= 1
                if step == 1:  # A kill, but two enemies remain.
                    bridge.state["events"].update(damage_dealt=7, kills=1)
                    bridge.state["room"]["enemies"] = 2
                if step == 2:
                    bridge.state["events"].update(damage_dealt=14, kills=3)
                    bridge.state["room"].update(clear=True, enemies=0)
                if step == 4:
                    bridge.state["terminal"] = True
                    bridge.state["player"]["dead"] = True
            old, new = [env.step([0, 1, 0]) for env in envs]
            for key in old[0]:
                np.testing.assert_array_equal(old[0][key], new[0][key])
            assert old[2:4] == new[2:4]
            before, after = old[4]["reward_components"], new[4]["reward_components"]
            assert after["damage"] == after["kills"] == 0
            for key in before.keys() - {"damage", "kills"}:
                assert before[key] == after[key]
            assert old[1] - new[1] == pytest.approx(before["damage"] + before["kills"])
            assert old[4]["idle_steps"] == new[4]["idle_steps"]
            if step == 0:
                assert old[1] == pytest.approx(.195)
                assert new[1] == pytest.approx(-.505)
            if step == 1:
                assert after["room_clear"] == 0
                assert new[4]["kills"] == 1  # Still measured, not paid.
            if step == 2:
                assert after["room_clear"] == 10 and new[4]["combat_clears"] == 1
            if step == 3:
                assert after["room_clear"] == 0
            if step == 4:
                assert new[2] and not new[3] and after["death"] == -5
                episode = new[4]["episode"]
                assert episode["damage_dealt"] == 14 and episode["kills"] == 3
                assert episode["reason"] == "death" and not episode["success"]
                assert episode["reward_profile"] == "completion_v4"
    finally:
        for env in envs:
            env.close()


def test_boss_reward_and_native_accounting_requirement_remain(state):
    bridge = AdvancingBridge(state)
    env = IsaacEnv(bridge, frames=4, timing_profile="physical_v1", reward_profile="completion_v4")
    with pytest.raises(BridgeError, match="completion_v4 requires.*resolved HP-delta"):
        env.reset()
    state["damage_signal"] = "hp_delta_v1"
    state["room"].update(type=5, clear=False, enemies=1)
    state["cleared"] = {}
    env.reset()
    state.update(terminal=True, success=True, boss_seen=True, boss_defeated=True)
    state["room"].update(clear=True, enemies=0)
    state["events"].update(damage_dealt=100, kills=1)
    _, reward, terminated, truncated, info = env.step([0, 1, 0])
    assert terminated and not truncated and reward == pytest.approx(109.995)
    assert info["reward_components"]["floor_clear"] == 100
    assert info["episode"]["reason"] == "boss_clear"
    env.close()


def assert_exact(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_exact(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert_exact(a, e)
    else:
        assert actual == expected


@pytest.mark.parametrize("module", [train, train_vector])
def test_training_fork_preserves_initial_weights_adam_rng_and_resumes_new_reward(tmp_path, monkeypatch, state, module):
    torch.set_num_threads(1)
    torch.manual_seed(93)
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    sum(p.square().sum() for p in model.parameters()).backward()
    optimizer.step()  # Nonempty optimizer state must survive the fork.
    control = ControlTiming(4, "physical_v1")
    schedule = dict(rollout=256, batch_size=256, max_episode_steps=6750, idle_limit=900)
    parent = tmp_path / "baseline.pt"
    checkpoint(parent, model, optimizer, dict(architecture=1, steps=100, episodes=1, updates=1,
        recent=[dict(r=999., success=False)], losses=dict(loss=999.), training_seeds=["PARENT"],
        numpy_rng=np.random.get_state(), frames=4, timing_profile="physical_v1", entropy_coef=.002,
        control_timing=control.manifest(), training_schedule=schedule, native_frames=400,
        native_frames_origin_steps=0, reward_profile="confirmed_v3", observation_profile="terrain_v2"))
    digest = sha256(parent)
    original = torch.load(parent, weights_only=False)
    state["damage_signal"] = "hp_delta_v1"
    state["room"].update(clear=False, enemies=1)
    state["cleared"] = {}
    def fake_bridge(*a, **kw):
        return AdvancingBridge(deepcopy(state))
    # Real environment, collector, reward and PPO code, but no sockets/native games.
    monkeypatch.setattr(vector, "Bridge", fake_bridge)
    monkeypatch.setattr(train, "Bridge", fake_bridge)
    run = tmp_path / "treatment"
    snapshots = []
    def checked_checkpoint(*a, **kw):
        checkpoint(*a, **kw)
        snapshots.append(torch.load(a[0], weights_only=False))
    monkeypatch.setattr(module, "checkpoint", checked_checkpoint)
    arguments = ["trainer", "--run", str(run), "--resume", str(parent),
                 "--reward-profile", "completion_v4", "--steps", "106"]
    if module is train_vector:
        arguments += ["--ports", "18000", "18001", "18002", "18003", "18004", "18005"]
    monkeypatch.setattr(sys, "argv", arguments)
    module.main()
    initial, trained = snapshots[0], snapshots[-1]
    for key in ("model", "optimizer", "torch_rng", "numpy_rng", "training_schedule",
                "frames", "timing_profile", "control_timing", "entropy_coef", "observation_profile"):
        assert_exact(initial[key], original[key])
    assert initial["recent"] == [] and initial["losses"] == {}
    assert initial["reward_profile"] == trained["reward_profile"] == "completion_v4"
    assert trained["steps"] == 106 and trained["native_frames"] == 424
    assert all(torch.isfinite(t).all() for t in trained["model"].values())
    assert any(not torch.equal(trained["model"][k], v) for k, v in original["model"].items())
    config = json.loads((run / "config.json").read_text())
    assert config["reward"] == profile_manifest("completion_v4", control)
    assert config["training_schedule"] == schedule
    assert trained["optimizer"]["param_groups"] == original["optimizer"]["param_groups"]
    if module is train_vector:
        assert len(config["ports"]) == 6
    # Normal resume inherits the treatment; an in-place reward change is rejected.
    resume = ["trainer", "--run", str(run), "--resume", str(run / "latest.pt"), "--steps", "112"]
    if module is train_vector:
        resume += arguments[-7:]
    monkeypatch.setattr(sys, "argv", resume)
    module.main()
    assert snapshots[-1]["reward_profile"] == "completion_v4"
    assert snapshots[-1]["steps"] == 112
    monkeypatch.setattr(sys, "argv", resume + ["--reward-profile", "confirmed_v3"])
    with pytest.raises(RuntimeError, match="new run directory"):
        module.main()
    assert sha256(parent) == digest
