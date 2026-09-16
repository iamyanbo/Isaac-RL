"""Regression coverage for the native seed-collision training crash."""
from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isaac_rl import storage, train, train_vector, vector
from isaac_rl.bridge import BridgeError
from isaac_rl.history import ObservationHistory, validate
from isaac_rl.observation import observation_manifest
from isaac_rl.ppo import load_policy
from test_core import state
from test_completion_reward import AdvancingBridge, assert_exact
from test_observation_history import combat_state, entity, append, hazards


def test_shared_seeds_remain_distinct_and_follow_crossings(combat_state):
    a,_ = entity(combat_state,101,310,track_id=1)
    b,_ = entity(combat_state,101,400,track_id=2)
    history = ObservationHistory()
    first = append(history,combat_state)
    assert hazards(first)[:,0].sum() == 2
    a[0],b[0] = 500,330
    combat_state["episode_frame"] += 4
    result = append(history,combat_state,[2,1,0])
    assert hazards(result)[0,1] == pytest.approx(10/560)
    assert hazards(result,1)[0,1] == pytest.approx(80/560)
    assert hazards(result,1)[1,1] == pytest.approx(-10/560)
    # UID values do not become numerical network inputs.
    renamed = deepcopy(combat_state)
    for x in renamed["combat_entities"]:
        x["track_id"] += 9999999
    original,renamed = [append(ObservationHistory(),s) for s in (combat_state,renamed)]
    for key in original:
        np.testing.assert_array_equal(original[key],renamed[key])


def test_despawn_and_same_seed_replacement_are_not_false_history_matches(combat_state):
    entity(combat_state,101,310,track_id=1)
    history = ObservationHistory()
    append(history,combat_state)
    combat_state["entities"].clear()
    combat_state["combat_entities"].clear()
    entity(combat_state,101,330,track_id=2)
    combat_state["episode_frame"] += 4
    result = append(history,combat_state)
    np.testing.assert_array_equal(hazards(result)[:2,0],[1,0])
    np.testing.assert_array_equal(hazards(result,1)[:2,0],[0,1])


@pytest.mark.parametrize("bad",[0,-1,True,1.5,None])
def test_invalid_track_id_rejected(combat_state,bad):
    entity(combat_state,1,300,track_id=bad)
    with pytest.raises(BridgeError,match="track_id"):
        validate(combat_state)


def test_duplicate_track_id_and_seed_mismatch_have_distinct_errors(combat_state):
    entity(combat_state,101,310,track_id=1)
    _,sidecar = entity(combat_state,202,330,track_id=1)
    with pytest.raises(BridgeError,match="duplicate entity track_id"):
        validate(combat_state)
    sidecar["track_id"] = 2
    sidecar["id"] = 999
    with pytest.raises(BridgeError,match="seed/sidecar identity mismatch"):
        validate(combat_state)
    sidecar["id"] = 202
    del sidecar["track_id"]
    with pytest.raises(BridgeError,match="track_id"):
        validate(combat_state)


def test_category_morph_does_not_read_wrong_sidecar_fields(combat_state):
    e,x = entity(combat_state,1,300)
    del x["npc_state"],x["state_frame"],x["animation_frame"]
    history = ObservationHistory()
    append(history,combat_state)
    e[4] = 1
    x.update(npc_state=1,state_frame=1,animation_frame=1)
    combat_state["episode_frame"] += 4
    result = append(history,combat_state)
    assert np.isfinite(result["vector"]).all()
    assert not hazards(result)[:,0].any()
    assert hazards(result,1)[0,0] == 1


def previous_layout():
    return dict(observation_manifest("combat_history_v3"),native_schema="combat_v1",
        identity="InitSeed-aligned entity tracks within each sample; no seed values input")


def test_exact_old_manifest_upgrades_without_touching_model_or_adam():
    torch.set_num_threads(2)
    model,adam = load_policy(observation_profile="combat_history_v3",with_optimizer=True)
    sum(p.square().sum() for p in model.parameters()).backward()
    adam.step()
    saved = dict(architecture=2,observation_profile="combat_history_v3",
        observation_layout=previous_layout(),model=model.state_dict(),optimizer=adam.state_dict())
    resumed,optimizer = load_policy(saved,with_optimizer=True)
    assert_exact(resumed.state_dict(),saved["model"])
    assert_exact(optimizer.state_dict(),saved["optimizer"])
    assert resumed.observation_layout == observation_manifest("combat_history_v3")
    for key in ("history","vector_size","identity","native_schema"):
        broken = deepcopy(saved)
        broken["observation_layout"][key] = "unknown"
        with pytest.raises(ValueError,match="layout"):
            load_policy(broken)


@pytest.mark.parametrize("module",[train,train_vector])
def test_actual_trainer_preserves_failing_raw_frame_not_unoptimized_checkpoint(tmp_path,monkeypatch,combat_state,module):
    class BadFrameBridge(AdvancingBridge):
        def request(self,op,**kwargs):
            result = super().request(op,**kwargs)
            if op == "step":
                result["combat_schema"] = "invalid_failure_marker"
            return result
    factory = lambda *a,**kw: BadFrameBridge(deepcopy(combat_state))
    combat_state["damage_signal"] = "hp_delta_v1"
    monkeypatch.setattr(vector,"Bridge",factory)
    monkeypatch.setattr(train,"Bridge",factory)
    ports = ["--ports","18000","18001","18002","18003","18004","18005"] if module is train_vector else []
    monkeypatch.setattr(sys,"argv",["trainer","--run",str(tmp_path),"--steps","6",
        "--observation-profile","combat_history_v3","--reward-profile","completion_v4",*ports])
    with pytest.raises(BridgeError,match="combat_v2"):
        module.main()
    status = json.loads((tmp_path/"status.json").read_text())
    failed = json.loads(open(status["failure_states"]).read())
    assert len(failed) == (6 if module is train_vector else 1)
    assert all(s["combat_schema"] == "invalid_failure_marker" for s in failed)
    assert status["cleanup_complete"] and status["exit_code"] == 1
    saved = torch.load(tmp_path/"latest.pt",weights_only=False)
    assert saved["steps"] == 0 and saved["updates"] == 0


def test_capture_failure_does_not_mask_training_error(monkeypatch,tmp_path):
    def fail(*args):
        raise OSError("disk full")
    monkeypatch.setattr(storage,"atomic_json",fail)
    result = storage.capture_failure_states(tmp_path/"failed.json",[SimpleNamespace(state={})])
    assert result == {"failure_capture_error":"OSError: disk full"}
