"""Lifecycle tests use fake games only; never touch running Isaac processes."""
import json
import socket
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from isaac_rl.bridge import Bridge
from isaac_rl.runtime import RunLease
from isaac_rl import train_vector


def test_close_does_not_wait_gameplay_timeout_for_unresponsive_game():
    bridge = Bridge(port=0,timeout=120)
    client, peer = socket.socketpair()
    bridge.client = client  # Peer deliberately never acknowledges shutdown.
    t = time.perf_counter()
    try:
        bridge.close()
        assert time.perf_counter()-t < 2
        assert bridge.client is None
        assert bridge.listener.fileno() == -1
    finally:
        peer.close()


@pytest.mark.parametrize("fail",[False,True])
def test_vector_session_has_finite_or_error_terminal_and_releases_lease(tmp_path,monkeypatch,fail):
    class FakeGames:
        closed = False
        def __init__(self,ports,*args):
            self.envs = [SimpleNamespace(state={},episode_reward=0.,steps=0) for _ in ports]
            self.n = len(ports)
        def observation(self):
            return dict(grid=np.zeros((self.n,10,16,28),np.float32),vector=np.zeros((self.n,392),np.float32))
        def reset(self):
            if fail:
                raise ConnectionResetError("fake worker failed")
            return self.observation(),[dict(game_seed=f"FAKE {i}") for i in range(self.n)]
        def step(self,actions):
            return self.observation(),np.zeros(self.n,np.float32),np.zeros(self.n,bool),np.zeros(self.n,bool),[{} for _ in range(self.n)]
        def reset_done(self,obs,ended):
            return {}
        def close(self):
            FakeGames.closed = True
    monkeypatch.setattr(train_vector,"ParallelIsaac",FakeGames)
    monkeypatch.setattr(sys,"argv",["train_vector","--run",str(tmp_path),"--ports","18000","18001","--steps","4","--rollout","2"])
    if fail:
        with pytest.raises(ConnectionResetError):
            train_vector.main()
    else:
        train_vector.main()
        update = json.loads((tmp_path/"updates.jsonl").read_text().splitlines()[0])
        assert update["steps"] == 4 and update["updates"] == 1
        assert update["rollout_sps"] > 0
        assert {"collection_s","inference_s","update_s","save_s","wall_s"} <= update["timing"].keys()
        assert all(v >= 0 for v in update["timing"].values())
    state = json.loads((tmp_path/"status.json").read_text())
    assert state["cleanup_complete"] and FakeGames.closed
    assert state["exit_code"] == (1 if fail else 0)
    assert state["status"] == ("error" if fail else "stopped")
    assert state["exit_reason"] == ("exception" if fail else "step_budget")
    assert state["session_id"] and state["ended_at"]
    with RunLease(tmp_path):
        pass
