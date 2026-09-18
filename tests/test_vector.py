import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from isaac_rl.ppo import compute_gae, ActorCritic, optimize
from isaac_rl.vector import ParallelIsaac
from isaac_rl.train_vector import rollout_length


@pytest.mark.parametrize("steps",[0,1_000_002,10_000_000])
def test_uncapped_detached_training_keeps_full_rollouts(steps):
    assert rollout_length(steps,0,128,6) == 128


def test_explicit_finite_budget_still_finishes_without_empty_update():
    assert rollout_length(0,1000,128,6) == 128
    assert rollout_length(996,1000,128,6) == 1
    assert rollout_length(1002,1000,128,6) == 0
    with pytest.raises(ValueError):
        rollout_length(0,-1,128,6)


def test_gae_does_not_mix_games_and_bootstraps_each_boundary():
    rewards = [[1,10],[2,20]]
    values = [[0,0],[0,0]]
    next_values = [[0,5],[0,0]]
    # Game B truncates at step 0, then restarts. Game A ends at step 1.
    terminated = [[False,False],[True,True]]
    ended = [[False,True],[True,True]]
    _,returns = compute_gae(rewards,values,next_values,terminated,ended,gamma=1,lam=1)
    np.testing.assert_allclose(returns,[[3,15],[2,20]])


def observation(value):
    return {"grid":np.full((10,16,28),value,np.float32),"vector":np.full(392,value,np.float32)}


@pytest.mark.parametrize("game_count",[3,6])
def test_parallel_steps_overlap_and_preserve_final_observation_until_bootstrap(game_count):
    barrier = threading.Barrier(game_count,timeout=3)
    class Game:
        def __init__(self,index):
            self.index = index
        def step(self,action):
            barrier.wait()  # Would fail if the collector serialized games.
            return observation(self.index),float(action[0]),False,self.index==1,{"index":self.index}
        def reset(self):
            return observation(99),{"reset":self.index}
        def close(self):
            pass
    collector = ParallelIsaac.__new__(ParallelIsaac)
    collector.envs = [Game(i) for i in range(game_count)]
    collector.pool = ThreadPoolExecutor(max_workers=game_count)
    try:
        obs,rewards,terminated,truncated,infos = collector.step([[i+1,0,0] for i in range(game_count)])
        np.testing.assert_array_equal(rewards,np.arange(1,game_count+1))
        np.testing.assert_array_equal(obs["vector"][:,0],np.arange(game_count))
        assert not terminated.any() and truncated.tolist() == [i==1 for i in range(game_count)]
        final_obs = obs["vector"].copy()
        resets = collector.reset_done(obs,terminated|truncated)
        assert resets == {1:{"reset":1}}
        np.testing.assert_array_equal(final_obs[:,0],np.arange(game_count))
        np.testing.assert_array_equal(obs["vector"][:,0],[99 if i==1 else i for i in range(game_count)])
    finally:
        collector.close()


@pytest.mark.parametrize("game_count",[3,6])
def test_vector_ppo_update_uses_all_environment_transitions(game_count):
    torch.set_num_threads(1)
    torch.manual_seed(6)
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)
    obs = {"grid":np.zeros((game_count,10,16,28),np.float32),"vector":np.zeros((game_count,392),np.float32)}
    rollout = {key:[] for key in ["obs","actions","log_probs","values","rewards","next_values","terminated","ended"]}
    for i in range(8):
        with torch.no_grad():
            action,log_probs,_,values = model.act({key:torch.from_numpy(value) for key,value in obs.items()})
        ends = np.array([i==7 if j%2==0 else i in (3,7) for j in range(game_count)])
        record = dict(obs=obs,actions=action.numpy(),log_probs=log_probs.numpy(),values=values.numpy(),
            rewards=np.array([i%(j+2) for j in range(game_count)],np.float32),next_values=np.zeros(game_count,np.float32),
            terminated=ends,ended=ends)
        for key,value in record.items():
            rollout[key].append(value)
    before = model.actor.weight.detach().clone()
    metrics = optimize(model,optimizer,rollout,epochs=2,batch_size=8)
    assert all(np.isfinite(v) for v in metrics.values())
    assert not torch.equal(before,model.actor.weight)
