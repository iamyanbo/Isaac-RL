"""Collect native, independent game steps concurrently, preserving Gym semantics."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .bridge import Bridge
from .env import IsaacEnv


def stack_obs(observations):
    return {key:np.stack([obs[key] for obs in observations]) for key in observations[0]}


class ParallelIsaac:
    def __init__(self,ports,frames=8,max_steps=3375,idle_limit=450,reward_profile="legacy_v1",observation_profile="legacy_v1"):
        if not ports or len(set(ports)) != len(ports):
            raise ValueError("Provide unique worker ports")
        self.ports = list(ports)
        self.envs = []
        try:
            for port in ports:
                self.envs.append(IsaacEnv(Bridge(port=port),frames,max_steps,idle_limit,reward_profile,observation_profile))
        except Exception:
            for env in self.envs:
                env.close()
            raise
        self.pool = ThreadPoolExecutor(max_workers=len(ports),thread_name_prefix="isaac-worker")

    def _all(self,functions):
        futures = [self.pool.submit(fn) for fn in functions]
        results,failures = [],[]
        for future in futures:
            try:
                results.append(future.result())
            except BaseException as error:
                failures.append(error)
                results.append(None)
        if failures:
            raise failures[0]
        return results

    def reset(self):
        results = self._all([env.reset for env in self.envs])
        return stack_obs([r[0] for r in results]),[r[1] for r in results]

    def step(self,actions):
        if len(actions) != len(self.envs):
            raise ValueError("One action per game is required")
        results = self._all([lambda env=env,action=action:env.step(action) for env,action in zip(self.envs,actions)])
        observations,rewards,terminated,truncated,infos = zip(*results)
        return stack_obs(observations),np.asarray(rewards,np.float32),np.asarray(terminated,bool),np.asarray(truncated,bool),list(infos)

    def reset_done(self,obs,ended):
        indices = np.flatnonzero(ended)
        results = self._all([self.envs[i].reset for i in indices])
        infos = {}
        for i,(new_obs,info) in zip(indices,results):
            for key in obs:
                obs[key][i] = new_obs[key]
            infos[int(i)] = info
        return infos

    def close(self):
        try:
            self._all([env.close for env in self.envs])
        finally:
            self.pool.shutdown(wait=True)
