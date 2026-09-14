from collections import Counter

import gymnasium as gym
import numpy as np

from .bridge import Bridge, BridgeError
from .observation import CHANNELS, HEIGHT, WIDTH, VECTOR_SIZE, encode, resolve_observation_profile
from .rewards import PROFILES, door_potential
from .timing import ControlTiming, NATIVE_HZ, REFERENCE_FRAMES


class IsaacEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, bridge=None, frames=8, max_steps=3375, idle_limit=450, reward_profile="legacy_v1", observation_profile="legacy_v1", timing_profile="legacy_v1"):
        self.control = ControlTiming(frames, timing_profile)
        self.observation_profile = resolve_observation_profile(observation_profile)
        self.reward_profile = reward_profile
        self.reward_config = PROFILES[reward_profile]
        self.bridge = bridge or Bridge()
        self.frames, self.max_steps, self.idle_limit = frames,max_steps,idle_limit
        self.action_space = gym.spaces.MultiDiscrete([9,5,4])
        self.observation_space = gym.spaces.Dict({
            "grid":gym.spaces.Box(0,1,(CHANNELS,HEIGHT,WIDTH),np.float32),
            "vector":gym.spaces.Box(-1,1,(VECTOR_SIZE,),np.float32),
        })
        self.state = None
        self.connected = False
        self.finished = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if not self.connected:
            self.bridge.connect()
            self.connected = True
        options = options or {}
        self.state = self.bridge.request("reset", seed=options.get("game_seed"))
        if self.reward_config.damage_signal == "hp_delta_v1" and self.state.get("damage_signal") != "hp_delta_v1":
            raise BridgeError(f"{self.reward_profile} requires a bridge with resolved HP-delta damage accounting")
        if options.get("game_seed") and self.state["seed"].replace(" ","") != options["game_seed"].replace(" ",""):
            raise BridgeError("Game ignored the requested seed; launch WITHOUT --set-stage")
        if self.state["stage"] != 1 or self.state["character"] != 0 or self.state["difficulty"] != 0:
            raise BridgeError("Training requires a normal-mode Isaac run starting on floor 1")
        self.cells = Counter()
        self.room_ids = {self.state["room"]["id"]}
        self.clear_ids = {int(k) for k,v in self.state["cleared"].items() if v}
        self.combat_rooms = {self.state["room"]["id"]} if self.state["room"]["enemies"] else set()
        self.combat_clear_ids = set()
        self.steps, self.episode_reward, self.idle = 0,0.0,0
        self.episode_native_frames = self.frame_delta = 0
        self.finished = False
        self._visit(self.state)
        return encode(self.state,self.cells,self.observation_profile,self.control.scale), self._info()

    def _visit(self, state):
        p = state["player"]
        key = (state["room"]["id"],int(p["x"]//40),int(p["y"]//40))
        self.cells[key] += 1
        return self.cells[key]

    def step(self, action):
        if self.finished:
            raise RuntimeError("reset() required before stepping a terminated episode")
        action = np.asarray(action,dtype=np.int64)
        if not self.action_space.contains(action):
            raise ValueError(f"Action outside MultiDiscrete([9,5,4]): {action}")
        before = self.state
        self.state = after = self.bridge.request("step",action=action.tolist(),frames=self.frames)
        if after["episode"] != before["episode"]:
            raise BridgeError("Unexpected game restart inside a transition")
        self.frame_delta = after["episode_frame"]-before["episode_frame"]
        if self.control.profile == "physical_v1" and not 1 <= self.frame_delta <= self.frames:
            raise BridgeError(f"Native action advanced {self.frame_delta} ticks; expected 1..{self.frames}")
        self.episode_native_frames += self.frame_delta
        self.steps += 1
        count = self._visit(after)
        new_room = after["room"]["id"] not in self.room_ids
        new_clear = after["room"]["clear"] and after["room"]["id"] not in self.clear_ids
        if after["room"]["enemies"] > 0:
            self.combat_rooms.add(after["room"]["id"])
        new_combat_clear = new_clear and after["room"]["id"] in self.combat_rooms
        if new_combat_clear:
            self.combat_clear_ids.add(after["room"]["id"])
        self.room_ids.add(after["room"]["id"])
        if new_clear:
            self.clear_ids.add(after["room"]["id"])
        damage = max(0,after["events"]["damage_dealt"]-before["events"]["damage_dealt"])
        hurt = max(0,after["events"]["damage_taken"]-before["events"]["damage_taken"])
        kills = max(0,after["events"]["kills"]-before["events"]["kills"])
        p,q = before["player"],after["player"]
        pickups = max(0,q["items"]-p["items"])*2 + max(0,q["hearts"]+q["soul"]-p["hearts"]-p["soul"])*0.3
        # Novelty is paid once per 40-unit cell per episode; room re-entry cannot farm it.
        cfg = self.reward_config
        # Death can end a hold early; charge only the elapsed native time.
        time_scale = self.frame_delta/REFERENCE_FRAMES if self.control.profile == "physical_v1" else 1.0
        clear_bonus = new_combat_clear if cfg.combat_clear_only else new_clear
        components = {
            "time":cfg.time*time_scale, "explore_cell":cfg.cell if count == 1 else 0,
            "explore_room":cfg.room if new_room else 0,
            "room_clear":cfg.clear if clear_bonus else 0,
            "damage":min(damage,200)*cfg.damage,"hurt":hurt*cfg.hurt,
            "kills":kills*cfg.kill,"pickups":pickups,
            "death":cfg.death if q["dead"] else 0,
            "floor_clear":cfg.boss if after["success"] else 0,
            "navigation":self.control.gamma*door_potential(after,cfg.door_potential)-door_potential(before,cfg.door_potential),
        }
        meaningful = new_room or new_clear or damage > 0 or pickups > 0 or kills > 0 or (cfg.cell_resets_idle and count == 1)
        self.idle = 0 if meaningful else self.idle+1
        terminated = bool(after["terminal"])
        truncated = not terminated and (self.steps >= self.max_steps or self.idle >= self.idle_limit or after["stage"] != 1)
        self.finished = terminated or truncated
        reward = float(sum(components.values()))
        self.episode_reward += reward
        info = self._info()
        info["reward_components"] = components
        if self.finished:
            info["episode"] = {"r":self.episode_reward,"l":self.steps,"success":after["success"],
                "native_frames":self.episode_native_frames,"simulated_seconds":self.episode_native_frames/NATIVE_HZ,
                "frames":self.frames,"timing_profile":self.control.profile,
                "seed":after["seed"],"rooms":len(self.room_ids),"clears":len(self.clear_ids),
                "damage_dealt":after["events"]["damage_dealt"],"damage_taken":after["events"]["damage_taken"],
                "damage_attempted":after["events"].get("damage_attempted",after["events"]["damage_dealt"]),
                "kills":after["events"]["kills"],
                "combat_clears":len(self.combat_clear_ids),"reward_profile":self.reward_profile,
                "observation_profile":self.observation_profile,
                "boss_seen":after["boss_seen"],"boss_defeated":after["boss_defeated"],
                "reason":"boss_clear" if after["success"] else "death" if q["dead"] else "idle" if self.idle >= self.idle_limit else "time_limit"}
        return encode(after,self.cells,self.observation_profile,self.control.scale),reward,terminated,truncated,info

    def _info(self):
        s = self.state
        return {"game_seed":s["seed"],"room":s["room"]["id"],"health":s["player"]["hearts"]+s["player"]["soul"],
            "frame_delta":self.frame_delta,"episode_native_frames":self.episode_native_frames,
            "rooms":len(self.room_ids),"clears":len(self.clear_ids),"success":s["success"],
            "boss_seen":s["boss_seen"],"game_frame":s["frame"],"game_episode":s["episode"],
            "kills":s["events"]["kills"],"damage_dealt":s["events"]["damage_dealt"],
            "combat_clears":len(self.combat_clear_ids),"idle_steps":self.idle,"reward_profile":self.reward_profile,
            "observation_profile":self.observation_profile}

    def close(self):
        self.bridge.close()
