"""Enriched current state and finite, identity-aligned observation/action history.

Each PPO transition owns its complete input, so shuffled transition minibatches
remain valid. No recurrent hidden states or cross-episode history are used.
"""
from collections import deque
from copy import deepcopy
import math

import numpy as np

from .bridge import BridgeError
from .observation import CHANNELS, HEIGHT, WIDTH, VECTOR_SIZE, encode

HISTORY = 4
PLAYER_SIZE, ENEMY_SIZE, HAZARD_SIZE = 12, 16, 46
ENEMIES, HAZARDS = 16, 32
FRAME_VECTOR_SIZE = VECTOR_SIZE + PLAYER_SIZE + ENEMIES*ENEMY_SIZE + HAZARDS*HAZARD_SIZE
HISTORY_VECTOR_SIZE = HISTORY*FRAME_VECTOR_SIZE + HISTORY*2 + (HISTORY-1)*18


def squash(value, scale):
    """Bounded continuous encoding without hard-clipping fast projectiles."""
    return value/(abs(value)+scale)


def validate(state):
    if state.get("combat_schema") != "combat_v1":
        raise BridgeError("combat_history_v3 requires native combat_v1 (bridge 0.1.4); do not silently pad missing fields")
    try:
        player = state["combat_player"]
        for key in ("size","size_multi","fire_cooldown","damage_cooldown_render_frames",
                    "invincible","can_shoot","can_fly","collision"):
            player[key]
        if len(state["entities"]) != len(state["combat_entities"]):
            raise ValueError("entity sidecar length mismatch")
        seen = set()
        for entity, extra in zip(state["entities"],state["combat_entities"]):
            if entity[11] != extra["id"] or entity[11] in seen:
                raise ValueError("entity identity mismatch or duplicate")
            seen.add(entity[11])
            for key in ("size_multi","collision","collision_damage"):
                extra[key]
            keys = {1:("npc_state","state_frame","animation_frame"),
                    2:("height","falling_speed","falling_accel"),4:("countdown",),
                    7:("endpoint","angle","radius","circle","sample","timeout","samples","sample_count","geometry_valid")}.get(entity[4],())
            for key in keys:
                extra[key]
            if entity[4] == 7 and extra["geometry_valid"] and len(extra["samples"]) != min(extra["sample_count"],8):
                raise ValueError("incomplete laser samples")
    except (KeyError,IndexError,TypeError,ValueError) as error:
        raise BridgeError(f"Invalid combat_v1 state: {error}") from error


def segment_distance(p, a, b):
    dx,dy = b[0]-a[0],b[1]-a[1]
    t = np.clip(((p[0]-a[0])*dx+(p[1]-a[1])*dy)/max(dx*dx+dy*dy,1e-9),0,1)
    return math.hypot(p[0]-a[0]-t*dx,p[1]-a[1]-t*dy)


def threat_distance(entity, extra, player):
    p = (player["x"],player["y"])
    if entity[4] != 7:
        return max(0,math.hypot(entity[0]-p[0],entity[1]-p[1])-entity[10])
    if not extra["geometry_valid"]:
        return 0  # Preserve unknown beams at the front of the finite input budget.
    if extra["circle"]:
        return max(0,abs(math.hypot(entity[0]-p[0],entity[1]-p[1])-extra["radius"])-entity[10])
    points = extra["samples"] or [entity[:2],extra["endpoint"]]
    if len(points) == 1:
        points = [entity[:2],points[0]]
    return max(0,min(segment_distance(p,a,b) for a,b in zip(points,points[1:]))-entity[10])


class ObservationHistory:
    def __init__(self):
        self.frames = deque(maxlen=HISTORY)
        self.stats = {}

    def reset(self):
        self.frames.clear()
        self.stats = {}

    def append(self, state, cells, visit_scale, action=None):
        validate(state)
        key = (state["episode"],state["stage"],state["room"]["id"])
        if self.frames and key != self.frames[0]["key"]:
            self.reset()
        if not self.frames:
            action = None
        elif state["episode_frame"] <= self.frames[0]["state"]["episode_frame"]:
            raise BridgeError("History requires increasing native episode ticks")
        self.frames.appendleft(dict(key=key,state=deepcopy(state),
            base=encode(state,cells,"terrain_v2",visit_scale),
            action=None if action is None else np.asarray(action,dtype=np.int64).copy()))
        return self._encode()

    def _encode(self):
        # Choose current threats first, then fill spare slots with disappeared
        # recent entities. The SAME identity order is used for every age in this
        # sample, even when distance order changed. IDs themselves are not input.
        tracks = {"enemies":[],"hazards":[]}
        maps = []
        current = self.frames[0]["state"]
        for frame in self.frames:
            s = frame["state"]
            mapping = {e[11]:(e,x) for e,x in zip(s["entities"],s["combat_entities"])}
            maps.append(mapping)
            for label,kinds,limit in (("enemies",(1,),ENEMIES),("hazards",(2,4,7),HAZARDS)):
                candidates = [(e,x) for e,x in mapping.values() if e[4] in kinds]
                candidates.sort(key=lambda pair:(threat_distance(*pair,current["player"]),pair[0][11]))
                for e,_ in candidates:
                    if e[11] not in tracks[label] and len(tracks[label]) < limit:
                        tracks[label].append(e[11])
        grid = np.zeros((HISTORY*CHANNELS,HEIGHT,WIDTH),np.float32)
        vector = np.zeros(HISTORY_VECTOR_SIZE,np.float32)
        frame_vectors = vector[:HISTORY*FRAME_VECTOR_SIZE].reshape(HISTORY,FRAME_VECTOR_SIZE)
        validity_age = vector[HISTORY*FRAME_VECTOR_SIZE:HISTORY*FRAME_VECTOR_SIZE+HISTORY*2].reshape(HISTORY,2)
        actions = vector[-(HISTORY-1)*18:].reshape(HISTORY-1,18)
        for age,(frame,mapping) in enumerate(zip(self.frames,maps)):
            s,p = frame["state"],frame["state"]["player"]
            left,top,right,bottom = s["room"]["bounds"]
            width,height = max(1,right-left),max(1,bottom-top)
            def relative(x,y):
                return [(x-p["x"])/width,(y-p["y"])/height]
            grid[age*CHANNELS:(age+1)*CHANNELS] = frame["base"]["grid"]
            row = frame_vectors[age]
            row[:VECTOR_SIZE] = frame["base"]["vector"]
            q = s["combat_player"]
            row[VECTOR_SIZE:VECTOR_SIZE+PLAYER_SIZE] = [squash(q["size"],50),
                *[squash(v,2) for v in q["size_multi"]],squash(q["fire_cooldown"],30),
                squash(q["damage_cooldown_render_frames"],120),float(q["invincible"]),
                float(q["can_shoot"]),float(q["can_fly"]),q["collision"]/4,
                squash(width,1200),squash(height,1200),squash(s["episode_frame"],27000)]
            enemies = row[VECTOR_SIZE+PLAYER_SIZE:VECTOR_SIZE+PLAYER_SIZE+ENEMIES*ENEMY_SIZE].reshape(ENEMIES,ENEMY_SIZE)
            hazards = row[-HAZARDS*HAZARD_SIZE:].reshape(HAZARDS,HAZARD_SIZE)
            for label,dest in (("enemies",enemies),("hazards",hazards)):
                for slot,identity in enumerate(tracks[label]):
                    if identity not in mapping:
                        continue
                    e,x = mapping[identity]
                    common = [1,*relative(e[0],e[1]),squash(e[2],15),squash(e[3],15),
                        squash(e[10],50),*[squash(v,2) for v in x["size_multi"]]]
                    category = [squash(e[5],1000),squash(e[6],100),squash(e[7],100)]
                    if label == "enemies":
                        dest[slot] = [*common,*category,e[8]/max(e[9],1),x["collision"]/4,
                            squash(x["npc_state"],20),squash(x["state_frame"],60),squash(x["animation_frame"],60)]
                    else:
                        laser = e[4] == 7
                        angle = math.radians(x["angle"]) if laser else 0
                        samples = np.zeros((8,2),np.float32)
                        for i,point in enumerate(x.get("samples",[])):
                            samples[i] = relative(*point)
                        dest[slot] = [*common,*[float(e[4] == kind) for kind in (2,4,7)],*category,
                            x["collision"]/4,squash(x.get("height",0),100),squash(x.get("falling_speed",0),10),
                            squash(x.get("falling_accel",0),1),float(x.get("circle",False)),float(x.get("sample",False)),
                            *(relative(*x["endpoint"]) if laser else [0,0]),squash(x.get("radius",0),500),
                            math.sin(angle) if laser else 0,math.cos(angle) if laser else 0,squash(x.get("timeout",0),60),
                            len(x.get("samples",[]))/8,*samples.ravel(),squash(x.get("countdown",0),60),
                            squash(x["collision_damage"],10),float(x.get("geometry_valid",False))]
            validity_age[age] = [1,(current["episode_frame"]-s["episode_frame"])/90]
            # Action stored on the resulting snapshot. Only expose it when the
            # preceding observation is also present (three intervening actions).
            if age < len(self.frames)-1 and frame["action"] is not None:
                for value,offset in zip(frame["action"],(0,9,14)):
                    actions[age,offset+int(value)] = 1
        self.stats = dict(history_valid=len(self.frames),history_span_ticks=current["episode_frame"]-self.frames[-1]["state"]["episode_frame"],
            omitted_enemies=max(0,sum(e[4] == 1 for e in current["entities"])-ENEMIES),
            omitted_hazards=max(0,sum(e[4] in (2,4,7) for e in current["entities"])-HAZARDS),
            simplified_lasers=sum(x.get("sample_count",0)>8 for x in current["combat_entities"]),
            invalid_laser_geometry=sum(e[4]==7 and not x["geometry_valid"] for e,x in maps[0].values()))
        if not np.isfinite(vector).all():
            raise BridgeError("Nonfinite combat history")
        return dict(grid=grid,vector=np.clip(vector,-1,1))
