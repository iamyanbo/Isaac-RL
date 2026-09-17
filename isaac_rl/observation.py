"""Fixed-size features from currently observable room state and episode memory.

No map reveal, hidden room content, target controller, or recommended actions.
"""
import numpy as np

CHANNELS, HEIGHT, WIDTH = 10, 16, 28
VECTOR_SIZE = 24 + 8 * 8 + 16 * 10 + 16 * 6 + 8 * 6
OBSERVATION_PROFILES = ("legacy_v1","terrain_v2","combat_history_v3","combat_gru_v4")
# Values verified against the installed game's resources/scripts/enums.lua.
PLAYER_SOLID_COLLISIONS = frozenset((2,3,4))  # OBJECT, SOLID, WALL; not 5 (except player)
TERRAIN_HAZARDS = frozenset((8,9,12))  # SPIKES, SPIKES_ONOFF, TNT


def resolve_observation_profile(requested=None, checkpoint=None, existing_run=False):
    parent = checkpoint.get("observation_profile","legacy_v1") if checkpoint is not None else None
    if parent is not None and parent not in OBSERVATION_PROFILES:
        raise ValueError(f"Unsupported checkpoint observation profile: {parent}")
    profile = requested or parent or "terrain_v2"
    if profile not in OBSERVATION_PROFILES:
        raise ValueError(f"Unsupported observation profile: {profile}")
    if existing_run and parent is not None and profile != parent:
        raise ValueError("Changing observation profile requires a new run directory")
    return profile


def encode(state, cells, profile="terrain_v2", visit_scale=1.0):
    if profile in ("combat_history_v3","combat_gru_v4"):
        raise ValueError(f"{profile} requires a per-environment combat encoder")
    if profile not in OBSERVATION_PROFILES:
        raise ValueError(f"Unsupported observation profile: {profile}")
    grid = np.zeros((CHANNELS, HEIGHT, WIDTH), np.float32)
    p, room = state["player"], state["room"]
    left, top, right, bottom = room["bounds"]
    width, height = max(right-left, 1), max(bottom-top, 1)
    def position(x, y):
        return (int(np.clip((y-top)/height*(HEIGHT-1), 0, HEIGHT-1)),
                int(np.clip((x-left)/width*(WIDTH-1), 0, WIDTH-1)))
    def stamp(channel, x, y, value=1):
        row, col = position(x, y)
        grid[channel, row, col] = max(grid[channel, row, col], value)
    def relative(x, y):
        return [(x-p["x"])/width, (y-p["y"])/height]
    stamp(0, p["x"], p["y"])
    for g in state["grid"]:
        x,y,kind,collision,_ = g
        if (collision >= 2 if profile == "legacy_v1" else collision in PLAYER_SOLID_COLLISIONS):
            stamp(4, x,y)
        elif collision == 1:
            stamp(5,x,y)
        if kind in ((14,15) if profile == "legacy_v1" else TERRAIN_HAZARDS):
            stamp(6,x,y)
    for e in state["entities"]:
        channel = {1:1,2:2,3:3,4:6,5:6,6:8,7:6}.get(e[4])
        if channel is not None:
            stamp(channel,e[0],e[1])
    for door in state["doors"]:
        stamp(7,door["x"],door["y"],1 if door["open"] else 0.25)
    for (room_id, cx, cy), count in cells.items():
        if room_id == room["id"]:
            stamp(9,cx*40+20,cy*40+20,min(count*visit_scale/10,1))
    scalars = [
        (p["x"]-left)/width, (p["y"]-top)/height, p["vx"]/15,p["vy"]/15,
        p["hearts"]/24,p["soul"]/24,p["max_hearts"]/24,
        p["coins"]/99,p["keys"]/10,p["bombs"]/10,p["damage"]/20,p["speed"]/2,
        p["fire_delay"]/30,p["shot_speed"]/3,p["tear_range"]/1000,p["charge"]/12,
        p["items"]/20,float(room["clear"]),room["type"]/29,room["visits"]/10,
        room["enemies"]/20,state["episode_frame"]/27000,
        float(state["boss_seen"]),len(state["visited"])/30,
    ]
    doors = np.zeros((8,8),np.float32)
    for door in state["doors"]:
        doors[door["slot"]] = [1,*relative(door["x"],door["y"]),
            float(door["open"]),float(door["locked"]),door["type"]/29,
            min(door["visits"],10)/10,float(door["visits"] == 0)]
    entities = sorted(state["entities"], key=lambda e:(e[0]-p["x"])**2+(e[1]-p["y"])**2)
    enemies = np.zeros((16,10),np.float32)
    projectiles = np.zeros((16,6),np.float32)
    pickups = np.zeros((8,6),np.float32)
    for i,e in enumerate([e for e in entities if e[4] == 1][:16]):
        enemies[i] = [1,*relative(e[0],e[1]),e[2]/15,e[3]/15,
            e[8]/max(e[9],1),e[10]/50,e[5]/1000,e[6]/100,
            min(abs(e[0]-p["x"]),abs(e[1]-p["y"]))/100]
    for i,e in enumerate([e for e in entities if e[4] in (2,4,7)][:16]):
        projectiles[i] = [1,*relative(e[0],e[1]),e[2]/15,e[3]/15,e[10]/50]
    for i,e in enumerate([e for e in entities if e[4] == 3][:8]):
        pickups[i] = [1,*relative(e[0],e[1]),e[6]/400,e[7]/750,e[10]/50]
    vector = np.concatenate([np.asarray(scalars,np.float32),doors.ravel(),enemies.ravel(),projectiles.ravel(),pickups.ravel()])
    return {"grid":grid,"vector":np.clip(vector,-1,1)}


def observation_manifest(profile):
    if profile not in OBSERVATION_PROFILES:
        raise ValueError(f"Unsupported observation profile: {profile}")
    if profile == "combat_gru_v4":
        from .history import FRAME_VECTOR_SIZE
        return dict(profile=profile,architecture=3,channels=CHANNELS,
            vector_size=FRAME_VECTOR_SIZE+19,history=1,frame_vector_size=FRAME_VECTOR_SIZE,
            native_schema="combat_v2",actions="previous executed action one-hot 9/5/4 plus valid bit; zero at episode reset",
            identity="current threat-distance order; no ID values input",
            boundary="GRU memory persists across rooms; reset on episode end and process restart",
            enemy_tracks=16,hazard_tracks=32,laser_samples=8,
            memory=dict(kind="GRU",layers=1,hidden_size=256,residual_readout=True,
                sampler="ordered per-environment sequences; length=min(rollout,batch_size)",
                initial_state="detached behavior state at sequence start",padding=False))
    if profile == "combat_history_v3":
        from .history import HISTORY, HISTORY_VECTOR_SIZE, FRAME_VECTOR_SIZE
        return dict(profile=profile,architecture=2,channels=CHANNELS*HISTORY,
            vector_size=HISTORY_VECTOR_SIZE,history=HISTORY,frame_vector_size=FRAME_VECTOR_SIZE,
            native_schema="combat_v2",order="newest_first",actions="three intervening executed actions, one-hot 9/5/4",
            identity="entity-lifetime track_id alignment within each sample; no ID values input",
            boundary="zero-pad and invalidate history on episode or room change",
            enemy_tracks=16,hazard_tracks=32,laser_samples=8)
    return dict(profile=profile,architecture=1,channels=CHANNELS,vector_size=VECTOR_SIZE,history=1)


def compatible_observation_layout(layout, profile):
    """Explicit metadata-only upgrade of the original history association bug.

    No feature widths/order/meaning, model tensors or Adam state change. Accept
    only the exact previous manifest, not arbitrary architecture-2 variants.
    """
    current = observation_manifest(profile)
    if layout == current:
        return True
    previous = dict(current, native_schema="combat_v1",
        identity="InitSeed-aligned entity tracks within each sample; no seed values input")
    return profile == "combat_history_v3" and layout == previous
