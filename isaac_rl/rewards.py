"""Versioned learning rewards; game physics and the boss-clear gate are unchanged."""
from dataclasses import asdict, dataclass, replace
import math


GAMMA = 0.995


@dataclass(frozen=True)
class RewardProfile:
    time: float = -0.003
    cell: float = 0.025
    room: float = 2.0
    clear: float = 5.0
    damage: float = 0.06
    hurt: float = -2.0
    kill: float = 0.25
    death: float = -15.0
    boss: float = 50.0
    door_potential: float = 0.0
    combat_clear_only: bool = False
    cell_resets_idle: bool = True
    damage_signal: str = "attempted_v1"


PROFILES = {
    "legacy_v1": RewardProfile(),
    "balanced_v2": RewardProfile(time=-0.01,cell=0.01,clear=10.0,damage=0.2,
        hurt=-0.5,kill=0.5,death=-5.0,boss=100.0,door_potential=2.0,
        combat_clear_only=True,cell_resets_idle=False),
}
PROFILES["confirmed_v3"] = replace(PROFILES["balanced_v2"],damage_signal="hp_delta_v1")


def profile_manifest(name):
    return dict(name=name,gamma=GAMMA,**asdict(PROFILES[name]))


def door_potential(state, scale):
    """A bounded state potential from visible doors, never a selected action.

    Prefer least-visited open destinations in cleared rooms. The shaped reward
    is gamma * Phi(next) - Phi(now), including room changes and death, so moving
    back and forth cannot accumulate positive discounted shaping return.
    See Ng, Harada & Russell (1999), https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf
    """
    if not scale or state["terminal"] or not state["room"]["clear"]:
        return 0.0
    doors = [d for d in state["doors"] if d["open"] and not d["locked"]]
    if not doors:
        return 0.0
    visits = min(d["visits"] for d in doors)
    p = state["player"]
    distance = min(math.hypot(d["x"]-p["x"],d["y"]-p["y"])
                   for d in doors if d["visits"] == visits)
    left,top,right,bottom = state["room"]["bounds"]
    diagonal = max(math.hypot(right-left,bottom-top),1.0)
    return scale * max(0.0,1.0-distance/diagonal)
