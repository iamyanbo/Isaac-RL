"""Bounded schema smoke test on reserved native instance 3; no policy/evaluation.

Requires the test-only observability fixture in that private runtime. Never
connects to training ports, writes checkpoints, or performs learner updates.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from isaac_rl.bridge import Bridge
from isaac_rl.env import IsaacEnv
from isaac_rl.evaluate import park_after_evaluation
from isaac_rl.storage import atomic_json, source_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    atomic_json(args.output/"manifest.json",dict(purpose="native_schema_smoke_only",port=10002,
        fixture="scripts/fixtures/observability.lua",training_transitions=False,started=time.time(),
        source_hashes=source_fingerprint(Path(__file__).resolve().parents[1])))
    env = IsaacEnv(Bridge(port=10002,timeout=30,trace=args.output/"trace.jsonl"),frames=4,
        observation_profile="combat_history_v3",reward_profile="completion_v4",timing_profile="physical_v1")
    states,timings = [],[]
    completed = False
    try:
        obs,_ = env.reset()
        states.append(env.state)
        for _ in range(24):
            started = time.perf_counter()
            obs,_,terminated,truncated,info = env.step([0,2,0])
            timings.append(time.perf_counter()-started)
            assert env.observation_space.contains(obs)
            states.append(env.state)
            if terminated or truncated:
                break
        players = [s["combat_player"] for s in states]
        hazards = [(e,x) for s in states for e,x in zip(s["entities"],s["combat_entities"])]
        shared = [[(e,x) for e,x in zip(s["entities"],s["combat_entities"])
                   if e[4]==2 and e[11]==581234] for s in states]
        stable = {variant:{x["track_id"] for frame in shared for e,x in frame if e[6]==variant}
                  for variant in (0,4)}
        checks = dict(native_four_tick_hold=all(b["episode_frame"]-a["episode_frame"] == 4 for a,b in zip(states,states[1:])),
            duplicate_native_seeds_observed=sum(len(frame)==2 for frame in shared)>=2,
            shared_seed_tracks_distinct=any(len(frame)==2 and len({x["track_id"] for e,x in frame})==2 for frame in shared),
            lifetime_tracks_stable=all(len(ids)==1 for ids in stable.values()),
            all_native_tracks_unique=all(len({x["track_id"] for x in s["combat_entities"]})==len(s["entities"]) for s in states),
            current_firing_cooldown_changes=len({p["fire_cooldown"] for p in players})>1,
            damage_cooldown_observed=any(p["damage_cooldown_render_frames"]>0 for p in players),
            invulnerability_observed=any(p["invincible"] for p in players),
            player_hitbox_observed=all(p["size"]>0 and len(p["size_multi"])==2 for p in players),
            projectile_variants_observed=len({e[6] for e,x in hazards if e[4]==2})>=2,
            enemy_phase_observed=any(e[4]==1 and "state_frame" in x for e,x in hazards),
            bomb_age_observed=len({x["bomb_age"] for e,x in hazards if e[4]==4})>=2,
            linear_laser_geometry=any(e[4]==7 and not x["circle"] and len(x["samples"])>=2 and x["endpoint"] != e[:2] for e,x in hazards),
            ring_laser_geometry=any(e[4]==7 and x["circle"] and x["radius"]>0 for e,x in hazards),
            history_four_snapshots=info["history_valid"]==4,history_twelve_ticks=info["history_span_ticks"]==12)
        atomic_json(args.output/"states.json",states)
        result = dict(passed=all(checks.values()),checks=checks,steps=len(states)-1,
            median_step_seconds=float(np.median(timings)),last_history=env.history.stats,
            fire_cooldowns=[p["fire_cooldown"] for p in players],
            damage_cooldowns_render_frames=[p["damage_cooldown_render_frames"] for p in players])
        atomic_json(args.output/"result.json",result)
        completed = True
        print(json.dumps(result,indent=2),flush=True)
        if not result["passed"]:
            raise RuntimeError("Native schema smoke check failed")
    except Exception:
        atomic_json(args.output/"failure-state.json",env.state)
        raise
    finally:
        if completed:
            park_after_evaluation(env,args.output)
        env.close()


if __name__ == "__main__":
    main()
