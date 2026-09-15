"""Frozen-policy evaluation. Never trains on evaluation transitions."""
import argparse
import json
import math
from pathlib import Path
import time

import torch

from .bridge import Bridge
from .env import IsaacEnv
from .ppo import load_policy, as_tensor
from .observation import resolve_observation_profile
from .storage import atomic_json, load_snapshot, source_fingerprint
from .rewards import profile_manifest
from .analyze import distribution_summary
from .timing import configure_evaluation


def canonical_seed(seed):
    return "".join(str(seed).upper().split())


def success_gate(episodes):
    n = len(episodes)
    wins = sum(bool(e["success"]) for e in episodes)
    rate = wins/n if n else 0
    z = 1.96
    lower = ((rate+z*z/(2*n)-z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n)))/(1+z*z/n)) if n else 0
    unique = len({canonical_seed(e["seed"]) for e in episodes}) == n
    verified_wins = all(not e["success"] or (e["boss_seen"] and e["boss_defeated"] and e["reason"] == "boss_clear") for e in episodes)
    return {"episodes":n,"wins":wins,"rate":rate,"wilson_95_lower":max(0.0,min(1.0,lower)),"unique_seeds":unique,
        "passed":n >= 100 and rate >= 0.9 and lower >= 0.8 and unique and verified_wins}


def park_after_evaluation(env, output):
    """Leave the reserved game reconnectable; this reset is never scored.

    Native game-over screens stop MC_POST_UPDATE, which owns the bridge's
    reconnect loop. An alive but truncated combat run can also die after close.
    Only call after every requested evaluation result has been persisted, not
    after a failed/timed-out transition that may still be outstanding.
    """
    previous = env.state
    record = {"purpose":"operational_reset_after_evaluation","scored":False,
              "previous_seed":previous["seed"],"previous_episode":previous["episode"]}
    try:
        env.reset()
        state = env.state
        if not (state["episode"] > previous["episode"] and state["stage"] == 1
            and state["character"] == 0 and state["difficulty"] == 0
            and not state["player"]["dead"] and state["room"]["clear"]
            and state["room"]["enemies"] == 0):
            raise RuntimeError("Cleanup did not reach a fresh normal starting room")
        record.update(parked=True,state=state)
    except Exception as error:
        # Preserve the completed experiment and report cleanup separately.
        # Never retry a reset whose response timed out.
        record.update(parked=False,error=f"{type(error).__name__}: {error}")
    atomic_json(output/"cleanup.json",record)
    print(json.dumps({"evaluation_cleanup":{k:v for k,v in record.items() if k != "state"}}),flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint",type=Path)
    parser.add_argument("--episodes",type=int,default=100)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--stochastic",action="store_true")
    parser.add_argument("--port",type=int,default=9999)
    parser.add_argument("--max-episode-steps",type=int)
    parser.add_argument("--idle-limit",type=int)
    args = parser.parse_args()
    if args.episodes < 1 or any(v is not None and v < 1 for v in (args.max_episode_steps,args.idle_limit)):
        parser.error("Episode count and limits must be positive")
    torch.set_num_threads(2)
    torch.manual_seed(946513)
    saved,digest,payload = load_snapshot(args.checkpoint,args.device)
    control = configure_evaluation(args,saved)
    observation_profile = resolve_observation_profile(checkpoint=saved)
    model,_ = load_policy(saved,args.device)
    model.eval()
    output = args.output or args.checkpoint.parent / f"eval-{time.time_ns()}"
    output.mkdir(parents=True,exist_ok=False)
    # Preserve the exact checkpoint being evaluated, not a moving latest.pt path.
    (output/"checkpoint.pt").write_bytes(payload)
    root = Path(__file__).resolve().parents[1]
    manifest = {"checkpoint":str(args.checkpoint.resolve()),"checkpoint_sha256":digest,
        "checkpoint_snapshot":str((output/"checkpoint.pt").resolve()),
        "trained_steps":saved["steps"],"episodes_requested":args.episodes,
        "policy":"stochastic" if args.stochastic else "deterministic", "source_hashes":source_fingerprint(root),
        "mode":"full_floor", "frames":saved["frames"], "started":time.time(),
        "port":args.port,"max_episode_steps":args.max_episode_steps,"idle_limit":args.idle_limit,
        "purpose":"consistency_evaluation" if args.episodes >= 100 else "diagnostic_only",
        "cleanup":"unscored reset after all requested episodes; recorded separately in cleanup.json",
        "reward":profile_manifest(saved.get("reward_profile","legacy_v1"),control),
        "timing_profile":control.profile,"control_timing":control.manifest(),
        "observation_profile":observation_profile}
    atomic_json(output/"manifest.json",manifest)
    forbidden = {canonical_seed(seed) for seed in saved["training_seeds"]}
    records = []
    env = IsaacEnv(Bridge(port=args.port,trace=output/"trace.jsonl"),frames=saved["frames"],
                   max_steps=args.max_episode_steps,idle_limit=args.idle_limit,
                   reward_profile=saved.get("reward_profile","legacy_v1"),observation_profile=observation_profile,
                   timing_profile=control.profile)
    try:
        with (output/"episodes.jsonl").open("w",buffering=1) as log:
            while len(records) < args.episodes:
                obs,info = env.reset()
                if canonical_seed(info["game_seed"]) in forbidden:
                    continue
                forbidden.add(canonical_seed(info["game_seed"]))
                initial = env.state
                while True:
                    with torch.inference_mode():
                        action,_,_,_ = model.act(as_tensor(obs,args.device),deterministic=not args.stochastic)
                    obs,_,terminated,truncated,info = env.step(action[0].cpu().numpy())
                    if terminated or truncated:
                        record = dict(time=time.time(),episode=len(records)+1,**info["episode"])
                        # Each success must be evidenced by real boss-room state.
                        if record["success"]:
                            final = env.state
                            if not (initial["stage"] == 1 and initial["character"] == 0 and initial["difficulty"] == 0
                                and final["stage"] == 1 and final["room"]["type"] == 5
                                and final["room"]["clear"] and final["room"]["enemies"] == 0
                                and final["boss_seen"] and final["boss_defeated"] and not final["player"]["dead"]):
                                raise RuntimeError("Success flag contradicted by actual game state")
                        records.append(record)
                        log.write(json.dumps(record)+"\n")
                        atomic_json(output/f"episode-{len(records):03d}.json",{"initial":initial,"final":env.state,"result":record})
                        atomic_json(output/"result.json",dict(**success_gate(records),complete=len(records) == args.episodes))
                        print(json.dumps(record),flush=True)
                        break
        atomic_json(output/"distribution.json",dict(**success_gate(records),**distribution_summary(records)))
        print(json.dumps(success_gate(records),indent=2))
        park_after_evaluation(env,output)
    finally:
        env.close()


if __name__ == "__main__":
    main()
