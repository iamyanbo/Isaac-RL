"""Replay a known terminal diagnostic and verify unscored cleanup/reconnection."""
import argparse
import json
import math
from pathlib import Path
import time

import torch

from isaac_rl.bridge import Bridge
from isaac_rl.env import IsaacEnv
from isaac_rl.evaluate import park_after_evaluation
from isaac_rl.observation import resolve_observation_profile
from isaac_rl.ppo import load_policy, as_tensor
from isaac_rl.storage import atomic_json, load_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint",type=Path)
    parser.add_argument("--seed",required=True)
    parser.add_argument("--port",type=int,default=10002)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--scripted-fire-contact",action="store_true",
        help="Operational-only: enter the visible treasure room and contact a native fire; never training/evaluation")
    args = parser.parse_args()
    torch.set_num_threads(2)
    saved,digest,_ = load_snapshot(args.checkpoint,"cpu")
    model,_ = load_policy(saved)
    model.eval()
    output = args.output or Path("runs")/f"evaluation-cleanup-probe-{time.time_ns()}"
    output.mkdir(parents=True,exist_ok=False)
    # Fine-grained native inputs avoid orbiting a tiny fire with 8-frame holds.
    # These operational probe settings never enter the learner or evaluator.
    frames = 1 if args.scripted_fire_contact else saved["frames"]
    max_steps = 1800 if args.scripted_fire_contact else 450
    idle_limit = 1800 if args.scripted_fire_contact else 180
    atomic_json(output/"manifest.json",{
        "purpose":"operational_regression_not_performance_evaluation",
        "checkpoint":str(args.checkpoint.resolve()),"checkpoint_sha256":digest,
        "trained_steps":saved["steps"],"seed":args.seed,"port":args.port,
        "controller":"scripted native fire contact" if args.scripted_fire_contact else "frozen deterministic policy",
        "frames":frames,"max_episode_steps":max_steps,"idle_limit":idle_limit})
    env = IsaacEnv(Bridge(port=args.port,trace=output/"trace.jsonl"),frames=frames,
        max_steps=max_steps,idle_limit=idle_limit,reward_profile=saved.get("reward_profile","legacy_v1"),
        observation_profile=resolve_observation_profile(checkpoint=saved))
    try:
        obs,_ = env.reset(options={"game_seed":args.seed})
        initial = env.state
        while True:
            if args.scripted_fire_contact:
                state = env.state
                fires = [e for e in state["entities"] if e[5] == 33 and e[8] > 0]
                if fires:
                    p = state["player"]
                    fire = min(fires,key=lambda e:math.hypot(e[0]-p["x"],e[1]-p["y"]))
                    dx,dy = fire[0]-p["x"],fire[1]-p["y"]
                else:
                    doors = [d for d in state["doors"] if d["type"] == 4 and d["open"] and not d["locked"]]
                    if not doors:
                        raise RuntimeError("Scripted probe requires a visible open treasure-room door or native fire")
                    p,door = state["player"],doors[0]
                    dx,dy = door["x"]-p["x"],door["y"]-p["y"]
                direction = (0 if abs(dx)<4 else 1 if dx>0 else -1,
                             0 if abs(dy)<4 else 1 if dy>0 else -1)
                moves = [(0,0),(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]
                chosen = [moves.index(direction),0,0]
            else:
                with torch.inference_mode():
                    action,_,_,_ = model.act(as_tensor(obs),deterministic=True)
                chosen = action[0].numpy()
            obs,_,terminated,truncated,info = env.step(chosen)
            if terminated or truncated:
                break
        final = env.state
        atomic_json(output/"replayed-episode.json",{
            "scored":False,"initial":initial,"final":final,"result":info["episode"]})
        cleanup = park_after_evaluation(env,output)
    finally:
        env.close()
    # This is a second TCP connection to the SAME native process after close.
    bridge = Bridge(port=args.port,timeout=15,trace=output/"reconnect-trace.jsonl")
    try:
        reconnected = bridge.connect()
        bridge.request("ping")
    finally:
        bridge.close()
    passed = (final["player"]["dead"] and cleanup["parked"]
        and reconnected["episode"] == cleanup["state"]["episode"]
        and reconnected["stage"] == 1 and reconnected["character"] == 0
        and reconnected["difficulty"] == 0 and not reconnected["player"]["dead"])
    result = {"passed":bool(passed),"scored":False,"replayed_reason":info["episode"]["reason"],
        "replayed_actions":info["episode"]["l"],"reconnected":reconnected}
    atomic_json(output/"result.json",result)
    print(json.dumps({"output":str(output),"passed":bool(passed),
                      "reason":info["episode"]["reason"],"actions":info["episode"]["l"]}),flush=True)
    if not passed:
        raise RuntimeError("Death cleanup/reconnection regression failed; inspect retained evidence")


if __name__ == "__main__":
    main()
