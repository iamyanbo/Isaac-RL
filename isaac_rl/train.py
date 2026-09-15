import argparse
from collections import deque
import json
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch
import psutil

from .bridge import Bridge
from .env import IsaacEnv
from .ppo import load_policy, as_tensor, optimize, resolve_entropy_coef
from .observation import OBSERVATION_PROFILES, resolve_observation_profile
from .rewards import PROFILES, profile_manifest
from .storage import atomic_json, checkpoint, source_fingerprint
from .runtime import RunLease, measure
from .timing import TIMING_PROFILES, configure_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,default=Path("runs/ppo-floor1"))
    parser.add_argument("--steps",type=int,default=0,help="Total step cap; zero runs indefinitely")
    parser.add_argument("--rollout",type=int)
    parser.add_argument("--batch-size",type=int)
    parser.add_argument("--frames",type=int)
    parser.add_argument("--timing-profile",choices=TIMING_PROFILES)
    parser.add_argument("--max-episode-steps",type=int)
    parser.add_argument("--idle-limit",type=int)
    parser.add_argument("--seed",type=int,default=20260912)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--entropy-coef",type=float,help="Inherit checkpoint value; old checkpoints/new runs default to 0.02")
    parser.add_argument("--resume",type=Path)
    parser.add_argument("--trace",action="store_true")
    parser.add_argument("--observation-profile",choices=OBSERVATION_PROFILES)
    parser.add_argument("--reward-profile",choices=list(PROFILES))
    args = parser.parse_args()
    if args.steps < 0 or any(v is not None and v < 1 for v in (args.rollout,args.batch_size)):
        parser.error("Nonnegative steps and positive rollout required")
    with RunLease(args.run):
        run_training(args)


def run_training(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    root = Path(__file__).resolve().parents[1]
    run = args.run.resolve()
    run.mkdir(parents=True,exist_ok=True)
    if (run/"status.json").exists() and not args.resume:
        raise RuntimeError("Run already exists; use --resume or choose a new run directory")
    steps,episodes,updates = 0,0,0
    recent = deque(maxlen=50)
    training_seeds = set()
    losses = {}
    saved = None
    if args.resume:
        saved = torch.load(args.resume,map_location=args.device,weights_only=False)
        steps,episodes,updates = saved["steps"],saved["episodes"],saved["updates"]
        recent.extend(saved["recent"])
        training_seeds.update(saved["training_seeds"])
        losses = saved.get("losses",{})
        if not losses and (run/"updates.jsonl").exists():
            update_lines = (run/"updates.jsonl").read_text().splitlines()
            if update_lines:
                losses = {k:v for k,v in json.loads(update_lines[-1]).items() if k not in ("time","steps","updates")}
    args.observation_profile = resolve_observation_profile(args.observation_profile,saved,(run/"status.json").exists())
    model,optimizer = load_policy(saved,args.device,args.observation_profile,with_optimizer=True)
    if saved is not None:
        torch.set_rng_state(saved["torch_rng"].cpu())
        np.random.set_state(saved["numpy_rng"])
        if args.device.startswith("cuda") and saved.get("cuda_rng"):
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
    parent_reward = saved.get("reward_profile","legacy_v1") if saved is not None else "legacy_v1"
    args.reward_profile = args.reward_profile or parent_reward
    if saved is not None and args.reward_profile != parent_reward:
        if (run/"status.json").exists():
            raise RuntimeError("Changing reward profile requires a new run directory")
        recent.clear()
        losses = {}
    if saved is not None and args.observation_profile != saved.get("observation_profile","legacy_v1"):
        recent.clear()
        losses = {}
    args.entropy_coef = resolve_entropy_coef(args.entropy_coef,saved,(run/"status.json").exists())
    previous_config = json.loads((run/"config.json").read_text()) if (run/"config.json").exists() else None
    control,schedule,timing_changed = configure_training(args,saved,(run/"status.json").exists(),previous_config,
                                                       default_rollout=256,default_batch=64)
    if timing_changed:
        recent.clear()
        losses = {}
    native_frames = (saved or {}).get("native_frames",0)
    native_frames_origin_steps = (saved or {}).get("native_frames_origin_steps",steps)
    config = {key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    config.update(architecture=model.observation_layout["architecture"],observation_layout=model.observation_layout,
        algorithm="PPO",initialization="random" if not args.resume else str(args.resume),
        observations="current room entities, spatial grid, and visit memory",reward=profile_manifest(args.reward_profile,control),
        control_timing=control.manifest(),training_schedule=schedule,
        source_hashes=source_fingerprint(root))
    session_name = f"session-{time.time_ns()}"
    atomic_json(run/f"{session_name}.json",config)
    atomic_json(run/"config.json",config)
    bridge = Bridge(trace=run/"trace.jsonl" if args.trace else None)
    env = IsaacEnv(bridge,frames=args.frames,max_steps=args.max_episode_steps,idle_limit=args.idle_limit,
                   observation_profile=args.observation_profile,reward_profile=args.reward_profile,timing_profile=args.timing_profile)
    started,initial_steps = time.time(),steps
    stop_requested = False
    def stop(*_):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGINT,stop)
    signal.signal(signal.SIGTERM,stop)
    status = {"pid":os.getpid(),"process_started":psutil.Process().create_time(),"status":"connecting","time":time.time(),"steps":steps,"episodes":episodes,
              "updates":updates,"message":"Waiting for the private game bridge on localhost:9999"}
    status.update(losses)
    status.update(session_id=session_name,num_envs=1,ports=[9999],device=args.device,
        step_target=args.steps,initial_steps=initial_steps,checkpoint_steps=steps,exit_code=None,
        entropy_coef=args.entropy_coef,frames=args.frames,timing_profile=args.timing_profile,
        control_timing=control.manifest(),native_frames_origin_steps=native_frames_origin_steps)
    atomic_json(run/"status.json",status)
    def update_status(info=None, **extra):
        status.update(time=time.time(),steps=steps,episodes=episodes,updates=updates,native_frames=native_frames,
            episode_reward=getattr(env,"episode_reward",0),
            mean_reward=float(np.mean([e["r"] for e in recent])) if recent else 0,
            recent_successes=sum(e["success"] for e in recent),recent_episodes=len(recent),
            sps=(steps-initial_steps)/max(time.time()-started,1))
        status.update(info or {})
        status.update(extra)
        atomic_json(run/"status.json",status)
    def save():
        checkpoint(run/"latest.pt",model,optimizer,dict(steps=steps,episodes=episodes,updates=updates,
            recent=list(recent),training_seeds=sorted(training_seeds),numpy_rng=np.random.get_state(),
            frames=args.frames,entropy_coef=args.entropy_coef,architecture=config["architecture"],
            observation_layout=model.observation_layout,source_hashes=config["source_hashes"],losses=losses,
            timing_profile=args.timing_profile,control_timing=control.manifest(),training_schedule=schedule,
            native_frames=native_frames,native_frames_origin_steps=native_frames_origin_steps,
            observation_profile=args.observation_profile,reward_profile=args.reward_profile,
            reward=profile_manifest(args.reward_profile,control),created=time.time()))
        atomic_json(run/"training_seeds.json",sorted(training_seeds))
        status["checkpoint_steps"] = steps
    try:
        obs,info = env.reset()
        training_seeds.add(info["game_seed"])
        update_status(info,status="training",message="Random-initialized PPO; real full-floor episodes")
        save()
        with (run/"episodes.jsonl").open("a",buffering=1) as episode_file, (run/"updates.jsonl").open("a",buffering=1) as update_file:
            while (args.steps == 0 or steps < args.steps) and not stop_requested:
                rollout_started, timing = time.perf_counter(), {}
                rollout_native_start = native_frames
                rollout = {key:[] for key in ["obs","actions","log_probs","values","rewards","next_values","terminated","ended"]}
                for _ in range(args.rollout if args.steps == 0 else min(args.rollout,args.steps-steps)):
                    with measure(timing,"inference_s",args.device), torch.no_grad():
                        action,log_prob,_,value = model.act(as_tensor(obs,args.device))
                        action_array = action[0].cpu().numpy()
                    with measure(timing,"collection_s"):
                        next_obs,reward,terminated,truncated,info = env.step(action_array)
                    with measure(timing,"inference_s",args.device), torch.no_grad():
                        next_value = 0.0 if terminated else model(as_tensor(next_obs,args.device))[1].item()
                    transition = dict(obs=obs,actions=action[0].cpu().numpy(),log_probs=log_prob.item(),values=value.item(),
                        rewards=reward,next_values=next_value,terminated=terminated,ended=terminated or truncated)
                    for key,val in transition.items():
                        rollout[key].append(val)
                    steps += 1
                    native_frames += info.get("frame_delta",0)
                    obs = next_obs
                    if terminated or truncated:
                        episodes += 1
                        record = dict(time=time.time(),session_id=session_name,episode=episodes,steps=steps,**info["episode"])
                        episode_file.write(json.dumps(record)+"\n")
                        recent.append(record)
                        print(f"episode={episodes} steps={steps} reward={record['r']:.2f} rooms={record['rooms']} boss={record['boss_seen']} success={record['success']} reason={record['reason']}",flush=True)
                        if record["success"]:
                            atomic_json(run/f"success-{episodes:06d}.json",env.state)
                        update_status(status="resetting")
                        with measure(timing,"reset_s"):
                            obs,info = env.reset()
                        training_seeds.add(info["game_seed"])
                    if steps % 16 == 0:
                        update_status(info,status="training")
                    if (run/"stop.request").exists():
                        stop_requested = True
                    if stop_requested:
                        break
                update_status(status="optimizing")
                with measure(timing,"update_s",args.device):
                    losses = optimize(model,optimizer,rollout,device=args.device,entropy_coef=args.entropy_coef,
                                      batch_size=args.batch_size,gamma=control.gamma,lam=control.lam)
                updates += 1
                with measure(timing,"save_s"):
                    save()
                timing["wall_s"] = time.perf_counter()-rollout_started
                rollout_sps = len(rollout["obs"])/timing["wall_s"]
                update_file.write(json.dumps(dict(time=time.time(),session_id=session_name,steps=steps,updates=updates,
                    frames=args.frames,timing_profile=args.timing_profile,control_timing=control.manifest(),
                    native_frames=native_frames,native_frames_origin_steps=native_frames_origin_steps,
                    rollout_native_frames=native_frames-rollout_native_start,
                    timing=timing,rollout_sps=rollout_sps,device=args.device,entropy_coef=args.entropy_coef,**losses))+"\n")
                update_status(info,status="training",timing=timing,rollout_sps=rollout_sps,**losses)
                print(f"update={updates} steps={steps} loss={losses['loss']:.5f} entropy={losses['entropy']:.3f} sps={status['sps']:.2f}",flush=True)
            save()
            update_status(status="stopped",exit_code=0,exit_reason="stop_requested" if stop_requested else "step_budget",
                message="Checkpoint saved; closing bridge and exiting this learner session")
    except Exception as exc:
        update_status(status="error",exit_code=1,exit_reason="exception",message=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            env.close()
        except Exception as cleanup_error:
            update_status(status="error",exit_code=1,exit_reason="cleanup_error",ended_at=time.time(),
                cleanup_complete=False,message=f"Bridge cleanup failed: {cleanup_error}")
            raise
        else:
            update_status(ended_at=time.time(),cleanup_complete=True)


if __name__ == "__main__":
    main()
