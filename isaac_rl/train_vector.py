"""PPO over independently running normal-speed Isaac instances."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import psutil
import torch

from .ppo import load_policy, optimize, resolve_entropy_coef
from .observation import OBSERVATION_PROFILES, resolve_observation_profile
from .storage import atomic_json, checkpoint, load_snapshot, source_fingerprint
from .rewards import PROFILES, profile_manifest
from .vector import ParallelIsaac
from .runtime import RunLease, measure, commit_memory
from .timing import TIMING_PROFILES, configure_training


def rollout_length(steps, target, rollout, environments):
    """A target of zero means detached training has no automatic step cap."""
    if target < 0 or rollout < 1 or environments < 1:
        raise ValueError("Invalid step budget or rollout dimensions")
    if target == 0:
        return rollout
    return min(rollout,max(0,(target-steps+environments-1)//environments))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--ports",type=int,nargs="+",default=[9999,10000,10001])
    parser.add_argument("--resume",type=Path)
    parser.add_argument("--steps",type=int,default=0,help="Aggregate step cap; 0 (default) runs indefinitely")
    parser.add_argument("--rollout",type=int,help="Steps per environment; inherits checkpoint or time-normalized default")
    parser.add_argument("--batch-size",type=int)
    parser.add_argument("--frames",type=int,help="Inherit checkpoint hold; eight ticks for new runs")
    parser.add_argument("--timing-profile",choices=TIMING_PROFILES,help="physical_v1 preserves physical-time quantities when changing hold")
    parser.add_argument("--max-episode-steps",type=int)
    parser.add_argument("--idle-limit",type=int)
    parser.add_argument("--seed",type=int,default=20260912)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--entropy-coef",type=float,help="Inherit checkpoint value; old checkpoints/new runs default to 0.02")
    parser.add_argument("--reward-profile",choices=list(PROFILES),help="Defaults to checkpoint profile on resume; balanced_v2 for new runs")
    parser.add_argument("--observation-profile",choices=OBSERVATION_PROFILES)
    args = parser.parse_args()
    if args.steps < 0 or any(v is not None and v < 1 for v in (args.rollout,args.batch_size)):
        parser.error("Steps must be nonnegative; rollout and batch size must be positive")
    with RunLease(args.run):
        run_training(args)


def run_training(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    root,run = Path(__file__).resolve().parents[1],args.run.resolve()
    run.mkdir(parents=True,exist_ok=True)
    if (run/"status.json").exists() and not args.resume:
        raise RuntimeError("Run already exists; resume its checkpoint or choose a new directory")
    steps,episodes,updates = 0,0,0
    recent,seeds,losses = deque(maxlen=50),set(),{}
    origin = {"initialization":"random","seed":args.seed}
    saved = None
    if args.resume:
        saved,parent_digest,_ = load_snapshot(args.resume,args.device)
        steps,episodes,updates = saved["steps"],saved["episodes"],saved["updates"]
        recent.extend(saved["recent"])
        seeds.update(saved["training_seeds"])
        losses = saved.get("losses",{})
        parent_profile = saved.get("reward_profile","legacy_v1")
        args.reward_profile = args.reward_profile or parent_profile
        if parent_profile != args.reward_profile:
            if (run/"status.json").exists():
                raise RuntimeError("Changing reward profile requires a new run directory; preserve the old run")
            # Rewards across profiles are not comparable. Keep training history,
            # model and optimizer, but start a new telemetry reward window.
            recent.clear()
            losses = {}
        origin = {"parent":str(args.resume.resolve()),"parent_sha256":parent_digest,
                  "parent_reward_profile":parent_profile,
                  "parent_observation_profile":saved.get("observation_profile","legacy_v1"),
                  "parent_steps":steps,"initialization":"continued from project-trained PPO"}
    args.observation_profile = resolve_observation_profile(args.observation_profile,saved,(run/"status.json").exists())
    model,optimizer = load_policy(saved,args.device,args.observation_profile,with_optimizer=True)
    if saved is not None:
        torch.set_rng_state(saved["torch_rng"].cpu())
        np.random.set_state(saved["numpy_rng"])
        if args.device.startswith("cuda") and saved.get("cuda_rng"):
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
    if saved is not None and args.observation_profile != saved.get("observation_profile","legacy_v1"):
        recent.clear()
        losses = {}
    args.reward_profile = args.reward_profile or "balanced_v2"
    args.entropy_coef = resolve_entropy_coef(args.entropy_coef,saved,(run/"status.json").exists())
    previous_config = json.loads((run/"config.json").read_text()) if (run/"config.json").exists() else None
    control,schedule,timing_changed = configure_training(args,saved,(run/"status.json").exists(),previous_config)
    if timing_changed:
        recent.clear()
        losses = {}
    native_frames = (saved or {}).get("native_frames",0)
    native_frames_origin_steps = (saved or {}).get("native_frames_origin_steps",steps)
    origin.update(parent_frames=saved.get("frames") if saved else None,
                  parent_timing_profile=saved.get("timing_profile","legacy_v1") if saved else None)
    config = {key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    config.update(architecture=model.observation_layout["architecture"],observation_layout=model.observation_layout,
        algorithm="PPO",mode="full_floor",origin=origin,
        reward=profile_manifest(args.reward_profile,control),control_timing=control.manifest(),
        training_schedule=schedule,source_hashes=source_fingerprint(root))
    session_name = f"session-{time.time_ns()}"
    atomic_json(run/f"{session_name}.json",config)
    atomic_json(run/"config.json",config)
    collector = ParallelIsaac(args.ports,args.frames,args.max_episode_steps,args.idle_limit,args.reward_profile,args.observation_profile,args.timing_profile)
    count = len(args.ports)
    started,initial_steps = time.time(),steps
    stop_requested = False
    def stop(*_):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGINT,stop)
    signal.signal(signal.SIGTERM,stop)
    status = dict(pid=os.getpid(),process_started=psutil.Process().create_time(),
        status="connecting",steps=steps,episodes=episodes,updates=updates,num_envs=count,ports=args.ports,
        session_id=session_name,initial_steps=initial_steps,step_target=args.steps,device=args.device,
        checkpoint_steps=steps,exit_code=None,
        time=time.time(),message=f"Waiting for {count} independently saved normal game instances",**losses)
    status["entropy_coef"] = args.entropy_coef
    status.update(frames=args.frames,timing_profile=args.timing_profile,control_timing=control.manifest(),
                  native_frames_origin_steps=native_frames_origin_steps)
    atomic_json(run/"status.json",status)
    def report(infos=None, **fields):
        status.update(time=time.time(),steps=steps,episodes=episodes,updates=updates,native_frames=native_frames,
            recent_successes=sum(e["success"] for e in recent),recent_episodes=len(recent),
            mean_reward=float(np.mean([e["r"] for e in recent])) if recent else 0,
            sps=(steps-initial_steps)/max(time.time()-started,1))
        if infos:
            status.update(infos[0])
            status["episode_reward"] = collector.envs[0].episode_reward
            status["workers"] = [dict(port=port,episode_reward=env.episode_reward,
                episode_steps=env.steps,**info) for port,env,info in zip(args.ports,collector.envs,infos)]
        status.update(fields)
        atomic_json(run/"status.json",status)
    def save(path=None):
        checkpoint(path or run/"latest.pt",model,optimizer,dict(steps=steps,episodes=episodes,updates=updates,
            recent=list(recent),training_seeds=sorted(seeds),numpy_rng=np.random.get_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else None,
            frames=args.frames,entropy_coef=args.entropy_coef,architecture=config["architecture"],
            observation_layout=model.observation_layout,source_hashes=config["source_hashes"],origin=origin,
            timing_profile=args.timing_profile,control_timing=control.manifest(),training_schedule=schedule,
            native_frames=native_frames,native_frames_origin_steps=native_frames_origin_steps,
            ports=args.ports,losses=losses,reward_profile=args.reward_profile,
            reward=profile_manifest(args.reward_profile,control),observation_profile=args.observation_profile,created=time.time()))
        atomic_json(run/"training_seeds.json",sorted(seeds))
        # Bounded live snapshots allow behavior audits without a second client
        # taking a game port, or unbounded raw training traces.
        atomic_json(run/"live-states.json",[env.state for env in collector.envs])
        status["checkpoint_steps"] = steps
    def tensor(obs):
        return {key:torch.as_tensor(value,device=args.device) for key,value in obs.items()}
    try:
        obs,infos = collector.reset()
        seeds.update(info["game_seed"] for info in infos)
        atomic_json(run/f"{session_name}-initial-states.json",[env.state for env in collector.envs])
        save()
        if not args.resume:
            save(run/"initial.pt")
        report(infos,status="training",message=f"{count} normal-speed games; learned movement, shooting, and utility actions")
        with (run/"episodes.jsonl").open("a",buffering=1) as episode_file, (run/"updates.jsonl").open("a",buffering=1) as update_file:
            while not stop_requested:
                length = rollout_length(steps,args.steps,args.rollout,count)
                if not length:
                    break
                rollout_started = time.perf_counter()
                rollout_native_start = native_frames
                timing = {}
                rollout = {key:[] for key in ["obs","actions","log_probs","values","rewards","next_values","terminated","ended"]}
                for _ in range(length):
                    with measure(timing,"inference_s",args.device), torch.no_grad():
                        actions,log_probs,_,values = model.act(tensor(obs))
                        action_array = actions.cpu().numpy()
                        log_array, value_array = log_probs.cpu().numpy(), values.cpu().numpy()
                    with measure(timing,"collection_s"):
                        next_obs,rewards,terminated,truncated,infos = collector.step(action_array)
                    ended = terminated | truncated
                    # Values come from terminal observations, BEFORE reset_done.
                    with measure(timing,"inference_s",args.device), torch.no_grad():
                        next_values = model(tensor(next_obs))[1].cpu().numpy()
                    next_values[terminated] = 0
                    transition = dict(obs=obs,actions=action_array,log_probs=log_array,
                        values=value_array,rewards=rewards,next_values=next_values,
                        terminated=terminated,ended=ended)
                    for key,val in transition.items():
                        rollout[key].append(val)
                    steps += count
                    native_frames += sum(info.get("frame_delta",0) for info in infos)
                    for i in np.flatnonzero(ended):
                        episodes += 1
                        record = dict(time=time.time(),session_id=session_name,episode=episodes,steps=steps,port=args.ports[i],**infos[i]["episode"])
                        episode_file.write(json.dumps(record)+"\n")
                        recent.append(record)
                        atomic_json(run/f"worker-{args.ports[i]}-last-episode.json",dict(result=record,final=collector.envs[i].state))
                        if record["success"]:
                            atomic_json(run/f"success-{episodes:06d}.json",collector.envs[i].state)
                        print(f"episode={episodes} port={args.ports[i]} steps={steps} reward={record['r']:.2f} rooms={record['rooms']} boss={record['boss_seen']} success={record['success']} reason={record['reason']}",flush=True)
                    if ended.any():
                        report(status="resetting")
                    with measure(timing,"reset_s"):
                        resets = collector.reset_done(next_obs,ended)
                    for i,info in resets.items():
                        seeds.add(info["game_seed"])
                        infos[i] = info
                    obs = next_obs
                    if len(rollout["obs"]) % 8 == 0:
                        report(infos,status="training")
                    if (run/"stop.request").exists():
                        stop_requested = True
                    if stop_requested:
                        break
                report(status="optimizing")
                with measure(timing,"update_s",args.device):
                    losses = optimize(model,optimizer,rollout,device=args.device,batch_size=args.batch_size,
                                      entropy_coef=args.entropy_coef,gamma=control.gamma,lam=control.lam)
                updates += 1
                with measure(timing,"save_s"):
                    save()
                timing["wall_s"] = time.perf_counter()-rollout_started
                timing["overhead_s"] = max(0.,timing["wall_s"]-sum(v for k,v in timing.items() if k != "wall_s"))
                rollout_sps = len(rollout["obs"])*count/timing["wall_s"]
                memory = commit_memory()
                update_file.write(json.dumps(dict(time=time.time(),session_id=session_name,steps=steps,updates=updates,num_envs=count,
                    frames=args.frames,timing_profile=args.timing_profile,control_timing=control.manifest(),
                    native_frames=native_frames,native_frames_origin_steps=native_frames_origin_steps,
                    rollout_native_frames=native_frames-rollout_native_start,
                    device=args.device,entropy_coef=args.entropy_coef,timing=timing,rollout_sps=rollout_sps,commit_memory=memory,**losses))+"\n")
                report(infos,status="training",timing=timing,rollout_sps=rollout_sps,commit_memory=memory,**losses)
                print(f"update={updates} steps={steps} loss={losses['loss']:.5f} entropy={losses['entropy']:.3f} sps={status['sps']:.2f}",flush=True)
            save()
            atomic_json(run/f"{session_name}-interrupted-states.json",[env.state for env in collector.envs])
            report(status="stopped",exit_code=0,exit_reason="stop_requested" if stop_requested else "step_budget",
                message="Checkpoint saved; closing bridges and exiting this learner session")
    except Exception as error:
        report(status="error",exit_code=1,exit_reason="exception",message=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            collector.close()
        except Exception as cleanup_error:
            report(status="error",exit_code=1,exit_reason="cleanup_error",ended_at=time.time(),
                cleanup_complete=False,message=f"Bridge cleanup failed: {cleanup_error}")
            raise
        else:
            report(ended_at=time.time(),cleanup_complete=True)


if __name__ == "__main__":
    main()
