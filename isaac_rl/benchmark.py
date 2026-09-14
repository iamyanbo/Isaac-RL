"""Unscored native collection and paired CPU/CUDA PPO benchmarks.

Never attaches to occupied ports or writes training checkpoints. Model updates
are disposable copies, using the SAME captured rollout, weights and optimizer.
"""
import argparse
import copy
import json
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
import psutil
import torch

from .ppo import ActorCritic, optimize
from .storage import atomic_json, load_snapshot, source_fingerprint
from .vector import ParallelIsaac
from .runtime import commit_memory


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def summary(samples):
    return dict(samples=samples, median=float(statistics.median(samples)),
                min=min(samples), max=max(samples))


def resources():
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return dict(cpu_percent=psutil.cpu_percent(), commit_memory=commit_memory(), free_ram_gib=psutil.virtual_memory().available/2**30,
        gpu=gpu.stdout.strip(), gpu_query_error=gpu.stderr.strip(),
        free_disk_gib=psutil.disk_usage(Path.cwd().anchor).free/2**30)


def model_pair(saved, device):
    model = ActorCritic().to(device)
    model.load_state_dict(saved["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
    return model, optimizer


def collect(saved, ports, length):
    model, _ = model_pair(saved, "cpu")
    collector = ParallelIsaac(ports, saved["frames"], 3375, 450,
        saved["reward_profile"], saved["observation_profile"])
    rollout = {key:[] for key in ("obs","actions","log_probs","values","rewards","next_values","terminated","ended")}
    timing = dict(collection_s=0., inference_s=0., reset_s=0.)
    completed = []
    samples = []
    psutil.cpu_percent()
    try:
        t = time.perf_counter()
        obs, _ = collector.reset()
        initial_reset_s = time.perf_counter()-t
        initial = [env.state for env in collector.envs]
        started = time.perf_counter()
        for i in range(length):
            t = time.perf_counter()
            with torch.no_grad():
                actions, logs, _, values = model.act({k:torch.as_tensor(v) for k,v in obs.items()})
            timing["inference_s"] += time.perf_counter()-t
            t = time.perf_counter()
            next_obs, rewards, terminated, truncated, infos = collector.step(actions.numpy())
            timing["collection_s"] += time.perf_counter()-t
            ended = terminated | truncated
            t = time.perf_counter()
            with torch.no_grad():
                next_values = model({k:torch.as_tensor(v) for k,v in next_obs.items()})[1].numpy()
            timing["inference_s"] += time.perf_counter()-t
            next_values[terminated] = 0
            transition = dict(obs=obs, actions=actions.numpy(), log_probs=logs.numpy(), values=values.numpy(),
                rewards=rewards, next_values=next_values, terminated=terminated, ended=ended)
            for key,value in transition.items():
                rollout[key].append(value)
            completed.extend(infos[j]["episode"] for j in np.flatnonzero(ended))
            t = time.perf_counter()
            collector.reset_done(next_obs, ended)
            timing["reset_s"] += time.perf_counter()-t
            obs = next_obs
            if (i+1) % 16 == 0:
                samples.append(psutil.cpu_percent())
        timing["wall_s"] = time.perf_counter()-started
        timing["initial_reset_s"] = initial_reset_s
        timing["transitions"] = length*len(ports)
        timing["steps_per_second"] = timing["transitions"]/timing["wall_s"]
        timing["cpu_percent"] = summary(samples)
        timing["resources_after"] = resources()
        final = [env.state for env in collector.envs]
        # Operational reset keeps idle native workers reconnectable; not scored.
        collector.reset()
        return rollout, dict(ports=ports, timing=timing, initial=initial, final=final, episodes=completed)
    finally:
        collector.close()


def update_benchmark(saved, rollout, repeats, devices=None):
    devices = devices or (["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
    results = {d:dict(update_s=[], full_four_epoch_s=[], inference_batch_s=[]) for d in devices}
    # Alternate order to reduce thermal/load order bias. Each measurement starts
    # from identical parameters/optimizer and the same NumPy minibatch shuffle.
    for iteration in range(repeats+1):
        for device in devices if iteration%2 == 0 else list(reversed(devices)):
            for full in (False, True):
                model, optimizer = model_pair(saved, device)
                np.random.seed(9132026)
                sync(device)
                t = time.perf_counter()
                metrics = optimize(model,optimizer,rollout,device=device,batch_size=128,
                    target_kl=float("inf") if full else .025)
                sync(device)
                elapsed = time.perf_counter()-t
                if iteration:
                    results[device]["full_four_epoch_s" if full else "update_s"].append(elapsed)
                if not full:
                    results[device]["last_metrics"] = metrics
                del model, optimizer
            model, unused_optimizer = model_pair(saved,device)
            del unused_optimizer
            def infer():
                with torch.no_grad():
                    obs = {k:torch.as_tensor(v,device=device) for k,v in rollout["obs"][0].items()}
                    a,lp,_,v = model.act(obs)
                    # Include the production transfers back to the host.
                    return a.cpu().numpy(),lp.cpu().numpy(),v.cpu().numpy()
            for _ in range(5):
                infer()
            sync(device)
            t = time.perf_counter()
            for _ in range(128):
                infer()
            sync(device)
            if iteration:
                results[device]["inference_batch_s"].append((time.perf_counter()-t)/128)
            del model
    for result in results.values():
        for key in ("update_s","full_four_epoch_s","inference_batch_s"):
            result[key] = summary(result[key])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--ports", type=int, nargs="+", required=True)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collection-only", action="store_true")
    parser.add_argument("--from-capture",type=Path,help="Reuse an existing native benchmark directory; no game connections")
    parser.add_argument("--devices",nargs="+",choices=["cpu","cuda"])
    args = parser.parse_args()
    if args.length < 16 or args.repeats < 1:
        parser.error("Use at least 16 native steps and one measurement repeat")
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(9132026)
    np.random.seed(9132026)
    saved,digest,payload = load_snapshot(args.checkpoint)
    (args.output/"checkpoint.pt").write_bytes(payload)
    report = dict(purpose="unscored_throughput_benchmark", scored=False,
        checkpoint_sha256=digest, checkpoint_steps=saved["steps"], torch_threads=2,
        torch_version=torch.__version__, cuda_available=torch.cuda.is_available(),
        logical_cpus=psutil.cpu_count(), physical_cpus=psutil.cpu_count(logical=False),
        resources_before=resources(), source_hashes=source_fingerprint(Path(__file__).resolve().parents[1]))
    atomic_json(args.output/"manifest.json", report)
    if args.from_capture:
        native = json.loads((args.from_capture/"native.json").read_text())
        rollout = torch.load(args.from_capture/"rollout.pt",weights_only=False)
        capture = json.loads((args.from_capture/"manifest.json").read_text())
        if capture["checkpoint_sha256"] != digest:
            raise ValueError("Captured rollout belongs to a different checkpoint")
        report["capture_source"] = str(args.from_capture.resolve())
    else:
        rollout, native = collect(saved,args.ports,args.length)
    atomic_json(args.output/"native.json", native)
    torch.save(rollout,args.output/"rollout.pt")
    report["collection"] = native["timing"]
    print(json.dumps(dict(native_complete=True, **native["timing"])),flush=True)
    if not args.collection_only:
        try:
            report["devices"] = update_benchmark(saved,rollout,args.repeats,args.devices)
        except RuntimeError as error:
            report["benchmark_error"] = f"{type(error).__name__}: {error}"
            report["resources_after"] = resources()
            atomic_json(args.output/"result.json",report)
            raise
    report["resources_after"] = resources()
    atomic_json(args.output/"result.json",report)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
