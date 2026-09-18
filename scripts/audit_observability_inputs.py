"""Offline migration/throughput checks on frozen native smoke states.

Synthetic repeated batches measure cost only; disposable optimizer updates do
not enter a checkpoint or training dataset, and are not a gameplay evaluation.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from isaac_rl.history import ObservationHistory
from isaac_rl.observation import encode
from isaac_rl.ppo import load_policy, optimize
from isaac_rl.storage import atomic_json, load_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline",type=Path)
    parser.add_argument("native_states",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    saved,digest,_ = load_snapshot(args.baseline)
    states = json.loads(args.native_states.read_text())
    history = ObservationHistory()
    inputs = {"terrain_v2":[],"combat_history_v3":[]}
    encoding_times = {key:[] for key in inputs}
    for state in states:
        for profile in inputs:
            start = time.perf_counter()
            obs = encode(state,Counter(),profile,.5) if profile == "terrain_v2" else history.append(state,Counter(),.5,[0,2,0])
            encoding_times[profile].append(time.perf_counter()-start)
            inputs[profile].append(obs)
    reports,outputs = {},{}
    for profile,observations in inputs.items():
        model,optimizer = load_policy(saved,observation_profile=profile,with_optimizer=True)
        obs = {key:np.stack([o[key] for o in observations]) for key in ("grid","vector")}
        with torch.no_grad():
            logits,values = model({key:torch.as_tensor(v) for key,v in obs.items()})
        outputs[profile] = (logits,values)
        six = {key:np.repeat(value[:1],6,axis=0) for key,value in obs.items()}
        tensor = {key:torch.as_tensor(v) for key,v in six.items()}
        inference = []
        for _ in range(20):
            start = time.perf_counter()
            with torch.no_grad():
                model(tensor)
            inference.append(time.perf_counter()-start)
        torch.manual_seed(71835)
        np.random.seed(71835)
        with torch.no_grad():
            actions,logp,_,values = model.act(tensor)
        rollout = dict(obs=[six]*256,actions=[actions.numpy()]*256,
            log_probs=[logp.numpy()]*256,values=[values.numpy()]*256,next_values=[values.numpy()]*256,
            rewards=[np.full(6,-.001,np.float32)]*256,
            terminated=[np.zeros(6,bool)]*256,ended=[np.zeros(6,bool)]*256)
        timings = []
        for _ in range(3):
            # Reload exact same parent for each disposable benchmark repeat.
            model,optimizer = load_policy(saved,observation_profile=profile,with_optimizer=True)
            np.random.seed(71835)
            start = time.perf_counter()
            metrics = optimize(model,optimizer,rollout,batch_size=256,epochs=4,entropy_coef=saved["entropy_coef"],
                gamma=saved["control_timing"]["gamma"],lam=saved["training_schedule"]["gae_lambda"]
                if "gae_lambda" in saved["training_schedule"] else saved["control_timing"]["gae_lambda"])
            timings.append(time.perf_counter()-start)
        reports[profile] = dict(parameters=sum(p.numel() for p in model.parameters()),
            encoding_median_ms=float(np.median(encoding_times[profile])*1000),
            six_inference_median_ms=float(np.median(inference)*1000),
            ppo_update_seconds=timings,rollout_observation_mib=sum(v.nbytes for v in six.values())*256/2**20,
            finite_metrics=all(np.isfinite(v) for v in metrics.values()))
    differences = {name:float((a-b).abs().max()) for name,a,b in zip(("logits","values"),outputs["terrain_v2"],outputs["combat_history_v3"])}
    for a,b in zip(outputs["terrain_v2"],outputs["combat_history_v3"]):
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)
    result = dict(purpose="offline_migration_and_synthetic_copy_cost_only",baseline_sha256=digest,
        baseline_steps=saved["steps"],device="cpu",torch_threads=2,observations=len(states),
        initial_policy_max_abs_differences=differences,profiles=reports,
        warning="Repeated smoke observations and synthetic rewards; no learning-quality or live collection claim.")
    atomic_json(args.output,result)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
