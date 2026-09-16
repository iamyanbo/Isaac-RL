"""Frozen parent/treatment comparison on reserved port10002; never optimizes.

Native seeds are selected before scoring. Every requested seed remains in the
denominator, including no-combat episodes. Full native traces are gzip-compressed
without dropping states. This process neither signals nor connects to training.
"""
import argparse
from collections import Counter
import gzip
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import psutil
import torch

from .behavior_eval import run_case
from .bridge import Bridge
from .env import IsaacEnv
from .evaluate import canonical_seed, park_after_evaluation, success_gate
from .ppo import load_policy
from .storage import atomic_json, capture_failure_states, load_snapshot, source_fingerprint
from .timing import configure_evaluation

ARMS = ("parent", "treatment")


class TaggedTransitions:
    def __init__(self, stream, arm):
        self.stream,self.arm = stream,arm

    def write(self, payload):
        record = json.loads(payload)
        record["comparison_arm"] = self.arm
        self.stream.write(json.dumps(record,allow_nan=False)+"\n")


class NativeEvidenceEnv(IsaacEnv):
    """Cross-check the derived clear counter against saved native room states."""
    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        self.initial = self.state
        self.native_combat = {}
        self.native_clears = {}
        self._record_native(None)
        return result

    def _record_native(self, before):
        s = self.state
        room = s["room"]
        identity = room["id"]
        if room["enemies"] > 0 and not room["clear"]:
            self.native_combat.setdefault(identity,s)
        if (identity in self.native_combat and identity not in self.native_clears
                and room["clear"] and room["enemies"] == 0 and not s["player"]["dead"]):
            self.native_clears[identity] = dict(combat_observed=self.native_combat[identity],
                before=before,after=s)

    def step(self, action):
        before = self.state
        result = super().step(action)
        self._record_native(before)
        return result


def settings(saved):
    # Generic evaluation defaults allow a longer idle budget. This comparison
    # explicitly inherits the actual training limits declared in the experiment.
    schedule = saved["training_schedule"]
    args = SimpleNamespace(max_episode_steps=schedule["max_episode_steps"],idle_limit=schedule["idle_limit"])
    control = configure_evaluation(args,saved)
    return dict(frames=saved["frames"],max_steps=args.max_episode_steps,
        idle_limit=args.idle_limit,reward_profile=saved["reward_profile"],timing_profile=control.profile)


def validate_comparison(saved):
    common = settings(saved["parent"])
    if settings(saved["treatment"]) != common:
        raise ValueError("Comparison holds, reward and episode limits must match")
    if saved["parent"]["observation_profile"] != "terrain_v2" or saved["treatment"]["observation_profile"] != "combat_history_v3":
        raise ValueError("Expected terrain_v2 parent and combat_history_v3 treatment")
    return common


def choose_seeds(env, count, forbidden, output, report):
    selected = []
    forbidden = {canonical_seed(s) for s in forbidden}
    for attempt in range(count*1000):
        _,info = env.reset()
        seed = canonical_seed(info["game_seed"])
        if seed not in forbidden:
            forbidden.add(seed)
            selected.append(seed)
            atomic_json(output/"seeds.json",selected)
            report(status="selecting_seeds",selected_seeds=len(selected))
        if len(selected) == count:
            return selected
    raise RuntimeError("Could not select enough unseen native seeds")


def wilson(successes, count):
    if not count:
        return 0.,1.
    p,z = successes/count,1.96
    center = (p+z*z/(2*count))/(1+z*z/count)
    radius = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/(1+z*z/count)
    return max(0.,center-radius),min(1.,center+radius)


def summarize(records, seeds, requested, budget_compliant):
    keys = [(r["seed"],r["arm"]) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate seed/arm result")
    if any(arm not in ARMS or seed not in seeds for seed,arm in keys):
        raise ValueError("Unexpected seed or arm")
    complete = (len(seeds) == requested and len(set(seeds)) == requested and
        set(keys) == {(s,a) for s in seeds for a in ARMS})
    groups = {}
    for arm in ARMS:
        rows = [r for r in records if r["arm"] == arm]
        clears = sum(r["native_combat_clears_alive"] > 0 for r in rows)
        groups[arm] = dict(episodes=len(rows),living_combat_clear_episodes=clears,
            living_combat_clear_rate=clears/len(rows) if rows else None,
            living_combat_clear_wilson=wilson(clears,len(rows)),
            boss_encounters=sum(bool(r["boss_seen"]) for r in rows),
            end_reasons=dict(Counter(r["reason"] for r in rows)),
            boss_success_gate=success_gate(rows))
    valid = all(r["native_counter_agrees"] for r in records)
    comparison = None
    if complete and valid:
        p,t = (groups[a] for a in ARMS)
        upper = t["living_combat_clear_wilson"][1]-p["living_combat_clear_wilson"][0]
        lower = t["living_combat_clear_wilson"][0]-p["living_combat_clear_wilson"][1]
        comparison = dict(delta=t["living_combat_clear_rate"]-p["living_combat_clear_rate"],
            conservative_95_lower=lower,conservative_95_upper=upper,
            upper_below_meaningful_0_10=upper < .10,
            original_fixed_budget_hypothesis_rejected=(upper < .10) if budget_compliant else None,
            interpretation="fixed-budget comparison" if budget_compliant else
                "exploratory late-checkpoint comparison; original fixed-budget claim not testable")
    return dict(complete=complete,completed_episodes=len(records),requested_episodes=requested*2,
        native_counters_valid=valid,budget_compliant=budget_compliant,arms=groups,comparison=comparison,
        mechanism_causally_isolated=False,
        limitation="Frozen parent is not a matched additional-training control; no attribution to observability alone.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review",type=Path,required=True,help="Immutable parent.pt, treatment.pt, review.json and excluded-seeds.json")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seeds",type=int,default=100)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("Seed count must be positive")
    torch.set_num_threads(2)
    review = json.loads((args.review/"review.json").read_text())
    saved,models,snapshots = {},{},{}
    for arm in ARMS:
        path = args.review/f"{arm}.pt"
        checkpoint,digest,_ = load_snapshot(path)
        if digest != review[f"{arm}_sha256"]:
            raise ValueError(f"{arm} snapshot hash differs from frozen review")
        saved[arm] = checkpoint
        model,_ = load_policy(checkpoint)
        if not all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()):
            raise ValueError("Nonfinite checkpoint")
        models[arm] = model.eval()
        snapshots[arm] = dict(path=str(path.resolve()),sha256=digest,steps=checkpoint["steps"])
    common = validate_comparison(saved)
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    forbidden = set(json.loads((args.review/"excluded-seeds.json").read_text()))
    for checkpoint in saved.values():
        forbidden.update(checkpoint["training_seeds"])
    forbidden = {canonical_seed(s) for s in forbidden}
    atomic_json(output/"excluded-seeds.json",sorted(forbidden))
    atomic_json(output/"manifest.json",dict(purpose="held_out_frozen_policy_comparison",port=10002,
        checkpoints=snapshots,seeds_requested=args.seeds,policy="stochastic",settings=common,
        per_seed_action_rng="946513 + zero-based seed index; reset for each arm",
        arm_order="alternating parent/treatment first by seed index",selection="all native seeds selected before scoring",
        primary="at least one independently native-evidenced living combat clear; all requested seeds in denominator",
        budget_compliant=review["budget_compliant"],budget_overrun_decisions=review["budget_overrun_decisions"],
        strict_fixed_budget_falsification_available=review["budget_compliant"],training_transitions=False,
        native_trace="trace.jsonl.gz",transition_trace="transitions.jsonl.gz",
        source_hashes=source_fingerprint(Path(__file__).resolve().parents[1]),started=time.time()))
    status = dict(pid=os.getpid(),process_started=psutil.Process().create_time(),status="connecting",
        requested_episodes=args.seeds*2,completed_episodes=0,cleanup_complete=False,exit_code=None)
    def report(**fields):
        status.update(time=time.time(),**fields)
        atomic_json(output/"status.json",status)
    report()
    bridge,envs,records = None,{},[]
    try:
        bridge = Bridge(port=10002)
        bridge.trace = gzip.open(output/"trace.jsonl.gz","at",encoding="utf-8",compresslevel=1)
        hello = bridge.connect()
        if hello.get("bridge_port") != 10002 or hello.get("combat_schema") != "combat_v2":
            raise RuntimeError("Reserved game needs combat_v2; do not connect a training game")
        for arm in ARMS:
            envs[arm] = NativeEvidenceEnv(bridge,observation_profile=saved[arm]["observation_profile"],**common)
            envs[arm].connected = True  # Exactly one shared native socket; sequential episodes only.
        seeds = choose_seeds(envs["parent"],args.seeds,forbidden,output,report)
        with gzip.open(output/"transitions.jsonl.gz","at",encoding="utf-8",compresslevel=1) as transitions, (output/"episodes.jsonl").open("x",buffering=1) as log:
            for index,seed in enumerate(seeds):
                for arm in ARMS if index%2 == 0 else ARMS[::-1]:
                    env = envs[arm]
                    def case_report(**fields):
                        report(**dict(fields,arm=arm))
                    report(status="resetting",pair=index,seed=seed,arm=arm,episode_steps=0)
                    # Reuse the tested full-floor stochastic rollout without any ablation.
                    result = run_case(env,models[arm],"stochastic",seed,946513+index,index,
                        TaggedTransitions(transitions,arm),case_report)
                    result.update(arm=arm,native_combat_clears_alive=len(env.native_clears),
                        native_counter_agrees=len(env.native_clears)==result["combat_clears_alive"])
                    atomic_json(output/f"pair-{index:03d}-{arm}.json",dict(result=result,
                        initial=env.initial,final=env.state,native_clear_evidence=env.native_clears))
                    records.append(result)
                    transitions.flush()
                    log.write(json.dumps(result,allow_nan=False)+"\n")
                    atomic_json(output/"result.json",summarize(records,seeds,args.seeds,review["budget_compliant"]))
                    report(completed_episodes=len(records))
                    print(json.dumps({k:result[k] for k in ("pair","arm","seed","l","reason","success","native_combat_clears_alive")}),flush=True)
        cleanup = park_after_evaluation(env,output)
        report(status="finished",exit_code=0,parked=cleanup["parked"],ended_at=time.time())
    except Exception as error:
        report(status="error",exit_code=1,error=f"{type(error).__name__}: {error}",ended_at=time.time(),
            **capture_failure_states(output/"failure-states.json",list(envs.values())))
        raise
    finally:
        if bridge is not None:
            bridge.close()
        report(cleanup_complete=True)


if __name__ == "__main__":
    main()
