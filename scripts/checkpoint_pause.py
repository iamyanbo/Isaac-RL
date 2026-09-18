"""Quiet milestone trigger: freeze a snapshot, evaluate, archive distribution, exit.

Only the trigger reads the step counter between milestones; it performs no
intermediate analysis or reporting. Training is never stopped at a checkpoint;
only the agent pauses while the background learner and evaluation keep running.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

from isaac_rl.evaluate import canonical_seed, success_gate
from isaac_rl.storage import atomic_json, load_snapshot
from isaac_rl.analyze import distribution_summary


def evaluation_distribution(records):
    return {"success":success_gate(records),**distribution_summary(records)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--pid",type=int,required=True)
    parser.add_argument("--process-started",type=float,required=True)
    parser.add_argument("--target",type=int,required=True)
    parser.add_argument("--number",type=int,required=True)
    parser.add_argument("--episodes",type=int,default=20)
    parser.add_argument("--port",type=int,default=10002)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.episodes < 20:
        parser.error("Checkpoint evaluations require at least 20 episodes")
    run = args.run.resolve()
    learner = psutil.Process(args.pid)
    command = learner.cmdline()
    if (abs(learner.create_time()-args.process_started) > .01
        or "isaac_rl.train_vector" not in command
        or not any(str(run).casefold() == token.casefold() for token in command)):
        raise RuntimeError("Learner process identity mismatch")
    args.output.mkdir(parents=True,exist_ok=False)
    print("Milestone gate armed; no intermediate progress reports or training stop.",flush=True)
    while True:
        if not learner.is_running() or abs(learner.create_time()-args.process_started) > .01:
            raise RuntimeError("Learner exited before milestone; inspect the actual failure")
        try:
            status = json.loads((run/"status.json").read_text())
        except (OSError,json.JSONDecodeError):
            time.sleep(2)
            continue
        if status["pid"] != args.pid or abs(status["process_started"]-args.process_started) > .01:
            raise RuntimeError("Telemetry identity mismatch")
        if status["steps"] >= args.target:
            break
        time.sleep(2)
    # Freeze one exact saved snapshot while the learner continues. The live
    # counter may precede its next atomic checkpoint by one PPO rollout.
    while True:
        saved,digest,payload = load_snapshot(run/"latest.pt","cpu")
        if saved["steps"] >= args.target:
            break
        time.sleep(2)
    frozen = args.output/"checkpoint.pt"
    frozen.write_bytes(payload)
    metadata = {"checkpoint_number":args.number,"target_aggregate_steps":args.target,
        "aggregate_steps":saved["steps"],"checkpoint_sha256":digest,"training_paused":False,
        "architecture":saved["architecture"],"reward_profile":saved["reward_profile"],
        "observation_profile":saved["observation_profile"],"ports":saved["ports"],
        "constraints":{"architecture_and_reward_frozen":True,"minimum_steps_per_worker_before_change":300000,
            "evaluation_interval_aggregate":100000,"minimum_evaluation_episodes":20,
            "full_reporting_interval_aggregate":500000,"agent_pause_only":True,
            "background_training_must_continue":True}}
    atomic_json(args.output/"checkpoint.json",metadata)
    print(f"Checkpoint {args.number}: frozen {saved['steps']}-action model; background training continues; evaluating {args.episodes} unseen seeds.",flush=True)
    evaluation = args.output/"evaluation"
    with (args.output/"evaluation.stdout.log").open("w") as stdout, (args.output/"evaluation.stderr.log").open("w") as stderr:
        subprocess.run([sys.executable,"-u","-m","isaac_rl.evaluate",str(frozen),"--port",str(args.port),
            "--episodes",str(args.episodes),"--output",str(evaluation)],stdout=stdout,stderr=stderr,check=True)
    records = [json.loads(line) for line in (evaluation/"episodes.jsonl").read_text().splitlines()]
    forbidden = {canonical_seed(seed) for seed in saved["training_seeds"]}
    if len(records) != args.episodes or len({canonical_seed(r["seed"]) for r in records}) != len(records):
        raise RuntimeError("Incomplete or duplicate-seed evaluation")
    if any(canonical_seed(r["seed"]) in forbidden for r in records):
        raise RuntimeError("Evaluation used a checkpoint training seed")
    report = dict(metadata,distribution=evaluation_distribution(records),
        evaluation_complete=True,seeds_unseen=True,next_checkpoint_aggregate=args.target+100000)
    atomic_json(args.output/"report.json",report)
    print(json.dumps({"checkpoint_number":args.number,"aggregate_steps":saved["steps"],
        "evaluation_episodes":len(records),"report":str(args.output/"report.json"),"agent_pause_only":True}),flush=True)


if __name__ == "__main__":
    main()
