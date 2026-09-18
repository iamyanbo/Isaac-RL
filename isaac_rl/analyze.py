"""Summarize observed learning without equating shaped reward with game success."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def distribution_summary(records):
    """Keep quantiles and raw samples so tails are not hidden by an average."""
    metrics = {}
    for key in ("r","l","rooms","combat_clears","damage_dealt","damage_taken","kills"):
        values = np.asarray([record[key] for record in records],dtype=float)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(f"Missing or nonfinite evaluation metric: {key}")
        quantiles = np.quantile(values,[0,.05,.25,.5,.75,.95,1])
        metrics[key] = dict(zip(("min","p05","p25","median","p75","p95","max"),map(float,quantiles)))
        metrics[key]["samples"] = sorted(map(float,values))
    return {"end_reasons":dict(Counter(r["reason"] for r in records)),
        "boss_encounters":sum(bool(r["boss_seen"]) for r in records),"metrics":metrics}


def summarize(records):
    if not records:
        return {"episodes":0,"wins":0,"boss_encounters":0}
    profiles = sorted({r.get("reward_profile","legacy_v1") for r in records})
    kills = [r["kills"] for r in records if "kills" in r]
    combat_clears = [r["combat_clears"] for r in records if "combat_clears" in r]
    return {
        "episodes":len(records),"wins":sum(bool(r["success"]) for r in records),
        "boss_encounters":sum(bool(r["boss_seen"]) for r in records),
        "reward_profiles":profiles,
        "mean_reward":round(float(np.mean([r["r"] for r in records])),3) if len(profiles) == 1 else None,
        "median_steps":float(np.median([r["l"] for r in records])),
        "mean_rooms_visited":round(float(np.mean([r["rooms"] for r in records])),3),
        "mean_rooms_marked_clear_including_empty":round(float(np.mean([r["clears"] for r in records])),3),
        "mean_damage_dealt":round(float(np.mean([r["damage_dealt"] for r in records])),3),
        "mean_damage_taken":round(float(np.mean([r["damage_taken"] for r in records])),3),
        "mean_kills":round(float(np.mean(kills)),3) if kills else None,
        "mean_observed_combat_clears":round(float(np.mean(combat_clears)),3) if combat_clears else None,
        "metric_coverage":dict(kill_episodes=len(kills),combat_clear_episodes=len(combat_clears)),
        "end_reasons":dict(Counter(r["reason"] for r in records)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs",type=Path,nargs="+")
    parser.add_argument("--window",type=int,default=20)
    args = parser.parse_args()
    records = []
    for run in args.runs:
        path = run/"episodes.jsonl"
        if path.exists():
            records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    records.sort(key=lambda r:r["time"])
    profiles = sorted({r.get("reward_profile","legacy_v1") for r in records})
    report = {"overall":summarize(records),"by_reward_profile":{
        profile:summarize([r for r in records if r.get("reward_profile","legacy_v1") == profile])
        for profile in profiles},"windows":[]}
    for start in range(0,len(records),args.window):
        batch = records[start:start+args.window]
        report["windows"].append(dict(first_step=batch[0]["steps"],last_step=batch[-1]["steps"],**summarize(batch)))
    report["completion_proven"] = False
    report["note"] = "Training is not held-out evaluation. Marked-clear rooms include empty rooms. Rewards are not averaged across profiles; missing old combat/kill metrics are not fabricated as zero."
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
