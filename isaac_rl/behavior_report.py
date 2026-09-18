"""Offline diagnostic analysis; never connects to or signals a native worker."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from .behavior_eval import BehaviorSummary, paired_results


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def case_key(row):
    return row["seed"], row.get("repetition", 0), row["arm"]


def ratio(numerator, denominator):
    return numerator/denominator if denominator else None


def mean(rows, key):
    return statistics.mean(r[key] for r in rows) if rows else None


def aggregate(records):
    """Pool transition counts explicitly; do not average episode percentages."""
    if len({case_key(r) for r in records}) != len(records):
        raise ValueError("Duplicate seed/repetition/arm records")
    output = {}
    for arm in sorted({r["arm"] for r in records}):
        rows = [r for r in records if r["arm"] == arm]
        behaviors = [r["behavior"] for r in rows]
        total = lambda key: sum(b[key] for b in behaviors)
        output[arm] = dict(episodes=len(rows), seeds=len({r["seed"] for r in rows}),
            wins=sum(bool(r["success"]) for r in rows),
            boss_encounters=sum(bool(r["boss_seen"]) for r in rows),
            combat_episodes=sum(r["combat_reached"] for r in rows),
            end_reasons=dict(Counter(r["reason"] for r in rows)),
            means={k: mean(rows, k) for k in ("r", "l", "rooms", "combat_clears", "damage_dealt", "damage_taken", "kills")},
            combat_steps=total("combat_steps"), same_room_combat_steps=total("same_room_combat_steps"),
            combat_move_zero_fraction=ratio(total("combat_move_zero"), total("combat_steps")),
            combat_stationary_fraction=ratio(total("combat_stationary"), total("same_room_combat_steps")),
            combat_shooting_fraction=ratio(total("combat_shooting"), total("combat_steps")),
            damage_during_stationary_transitions=total("damage_during_stationary_transitions"),
            damage_during_moving_transitions=total("damage_during_moving_transitions"),
            step_weighted_head_entropy=[ratio(sum(b["mean_entropy"][i]*b["steps"] for b in behaviors), total("steps")) for i in range(3)],
            mean_reward_components={k: sum(b["reward_components"].get(k, 0) for b in behaviors)/len(rows)
                                   for k in sorted({k for b in behaviors for k in b["reward_components"]})})
    pairs = paired_results(records)
    eligible = [p for p in pairs if p["movement_comparison_eligible"]]
    # Repetitions within a seed are not independent samples. Report each seed's
    # mean effect, then aggregate those means with equal seed weight.
    by_seed = defaultdict(list)
    for pair in eligible:
        by_seed[pair["seed"]].append(pair["normal_minus_ablated"])
    seed_effects = [{k: mean(rows, k) for k in rows[0]} for rows in by_seed.values()]
    effects = {k: dict(mean=mean(seed_effects, k), positive=sum(r[k] > 0 for r in seed_effects),
                      zero=sum(r[k] == 0 for r in seed_effects), negative=sum(r[k] < 0 for r in seed_effects))
               for k in (seed_effects[0] if seed_effects else [])}
    return dict(arms=output, movement_comparison=dict(total_pairs=len(pairs), eligible_pairs=len(eligible),
        eligible_seeds=len(by_seed), excluded_pairs=len(pairs)-len(eligible),
        equal_seed_weighted_effects=effects), completion_proven=False)


def analyze_transitions(run, records):
    """Recompute summaries and locate prefix divergence without loading raw traces."""
    stats, health, prefixes, starts = {}, defaultdict(Counter), defaultdict(list), {}
    for row in read_rows(run/"transitions.jsonl"):
        key = case_key(row)
        stats.setdefault(key, BehaviorSummary()).add(row)
        if row["step"] == 1:
            starts[row["before_sequence"]] = key
        if not prefixes[key] or not prefixes[key][-1]["combat_before"]:
            prefixes[key].append(row)
        before, after = row["player_before"], row["player_after"]
        hp_delta = before["hearts"]+before["soul"]-after["hearts"]-after["soul"]
        health[row["arm"]].update(hp_loss_units=max(0, hp_delta), hp_gain_units=max(0, -hp_delta),
            damage_taken_event_units=row["damage_taken"], transitions=1,
            event_hp_delta_mismatch_transitions=abs(row["damage_taken"]-hp_delta) > 1e-6,
            bomb_commands_without_bombs=row["executed"][2] == 1 and before["bombs"] == 0)
    for record in records:
        # JSON object keys (action IDs) are strings after the recorded round trip.
        recomputed = json.loads(json.dumps(stats[case_key(record)].result()))
        if recomputed != record["behavior"]:
            raise ValueError(f"Transition/episode summary mismatch: {case_key(record)}")
    if set(stats) != {case_key(r) for r in records}:
        raise ValueError("Unfinished or unreported transition cases")
    differing, wanted = [], {}
    for key, normal in prefixes.items():
        if key[2] != "stochastic":
            continue
        other = (key[0], key[1], "stochastic_no_move_combat")
        if other not in prefixes:
            continue
        for first, second in zip(normal, prefixes[other]):
            fields = [k for k in ("player_before", "room_before", "probabilities", "combat_before", "proposed")
                      if first[k] != second[k]]
            if fields:
                index = len(differing)
                differing.append(dict(seed=key[0], repetition=key[1], first_step=first["step"], changed_fields=fields))
                wanted[first["before_sequence"]] = (index, 0)
                wanted[second["before_sequence"]] = (index, 1)
                break
    raw, initial = {}, {}
    for row in read_rows(run/"trace.jsonl"):
        if row["direction"] != "game":
            continue
        state = row["message"]
        sequence = state.get("sequence")
        if sequence in wanted:
            raw[wanted[sequence]] = state
        if sequence in starts:
            initial[starts[sequence]] = state
    metadata = {"id", "sequence", "frame", "episode"}
    for index, entry in enumerate(differing):
        first, second = raw[(index, 0)], raw[(index, 1)]
        entry["native_changed_fields_excluding_transport_metadata"] = [k for k in first if k not in metadata and first[k] != second[k]]
        entry["entity_categories_at_divergence"] = sorted({e[4] for s in (first, second) for e in s["entities"]})
    reset_pairs = []
    for key, first in initial.items():
        other = (key[0], key[1], "stochastic_no_move_combat")
        if key[2] == "stochastic" and other in initial:
            second = initial[other]
            reset_pairs.append(dict(seed=key[0], repetition=key[1],
                equal_excluding_transport_metadata=all(first[k] == second[k] for k in first if k not in metadata)))
    return dict(transition_summaries_verified=len(records), health_and_bomb_commands=dict(health),
                initial_stochastic_pairs=reset_pairs, first_prefix_divergences=differing)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def complete_snapshot(path):
    """Hash the exact completed JSONL prefix read, even while a trainer appends."""
    payload = path.read_bytes()
    end = payload.rfind(b"\n")+1
    complete = payload[:end]
    rows = [json.loads(line) for line in complete.splitlines() if line.strip()]
    return rows, dict(path=str(path.resolve()), bytes=end, sha256=hashlib.sha256(complete).hexdigest(),
                      omitted_trailing_bytes=len(payload)-end)


def training_snapshot(run):
    episodes, episode_source = complete_snapshot(run/"episodes.jsonl")
    updates, update_source = complete_snapshot(run/"updates.jsonl")
    windows = []
    # Keep reward profiles separate; do not reinterpret missing old metrics as 0.
    groups = defaultdict(list)
    for row in episodes:
        groups[(row.get("reward_profile", "unknown"), row["steps"]//500000)].append(row)
    for (profile, bucket), rows in sorted(groups.items()):
        windows.append(dict(reward_profile=profile, bucket_start=bucket*500000,
            first_recorded_step=min(r["steps"] for r in rows), last_recorded_step=max(r["steps"] for r in rows),
            episodes=len(rows), wins=sum(bool(r["success"]) for r in rows),
            boss_encounters=sum(bool(r["boss_seen"]) for r in rows),
            end_reasons=dict(Counter(r["reason"] for r in rows)),
            means={k: mean(rows, k) if all(k in r for r in rows) else None
                   for k in ("r", "l", "rooms", "combat_clears", "damage_dealt", "damage_taken", "kills")}))
    latest = updates[-100:]
    return dict(episode_source=episode_source, update_source=update_source, windows=windows,
        last_updates=dict(count=len(latest), first_step=latest[0]["steps"], last_step=latest[-1]["steps"],
            means={k: mean(latest, k) for k in ("loss", "policy_loss", "value_loss", "entropy", "kl", "grad_norm", "explained_variance", "rollout_sps")},
            mean_timing={k: mean([r["timing"] for r in latest], k) for k in latest[0]["timing"]}),
        note="Training aggregates are not held-out outcomes. Final step bucket is partial. SHA256 covers only the recorded byte prefix.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-run", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.run/"manifest.json").read_text())
    status = json.loads((args.run/"status.json").read_text())
    if status["status"] != "finished":
        raise ValueError("Analyze a finished evaluation only")
    records = list(read_rows(args.run/"episodes.jsonl"))
    if len(records) != status["requested_episodes"]:
        raise ValueError("Episode coverage incomplete")
    report = dict(run=str(args.run.resolve()), checkpoint_steps=manifest["checkpoint_steps"],
        checkpoint_sha256=manifest["checkpoint_sha256"], **aggregate(records),
        transition_audit=analyze_transitions(args.run, records),
        evidence_sha256={name: digest(args.run/name) for name in
                         ("manifest.json", "episodes.jsonl", "transitions.jsonl", "trace.jsonl", "result.json")},
        limitations=["Damage/displacement association does not establish aiming or dodging.",
            "Matched observed prefixes do not fix hidden native RNG after intervention.",
            "The eligible movement subset may be selected and is not representative of all seeds.",
            "Stationary means less than one unit of same-room net displacement per action, not zero path length.",
            "Health loss/gain counts net sampled HP changes; within-action losses and pickups can cancel.",
            "No statistical significance or full-floor consistency is established by this diagnostic."])
    if args.training_run:
        report["training_snapshot"] = training_snapshot(args.training_run)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(args.output), arms=report["arms"], movement_comparison=report["movement_comparison"]), indent=2))


if __name__ == "__main__":
    main()
