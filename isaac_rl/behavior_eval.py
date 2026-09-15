"""Frozen, paired gameplay diagnostics on a reserved native worker.

Default: deterministic, stochastic, and combat-only movement suppression.
Optional uniform-random controls and within-seed repetitions use the same native
environment and action space. No optimizer, demonstrations, hidden-state
targeting, or training-port connections are used.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import psutil
import torch
from torch.distributions import Categorical

from .bridge import Bridge
from .env import IsaacEnv
from .evaluate import canonical_seed, park_after_evaluation
from .observation import resolve_observation_profile
from .ppo import load_policy, as_tensor
from .storage import atomic_json, load_snapshot, source_fingerprint
from .timing import configure_evaluation


ARMS = ("deterministic", "stochastic", "stochastic_no_move_combat")
AVAILABLE_ARMS = (*ARMS, "uniform_random")


def native_seed(seed):
    """The native console requires two groups; canonical keys omit the space."""
    canonical = canonical_seed(seed)
    if len(canonical) != 8 or not canonical.isascii() or not canonical.isalnum():
        raise ValueError("Expected an eight-character normal game seed")
    return canonical[:4]+" "+canonical[4:]


def intervention(proposed, state, arm):
    executed = np.asarray(proposed, dtype=np.int64).copy()
    if arm == "stochastic_no_move_combat" and state["room"]["enemies"] > 0:
        executed[0] = 0
    return executed


def transition_metrics(before, after, proposed, executed, probabilities, entropy, reward, components):
    """Attribution-neutral events; displacement excludes teleports across rooms."""
    same_room = before["room"]["id"] == after["room"]["id"]
    p, q = before["player"], after["player"]
    displacement = math.hypot(q["x"]-p["x"], q["y"]-p["y"]) if same_room else None
    enemies = [e for e in before["entities"] if e[4] == 1]
    threats = [e for e in before["entities"] if e[4] in (1, 2)]
    nearest_threat = min((math.hypot(e[0]-p["x"], e[1]-p["y"])-e[10]
                          for e in threats), default=None)
    # Geometric alignment only: does not model obstacles, hit timing, invulnerability,
    # or tear inheritance from player velocity, and is not scored as competence.
    direction = int(executed[1])
    aligned = any((direction == 1 and e[0] < p["x"] and abs(e[1]-p["y"]) <= e[10]+7)
                  or (direction == 2 and e[0] > p["x"] and abs(e[1]-p["y"]) <= e[10]+7)
                  or (direction == 3 and e[1] < p["y"] and abs(e[0]-p["x"]) <= e[10]+7)
                  or (direction == 4 and e[1] > p["y"] and abs(e[0]-p["x"]) <= e[10]+7)
                  for e in enemies)
    deltas = {key: after["events"].get(key, 0)-before["events"].get(key, 0)
              for key in ("damage_dealt", "damage_attempted", "damage_taken", "kills")}
    return dict(proposed=np.asarray(proposed).tolist(), executed=np.asarray(executed).tolist(),
        intervention_applied=bool(np.any(np.asarray(proposed) != executed)),
        probabilities=probabilities, entropy=entropy, reward=reward, reward_components=components,
        combat_before=before["room"]["enemies"] > 0, combat_after=after["room"]["enemies"] > 0,
        room_before=before["room"]["id"], room_after=after["room"]["id"],
        player_before=p, player_after=q, displacement=displacement,
        frame_delta=after["episode_frame"]-before["episode_frame"],
        nearest_threat_distance=nearest_threat, shot_geometrically_aligned=aligned,
        **deltas)


class BehaviorSummary:
    def __init__(self):
        self.steps = self.combat = self.stationary = self.move_zero = self.shooting = 0
        self.aligned = self.interventions = self.room_changes = self.displacement_count = 0
        self.displacement = self.combat_damage = self.combat_hurt = 0.0
        self.damage_while_stationary = self.damage_while_moving = 0.0
        self.components = Counter()
        self.actions = [Counter(), Counter(), Counter()]
        self.entropy = np.zeros(3)

    def add(self, record):
        self.steps += 1
        self.interventions += record["intervention_applied"]
        self.room_changes += record["room_before"] != record["room_after"]
        self.components.update(record["reward_components"])
        self.entropy += np.asarray(record["entropy"])
        for head, action in enumerate(record["executed"]):
            self.actions[head][int(action)] += 1
        if record["combat_before"]:
            self.combat += 1
            self.move_zero += record["executed"][0] == 0
            self.shooting += record["executed"][1] != 0
            self.aligned += record["shot_geometrically_aligned"]
            self.combat_damage += record["damage_dealt"]
            self.combat_hurt += record["damage_taken"]
            distance = record["displacement"]
            if distance is not None:
                self.displacement_count += 1
                self.displacement += distance
                self.stationary += distance < 1.0
                if distance < 1.0:
                    self.damage_while_stationary += record["damage_dealt"]
                else:
                    self.damage_while_moving += record["damage_dealt"]

    def result(self):
        return dict(steps=self.steps, combat_steps=self.combat,
            combat_move_zero=self.move_zero, combat_shooting=self.shooting,
            combat_stationary=self.stationary, same_room_combat_steps=self.displacement_count,
            combat_displacement=self.displacement, combat_damage=self.combat_damage,
            combat_damage_taken=self.combat_hurt,
            damage_during_stationary_transitions=self.damage_while_stationary,
            damage_during_moving_transitions=self.damage_while_moving,
            geometrically_aligned_combat_steps=self.aligned,
            intervention_steps=self.interventions, room_changes=self.room_changes,
            mean_entropy=(self.entropy/max(self.steps, 1)).tolist(),
            action_counts=[dict(c) for c in self.actions], reward_components=dict(self.components))


def paired_results(records):
    pairs = []
    for seed, repetition in sorted({(r["seed"], r.get("repetition", 0)) for r in records}):
        arms = {r["arm"]: r for r in records
                if r["seed"] == seed and r.get("repetition", 0) == repetition}
        normal, ablated = arms.get("stochastic"), arms.get("stochastic_no_move_combat")
        if normal is None or ablated is None:
            continue
        prefix_matched = (normal["prefix_hash"] == ablated["prefix_hash"] and
                          normal["prefix_steps"] == ablated["prefix_steps"])
        eligible = prefix_matched and normal["combat_reached"] and ablated["combat_reached"]
        pairs.append(dict(seed=seed, repetition=repetition, prefix_matched=prefix_matched,
            both_reached_combat=normal["combat_reached"] and ablated["combat_reached"],
            movement_comparison_eligible=eligible,
            normal_minus_ablated={key: float(normal[key])-float(ablated[key])
                for key in ("r", "l", "damage_dealt", "damage_taken", "kills", "combat_clears", "boss_seen", "success")}))
    return pairs


def summary(records, requested_seeds, arms=ARMS, repetitions=1):
    expected = requested_seeds*len(arms)*repetitions
    keys = {(r["seed"], r.get("repetition", 0), r["arm"]) for r in records}
    seeds = {r["seed"] for r in records}
    complete = (len(records) == expected and len(seeds) == requested_seeds and
                keys == {(seed, repeat, arm) for seed in seeds
                         for repeat in range(repetitions) for arm in arms})
    return dict(complete=complete,
        purpose="diagnostic_only", checkpoint_consistency_proven=False,
        completed_episodes=len(records), requested_episodes=expected,
        arms={arm: dict(episodes=len(rows), wins=sum(bool(r["success"]) for r in rows),
            boss_encounters=sum(bool(r["boss_seen"]) for r in rows),
            end_reasons=dict(Counter(r["reason"] for r in rows)),
            mean_return=sum(r["r"] for r in rows)/len(rows) if rows else None)
            for arm in arms for rows in [[r for r in records if r["arm"] == arm]]},
        pairs=paired_results(records))


def run_case(env, model, arm, seed, rng_seed, pair, stream, report, repetition=0):
    if arm not in AVAILABLE_ARMS:
        raise ValueError(f"Unknown behavior arm: {arm}")
    obs, _ = env.reset(options={"game_seed": native_seed(seed)})
    initial = env.state
    # Same random samples in the two stochastic arms before any intervention.
    torch.manual_seed(rng_seed)
    prefix = hashlib.sha256()
    prefix_steps, combat_reached = 0, False
    stats = BehaviorSummary()
    while True:
        before = env.state
        combat_reached |= before["room"]["enemies"] > 0
        if not combat_reached:
            prefix_steps += 1
            for key in ("grid", "vector"):
                prefix.update(np.ascontiguousarray(obs[key]).tobytes())
        elif stats.combat == 0:
            # Include the entry observation to validate equivalent combat starts.
            for key in ("grid", "vector"):
                prefix.update(np.ascontiguousarray(obs[key]).tobytes())
        with torch.inference_mode():
            logits, _ = model(as_tensor(obs))
            heads = [Categorical(logits=torch.zeros_like(x) if arm == "uniform_random" else x)
                     for x in logits[0].split([9, 5, 4])]
            proposed = np.array([int(d.probs.argmax()) if arm == "deterministic" else int(d.sample()) for d in heads])
            probabilities = [d.probs.tolist() for d in heads]
            entropy = [float(d.entropy()) for d in heads]
        executed = intervention(proposed, before, arm)
        obs, reward, terminated, truncated, info = env.step(executed)
        record = transition_metrics(before, env.state, proposed, executed, probabilities, entropy,
                                    reward, info["reward_components"])
        stats.add(record)
        stream.write(json.dumps(dict(pair=pair, repetition=repetition, arm=arm, seed=seed, step=env.steps,
            probability_source="uniform_random" if arm == "uniform_random" else "checkpoint",
            before_sequence=before["sequence"], after_sequence=env.state["sequence"],
            time=time.time(), **record), allow_nan=False)+"\n")
        if env.steps % 8 == 0:
            report(status="evaluating", pair=pair, arm=arm, seed=seed,
                   episode_steps=env.steps, combat_steps=stats.combat)
        if terminated or truncated:
            outcome = info["episode"]
            if outcome["success"]:
                final = env.state
                if not (initial["stage"] == 1 and final["stage"] == 1 and final["room"]["type"] == 5
                        and final["room"]["clear"] and final["room"]["enemies"] == 0
                        and final["boss_seen"] and final["boss_defeated"] and not final["player"]["dead"]):
                    raise RuntimeError("Win contradicted by native state")
            return dict(pair=pair, repetition=repetition, arm=arm, seed=seed, rng_seed=rng_seed,
                prefix_hash=prefix.hexdigest(), prefix_steps=prefix_steps,
                combat_reached=combat_reached, behavior=stats.result(), **{k:v for k,v in outcome.items() if k != "seed"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=10002)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--arms", nargs="+", choices=AVAILABLE_ARMS, default=list(ARMS))
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--idle-limit", type=int)
    args = parser.parse_args()
    if min(args.seeds, args.repetitions) < 1 or any(v is not None and v < 1 for v in (args.max_episode_steps,args.idle_limit)):
        parser.error("Counts and limits must be positive")
    if len(set(args.arms)) != len(args.arms):
        parser.error("Arms must be unique")
    torch.set_num_threads(2)
    saved, digest, payload = load_snapshot(args.checkpoint)
    control = configure_evaluation(args,saved)
    model,_ = load_policy(saved)
    if not all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()):
        raise ValueError("Nonfinite checkpoint")
    model.eval()
    profile = resolve_observation_profile(checkpoint=saved)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/"checkpoint.pt").write_bytes(payload)
    manifest = dict(purpose="diagnostic_only", checkpoint_steps=saved["steps"],
        checkpoint_sha256=digest, checkpoint_source=str(args.checkpoint.resolve()),
        frames=saved["frames"], max_episode_steps=args.max_episode_steps, idle_limit=args.idle_limit,
        timing_profile=control.profile,control_timing=control.manifest(),
        seeds_requested=args.seeds, arms=args.arms, repetitions=args.repetitions,
        port=args.port, observation_profile=profile,
        reward_profile=saved.get("reward_profile", "legacy_v1"),
        source_hashes=source_fingerprint(Path(__file__).resolve().parents[1]),
        intervention="Suppress only movement when the BEFORE observation contains enemies; other heads unchanged.",
        pairing="Same seed and action RNG seed; observation prefix hash includes combat entry. Exclude unmatched/no-combat pairs.",
        random_control="uniform_random samples each of the same 9/5/4 action heads uniformly, with the same action hold and no heuristic.",
        limitations=["Seeds, not repeated episodes, are the independent comparison units; native RNG can diverge before an intervention.",
            "Paired action RNG samples do not fix state-dependent trajectories.",
            "Damage during stationary transitions is association, not tear/source causation.",
            "No movement from spawn would confound navigation; this intervention is combat-only.",
            "No-movement counterfactual is not a learned-policy success evaluation."], started=time.time())
    atomic_json(output/"manifest.json", manifest)
    status = dict(pid=os.getpid(), process_started=psutil.Process().create_time(),
        status="connecting", time=time.time(), completed_episodes=0,
        requested_episodes=args.seeds*len(args.arms)*args.repetitions)

    def report(**fields):
        status.update(time=time.time(), **fields)
        atomic_json(output/"status.json", status)

    report()
    forbidden = {canonical_seed(seed) for seed in saved["training_seeds"]}
    records, chosen = [], []
    env = None
    try:
        env = IsaacEnv(Bridge(port=args.port, trace=output/"trace.jsonl"), frames=saved["frames"],
            max_steps=args.max_episode_steps, idle_limit=args.idle_limit,
            reward_profile=manifest["reward_profile"], observation_profile=profile,timing_profile=control.profile)
        with (output/"transitions.jsonl").open("x", buffering=1) as stream, (output/"episodes.jsonl").open("x", buffering=1) as episodes:
            for pair in range(args.seeds):
                for _ in range(1000):
                    _, info = env.reset()
                    seed = canonical_seed(info["game_seed"])
                    if seed not in forbidden:
                        break
                else:
                    raise RuntimeError("Could not discover an unseen seed")
                forbidden.add(seed)
                chosen.append(seed)
                atomic_json(output/"seeds.json", chosen)
                for repetition in range(args.repetitions):
                    # Counterbalance order across seeds and repetitions.
                    offset = (pair+repetition) % len(args.arms)
                    order = args.arms[offset:]+args.arms[:offset]
                    for arm in order:
                        report(status="resetting", pair=pair, repetition=repetition, arm=arm,
                               seed=seed, episode_steps=0, combat_steps=0)
                        result = run_case(env, model, arm, seed, 946513+pair*args.repetitions+repetition,
                                          pair, stream, report, repetition=repetition)
                        records.append(result)
                        episodes.write(json.dumps(result, allow_nan=False)+"\n")
                        atomic_json(output/f"pair-{pair:03d}-repeat-{repetition:02d}-{arm}.json",
                                    dict(result=result, final=env.state))
                        atomic_json(output/"result.json", summary(records, args.seeds, args.arms, args.repetitions))
                        report(completed_episodes=len(records))
                        print(json.dumps(dict(pair=pair, repetition=repetition, arm=arm, seed=seed,
                            reason=result["reason"], combat_steps=result["behavior"]["combat_steps"],
                            success=result["success"])), flush=True)
        park_after_evaluation(env, output)
        report(status="finished", exit_code=0, ended_at=time.time())
    except Exception as error:
        report(status="error", exit_code=1, error=f"{type(error).__name__}: {error}", ended_at=time.time())
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
