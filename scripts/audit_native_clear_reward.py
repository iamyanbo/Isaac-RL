"""Replay frozen native evidence through both reward profiles, without a game.

No optimizer or policy actions are generated. Observation encoding is skipped
only for speed; its equivalence is separately covered by rich-observation tests.
All original native states, executed actions, timing, reward components and
episode boundaries are checked against the recorded comparison.
"""
import argparse
import gzip
import json
import math
from pathlib import Path

from isaac_rl.env import IsaacEnv
from isaac_rl.storage import atomic_json,sha256


class ReplayBridge:
    state = None
    def connect(self): return self.state
    def request(self, op, **fields): return self.state
    def close(self): pass


class RewardReplay(IsaacEnv):
    def _observation(self, action=None): return None


def audit(comparison, output):
    records = {}
    for path in comparison.glob("pair-*.json"):
        case = json.loads(path.read_text())
        records[case["initial"]["episode"]] = case
    changed,steps,completed = [],0,set()
    envs,active_episode = None,None
    with gzip.open(comparison/"trace.jsonl.gz","rt",encoding="utf-8") as native, gzip.open(comparison/"transitions.jsonl.gz","rt",encoding="utf-8") as transitions:
        for line in native:
            entry = json.loads(line)
            if entry["direction"] != "game": continue
            s = entry["message"]
            episode = s.get("episode")
            case = records.get(episode)
            if case is None or not case["initial"]["sequence"] <= s.get("sequence",-1) <= case["final"]["sequence"]:
                continue
            if s["sequence"] == case["initial"]["sequence"]:
                assert active_episode is None or active_episode in completed
                active_episode = episode
                envs = []
                for profile in ("completion_v4","native_clear_v5"):
                    bridge = ReplayBridge();bridge.state = s
                    env = RewardReplay(bridge,frames=4,max_steps=6750,idle_limit=900,
                        reward_profile=profile,observation_profile=case["result"]["observation_profile"],timing_profile="physical_v1")
                    env.reset();envs.append(env)
                continue
            assert active_episode == episode
            transition = json.loads(next(transitions))
            result = case["result"]
            before = envs[0].state
            assert transition["before_sequence"] == before["sequence"]
            assert transition["after_sequence"] == s["sequence"]
            assert transition["pair"] == result["pair"] and transition["comparison_arm"] == result["arm"]
            for env in envs: env.bridge.state = s
            old,new = [env.step(transition["executed"]) for env in envs]
            assert old[2:4] == new[2:4] and envs[0].idle == envs[1].idle
            original,updated = old[4]["reward_components"],new[4]["reward_components"]
            for key in original:
                assert math.isclose(original[key],transition["reward_components"][key],rel_tol=0,abs_tol=1e-9),(episode,key)
                if key != "room_clear": assert original[key] == updated[key],(episode,key)
            assert math.isclose(old[1],transition["reward"],rel_tol=0,abs_tol=1e-9)
            if original["room_clear"] != updated["room_clear"]:
                assert original["room_clear"] == 10 and updated["room_clear"] == 0
                assert s["room"]["clear"] and s["room"]["id"] not in envs[1].combat_rooms
                changed.append(dict(pair=result["pair"],arm=result["arm"],step=envs[0].steps,
                    seed=result["seed"],before_sequence=before["sequence"],after_sequence=s["sequence"],
                    before=before,after=s,old_components=original,new_components=updated))
            steps += 1
            if s["sequence"] == case["final"]["sequence"]:
                assert old[2] or old[3]
                for key,value in old[4]["episode"].items():
                    expected = result[key]
                    if key == "seed":
                        assert value.replace(" ","") == expected.replace(" ","")
                    elif isinstance(value,float):
                        assert math.isclose(value,expected,rel_tol=0,abs_tol=1e-9),(episode,key,value,expected)
                    else:
                        assert value == expected,(episode,key,value,expected)
                assert envs[0].steps == result["l"]
                completed.add(episode)
                if len(completed)%25 == 0: print(f"replayed {len(completed)} episodes / {steps} actions",flush=True)
        assert next(transitions,None) is None
    assert len(completed) == len(records) == 200
    report = dict(passed=True,episodes=len(completed),actions=steps,changed_reward_events=len(changed),
        original_rewards_match_trace=True,all_other_reward_components_identical=True,
        all_termination_truncation_and_idle_states_identical=True,training_transitions=False,
        source_trace_sha256=sha256(comparison/"trace.jsonl.gz"),changes=changed)
    atomic_json(output,report)
    print(json.dumps({k:v for k,v in report.items() if k != "changes"},indent=2),flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    audit(args.comparison,args.output)
