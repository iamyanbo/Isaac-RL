"""Unscored regression: attempted hits on the known Roundy room must not farm HP."""
import argparse
import json
from pathlib import Path

from isaac_rl.bridge import Bridge
from isaac_rl.storage import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",type=int,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    bridge = Bridge(port=args.port,timeout=30,trace=args.output/"trace.jsonl")
    try:
        bridge.connect()
        state = bridge.request("reset",seed="QXDN 0F04")
        assert state.get("damage_signal") == "hp_delta_v1", "Corrected native bridge required"
        for _ in range(100):
            state = bridge.request("step",action=[4,0,0],frames=1)
            if state["room"]["id"] != 84:
                break
        for _ in range(100):
            x = state["player"]["x"]
            if abs(x-640)<5 or state["terminal"]:
                break
            state = bridge.request("step",action=[1 if x>640 else 2,0,0],frames=1)
        initial = state
        enemies = [e for e in initial["entities"] if e[4] == 1]
        assert len(enemies) == 4 and all(e[5] == 276 for e in enemies), "Unexpected native room"
        budget = sum(e[9] for e in enemies)
        for _ in range(100):
            if state["terminal"]:
                break
            state = bridge.request("step",action=[0,4,0],frames=8)
        damage = state["events"]["damage_dealt"]
        attempted = state["events"]["damage_attempted"]
        record = {"purpose":"unscored_native_damage_accounting_regression","scored":False,
            "initial":initial,"final":state,"initial_enemy_hp_budget":budget,
            "reported_damage":damage,"attempted_damage":attempted,
            "blocked_hit_check":0 <= damage <= budget+1e-6 and attempted > damage+10}
        # A separate positive control prevents an always-zero counter passing.
        state = bridge.request("reset",seed="QXDN 0F04")
        for _ in range(100):
            state = bridge.request("step",action=[3,0,0],frames=1)
            if state["room"]["id"] != 84:
                break
        positive_initial = state
        for _ in range(80):
            if state["terminal"] or state["events"]["kills"] > 0:
                break
            state = bridge.request("step",action=[0,3,0],frames=8)
        positive_damage = state["events"]["damage_dealt"]
        positive_kills = state["events"]["kills"]
        atomic_json(args.output/"positive-control.json",{"initial":positive_initial,"final":state,"scored":False})
        record.update(positive_damage=positive_damage,positive_kills=positive_kills,
            passed=record["blocked_hit_check"] and positive_damage>0 and positive_kills>0)
        atomic_json(args.output/"result.json",record)
        bridge.request("reset")
        print(json.dumps({key:record[key] for key in ("passed","scored","reported_damage","attempted_damage",
            "initial_enemy_hp_budget","positive_damage","positive_kills")}),flush=True)
        assert record["passed"], "Damage accounting violated the known room HP budget"
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
