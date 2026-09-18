"""Diagnostic only: verify real open-door observations and physical passage.

No policy or optimizer is used. Scripted probe actions never enter training.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from isaac_rl.bridge import Bridge
from isaac_rl.observation import encode, HEIGHT, WIDTH
from isaac_rl.storage import atomic_json, source_fingerprint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",type=int,default=10002)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root/"runs"/f"terrain-probe-{time.time_ns()}"
    output.mkdir(parents=True)
    bridge = Bridge(args.port,trace=output/"trace.jsonl")
    result = {"purpose":"scripted bridge diagnostic, not trained/evaluated policy", "port":args.port}
    try:
        bridge.connect()
        state = initial = bridge.request("reset")
        assert state["stage"] == 1 and state["difficulty"] == 0 and state["character"] == 0
        assert state["player"]["hearts"] == 6 and state["player"]["items"] == 0
        legacy = encode(state,Counter(),"legacy_v1")
        fixed = encode(state,Counter(),"terrain_v2")
        doors = [d for d in state["doors"] if d["open"] and not d["locked"]]
        assert doors, "Normal starting room unexpectedly has no open door"
        checked = []
        left,top,right,bottom = state["room"]["bounds"]
        for door in doors:
            cell = min(state["grid"],key=lambda g:(g[0]-door["x"])**2+(g[1]-door["y"])**2)
            assert cell[2] == 16 and cell[3] == 5, f"Unexpected open door grid state: {cell}"
            row = int(np.clip((door["y"]-top)/(bottom-top)*(HEIGHT-1),0,HEIGHT-1))
            col = int(np.clip((door["x"]-left)/(right-left)*(WIDTH-1),0,WIDTH-1))
            assert legacy["grid"][4,row,col] == 1
            assert fixed["grid"][4,row,col] == 0
            assert fixed["grid"][7,row,col] == 1
            checked.append(dict(slot=door["slot"],grid_collision=cell[3],legacy_solid=True,corrected_solid=False))
        target = min(doors,key=lambda d:(d["x"]-state["player"]["x"])**2+(d["y"]-state["player"]["y"])**2)
        for step in range(100):
            p = state["player"]
            dx,dy = target["x"]-p["x"],target["y"]-p["y"]
            if target["slot"] in (1,3,5,7):
                move = (2 if dx > 0 else 1) if abs(dx) > 6 else (4 if dy > 0 else 3)
            else:
                move = (4 if dy > 0 else 3) if abs(dy) > 6 else (2 if dx > 0 else 1)
            state = bridge.request("step",action=[move,0,0],frames=1)
            if state["room"]["id"] != initial["room"]["id"]:
                break
            assert not state["terminal"], "Probe ended before reaching door"
        assert state["room"]["id"] == target["target"], "Did not cross the selected open door"
        result.update(passed=True,seed=initial["seed"],checked_doors=checked,
                      selected_door=target["slot"],actions=step+1,
                      initial=initial,final=state,source_hashes=source_fingerprint(root))
        atomic_json(output/"result.json",result)
        print(json.dumps({key:value for key,value in result.items() if key not in ("initial","final","source_hashes")},indent=2))
        print(f"Evidence: {output}",flush=True)
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
