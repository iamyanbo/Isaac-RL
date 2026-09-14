"""Run actual neutral/movement/shoot/reset checks; save authoritative raw states."""
import argparse
import json
from pathlib import Path
import time

from isaac_rl.bridge import Bridge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--port",type=int,default=9999)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / "runs" / ("probe" if args.port == 9999 else f"probe-{args.port}")
    output.mkdir(parents=True, exist_ok=True)
    bridge = Bridge(port=args.port,trace=output / "trace.jsonl")
    print(f"Waiting for game bridge at 127.0.0.1:{args.port}", flush=True)
    try:
        first = bridge.connect()
        print("CONNECTED", json.dumps({k: first[k] for k in ["seed", "stage", "episode", "player"]}), flush=True)
        state = bridge.request("reset")
        assert state["episode"] > first["episode"]
        start = state
        trajectory = []
        saw_tears = False
        for i in range(args.steps):
            previous = state
            state = bridge.request("step", action=[1 if i < args.steps // 2 else 2, 3, 0], frames=4)
            assert state["episode"] == start["episode"]
            assert state["frame"]-previous["frame"] == 4
            saw_tears |= any(e[4] == 6 for e in state["entities"])
            trajectory.append([state["frame"],state["player"]["x"],state["player"]["y"]])
            print(json.dumps({"step": i, "frame": state["frame"], "x": state["player"]["x"], "y": state["player"]["y"], "entities": len(state["entities"])}), flush=True)
        (output / "last_state.json").write_text(json.dumps(state, indent=2))
        assert abs(state["player"]["x"] - start["player"]["x"]) > 0.1 or state["room"]["id"] != start["room"]["id"]
        assert saw_tears, "Shooting did not produce tears"
        time.sleep(0.3)
        stationary = bridge.request("ping")
        assert stationary["frame"] == state["frame"], "Simulation advanced while Python was thinking"
        seeded = bridge.request("reset", seed=state["seed"])
        assert seeded["seed"] == state["seed"]
        assert seeded["episode"] > state["episode"]
        assert seeded["doors"] == start["doors"], "Seeded reset changed the starting layout"
        replay = []
        for i in range(args.steps):
            state = bridge.request("step", action=[1 if i < args.steps // 2 else 2,3,0],frames=4)
            replay.append([state["frame"],state["player"]["x"],state["player"]["y"]])
        assert trajectory == replay, "Identical seeded input sequence did not reproduce movement"
        from isaac_rl.storage import source_fingerprint
        (output / "result.json").write_text(json.dumps({"passed": True, "timestamp": time.time(),
            "source_hashes":source_fingerprint(root),"seed":seeded["seed"],"trajectory":trajectory,
            "checks": ["connect", "fresh_reset", "movement", "shooting", "exact_frame_steps", "paused_inference", "seeded_reset", "deterministic_replay"]}, indent=2))
        print("Probe passed", flush=True)
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
