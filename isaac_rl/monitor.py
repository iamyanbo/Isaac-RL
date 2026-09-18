"""Session-bound, size-aware live telemetry with bounded rolling history."""
import argparse
from collections import deque
import json
import math
from pathlib import Path
import statistics
import time

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from .lifecycle import SessionWatch, lifecycle


def read_state(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class JsonlWindow:
    """Tail incrementally, retaining only N complete records."""
    def __init__(self, path, limit):
        self.path, self.limit = Path(path), limit
        self.rows = deque(maxlen=limit)
        self.offset = 0
        self.file_id = None

    def poll(self):
        try:
            stat = self.path.stat()
            file_id = (stat.st_dev, stat.st_ino)
            if self.file_id != file_id or stat.st_size < self.offset:
                self.rows.clear()
                self.offset = 0
            self.file_id = file_id
            with self.path.open("rb") as stream:
                stream.seek(self.offset)
                while True:
                    start = stream.tell()
                    line = stream.readline()
                    if not line or not line.endswith(b"\n"):
                        self.offset = start
                        break
                    self.offset = stream.tell()
                    try:
                        self.rows.append(json.loads(line))
                    except (ValueError, UnicodeDecodeError):
                        continue
        except OSError:
            pass
        return list(self.rows)


def numbers(rows, key):
    return [float(r[key]) for r in rows if isinstance(r.get(key),(int,float)) and math.isfinite(r[key])]


def spark(values, width):
    if not values:
        return "waiting for samples"
    width = max(1,width)
    if len(values) > width:
        values = [statistics.mean(values[i*len(values)//width:(i+1)*len(values)//width]) for i in range(width)]
    low, high = min(values), max(values)
    chars = "._:-=+*#%@"
    return "".join(chars[4 if high == low else min(9,int((v-low)/(high-low)*9))] for v in values)


def numeric(value, digits=2):
    return f"{value:.{digits}f}" if isinstance(value,(int,float)) and math.isfinite(value) else "--"


def display(path, state=None, episodes=None, updates=None, width=80, height=24, window=50, notice=""):
    state = state if state is not None else read_state(path)
    state = state or {}
    if episodes is None:
        episodes = JsonlWindow(path.parent/"episodes.jsonl",window).poll()
    if updates is None:
        updates = JsonlWindow(path.parent/"updates.jsonl",window).poll()
    health,phase,age = lifecycle(state)
    styles = {"ALIVE":"green","IDLE":"yellow","STALE":"yellow","EXITING":"yellow","DEAD":"red","UNKNOWN":"yellow"}
    def line(text, style=""):
        return Text(text,style=style,no_wrap=True,overflow="ellipsis")
    rows = [
        line(f"{health} | {phase} | PID {state.get('pid','--')} | heartbeat {age:.1f}s",styles[health]),
        line(f"Steps {state.get('steps',0):,}   Episodes {state.get('episodes',0):,}"),
        line(f"Updates {state.get('updates',0):,}   Workers {state.get('num_envs',0)}   {numeric(state.get('sps'))} steps/s"),
        line(f"Reward {numeric(state.get('episode_reward'))}   Mean {numeric(state.get('mean_reward'))}   Wins {state.get('recent_successes',0)}/{state.get('recent_episodes',0)}"),
        line(f"Loss {numeric(state.get('loss'),5)}   Entropy {numeric(state.get('entropy'),3)}   KL {numeric(state.get('kl'),5)}"),
    ]
    target = state.get("step_target")
    cap = "uncapped" if target == 0 else f"{target:,}" if isinstance(target,int) else "legacy/unknown"
    rows.insert(3,line(f"Target {cap} | saved {state.get('checkpoint_steps','--')} | {state.get('device','--')}"))
    timing = state.get("timing",{})
    rows.append(line(f"Collect {numeric(timing.get('collection_s'),3)}s  Infer {numeric(timing.get('inference_s'),3)}s  PPO {numeric(timing.get('update_s'),3)}s"))
    if state.get("commit_memory") and height >= 24:
        memory = state["commit_memory"]
        rows.append(line(f"Windows commit {memory['percent']:.1f}% | headroom {memory['available_gib']:.2f} GiB",
            "yellow" if memory["percent"] > 90 else "dim"))
    rows.append(line(f"Rolling window: last {window} episodes / updates (oldest -> newest)","cyan"))
    chart_width = max(8,width-40)
    for label,data in (("Reward",numbers(episodes,"r")),("Loss",numbers(updates,"loss")),("Steps/s",numbers(updates,"rollout_sps"))):
        scale = f"  [{min(data):.2g}, {max(data):.2g}]" if data else ""
        rows.append(line(f"{label:7} {spark(data,chart_width)}{scale}"))
        if data and height >= 28:
            rows.append(line(f"        n={len(data)}  min={min(data):.3f}  mean={statistics.mean(data):.3f}  max={max(data):.3f}","dim"))
    rewards = numbers(episodes,"r")
    if rewards:
        rows.append(line(f"Window reward: median {statistics.median(rewards):.2f}  min {min(rewards):.2f}  max {max(rewards):.2f}"))
    workers = state.get("workers",[])
    slots = max(0,height-4-len(rows)-2)
    for i in range(0,min(len(workers),slots*2),2):
        rows.append(line("  ".join(f":{w['port']} HP {w.get('health','?')} R {numeric(w.get('episode_reward'))}" for w in workers[i:i+2])))
    message = notice or state.get("message","Waiting for a learner; no active telemetry yet")
    rows.append(line(str(message),"dim"))
    return Panel(Group(*rows[:max(1,height-4)]),title="Isaac RL | PPO",subtitle=path.parent.name,padding=(0,1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run",type=Path)
    parser.add_argument("--once",action="store_true")
    parser.add_argument("--window",type=int,default=50)
    parser.add_argument("--follow",action="store_true",help="Explicitly keep watching future learner sessions")
    parser.add_argument("--pid",type=int)
    parser.add_argument("--process-started",type=float)
    parser.add_argument("--startup-timeout",type=float,default=30)
    args = parser.parse_args()
    if args.window < 1 or args.startup_timeout <= 0 or bool(args.pid) != bool(args.process_started):
        parser.error("Positive window/timeout required; --pid and --process-started must be supplied together")
    console = Console()
    path = args.run/"status.json"
    episode_window = JsonlWindow(args.run/"episodes.jsonl",args.window)
    update_window = JsonlWindow(args.run/"updates.jsonl",args.window)
    watcher = SessionWatch((args.pid,args.process_started) if args.pid else None,args.follow,args.startup_timeout)
    def render(state=None, notice=""):
        return display(path,state,episode_window.poll(),update_window.poll(),console.size.width,console.size.height,args.window,notice)
    if args.once:
        console.print(render())
        return
    final_notice, shown = "", None
    try:
        with Live(console=console,refresh_per_second=2,screen=True,vertical_overflow="crop") as live:
            while True:
                shown,done,final_notice = watcher.observe(read_state(path))
                live.update(render(shown or {},final_notice))
                if done:
                    break
                time.sleep(.5)
    except KeyboardInterrupt:
        final_notice = "Monitor closed by user; training was not signalled"
    console.print(render(shown or {},final_notice))
    console.print(final_notice)


if __name__ == "__main__":
    main()
