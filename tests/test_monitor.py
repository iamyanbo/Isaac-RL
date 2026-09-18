import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
import pytest
from rich.console import Console

from isaac_rl import lifecycle as life
from isaac_rl.monitor import JsonlWindow, display, spark
from isaac_rl.runtime import RunLease, measure, commit_memory


@pytest.mark.parametrize("steps,width,height",[(999999,80,24),(1000000,80,24),(1000002,80,24),
    (12000000,64,18),(1234567890,120,30),(1234567890,40,16)])
def test_large_step_display_and_history_fit_terminal(steps,width,height,tmp_path):
    state = dict(pid=os.getpid(),process_started=psutil.Process().create_time(),time=time.time(),
        status="training",steps=steps,episodes=3210,updates=1440,num_envs=8,
        loss=.15,mean_reward=-1.25,workers=[dict(port=9999+i,health=6,episode_reward=3.2) for i in range(8)])
    stream = io.StringIO()
    console = Console(file=stream,width=width,height=height,force_terminal=False,color_system=None)
    console.print(display(tmp_path/"status.json",state,[dict(r=i) for i in range(50)],
        [dict(loss=i*.1,rollout_sps=20+i) for i in range(50)],width,height))
    result = stream.getvalue()
    assert f"{steps:,}" in result
    assert "ALIVE" in result
    assert "Rolling window" in result
    assert "Reward" in result and "Loss" in result
    assert len(result.splitlines()) <= height
    assert all(len(line) <= width for line in result.splitlines())


def test_history_loads_bounded_tail_and_handles_partial_truncated_replaced_files(tmp_path):
    path = tmp_path/"episodes.jsonl"
    path.write_text(''.join(json.dumps(dict(r=i))+"\n" for i in range(100)))
    tail = JsonlWindow(path,3)
    assert [r["r"] for r in tail.poll()] == [97,98,99]
    with path.open("a") as stream:
        stream.write('{"r": 100')
    assert [r["r"] for r in tail.poll()] == [97,98,99]
    with path.open("a") as stream:
        stream.write('}\n')
    assert [r["r"] for r in tail.poll()] == [98,99,100]
    path.write_text('{"r": -1}\n')
    assert tail.poll() == [dict(r=-1)]
    replacement = tmp_path/"new.jsonl"
    replacement.write_text('{"r": -2}\n')
    os.replace(replacement,path)
    assert tail.poll() == [dict(r=-2)]


def test_spark_is_bounded_and_works_for_flat_values():
    assert len(spark(list(range(100)),10)) == 10
    assert spark([1,1,1],10) == "==="


@pytest.mark.parametrize("status,age,alive,label",[
    ("training",1,"alive","ALIVE"),("connecting",1,"alive","IDLE"),
    ("training",60,"alive","STALE"),("error",1,"alive","EXITING"),
    ("training",1,"dead","DEAD"),("stopped",1,"dead","DEAD"),
    ("training",1,"unknown","UNKNOWN")])
def test_lifecycle_distinguishes_identity_from_heartbeat(status,age,alive,label):
    assert life.lifecycle(dict(status=status,time=100-age),now=100,liveness=alive)[0] == label


def test_watch_does_not_reconnect_to_next_run_or_reused_pid(monkeypatch):
    monkeypatch.setattr(life,"process_liveness",lambda key: "alive")
    watch = life.SessionWatch()
    old = dict(pid=123,process_started=100.,status="training")
    assert not watch.observe(old)[1]
    newer = dict(pid=123,process_started=200.,status="training")
    shown,done,message = watch.observe(newer)
    assert shown == old and done and "different session" in message
    follower = life.SessionWatch(follow=True)
    follower.observe(old)
    assert follower.observe(newer)[0] == newer


def test_launcher_bound_watch_ignores_previous_status_before_first_publish(monkeypatch):
    monkeypatch.setattr(life,"process_liveness",lambda key: "alive")
    watch = life.SessionWatch((456,200.))
    assert watch.observe(dict(pid=123,process_started=100.)) == (None,False,"")
    assert watch.observe(dict(pid=456,process_started=200.))[0]["pid"] == 456


def test_windows_millisecond_launch_time_matches_native_timestamp(monkeypatch):
    monkeypatch.setattr(life,"process_liveness",lambda key: "alive")
    watch = life.SessionWatch((51700,1789318456.317))
    state = dict(pid=51700,process_started=1789318456.3176687,status="training")
    shown,done,_ = watch.observe(state)
    assert shown == state and not done
    assert watch.key == (51700,1789318456.3176687)
    assert not watch.observe(state,now=watch.started+60)[1]
    newer = dict(state,process_started=1789318460.317)
    assert watch.observe(newer)[1]


def test_process_identity_is_creation_time_sensitive():
    process = psutil.Process()
    assert life.process_liveness((process.pid,process.create_time())) == "alive"
    assert life.process_liveness((process.pid,process.create_time()-10)) == "dead"


def test_monitor_cli_exits_after_learner_death_and_startup_timeout(tmp_path):
    child = subprocess.Popen([sys.executable,"-c","pass"])
    started = psutil.Process(child.pid).create_time()
    child.wait(timeout=5)
    (tmp_path/"status.json").write_text(json.dumps(dict(pid=child.pid,process_started=started,
        status="error",steps=1234567,episodes=2,time=time.time())))
    result = subprocess.run([sys.executable,"-m","isaac_rl.monitor",str(tmp_path)],capture_output=True,text=True,timeout=8)
    assert result.returncode == 0 and "DEAD" in result.stdout and "monitor exiting" in result.stdout
    empty = tmp_path/"empty"
    empty.mkdir()
    result = subprocess.run([sys.executable,"-m","isaac_rl.monitor",str(empty),"--startup-timeout","0.2"],
        capture_output=True,text=True,timeout=8)
    assert result.returncode == 0 and "startup timeout" in result.stdout


def test_run_lease_rejects_duplicate_owner_then_releases(tmp_path):
    with RunLease(tmp_path):
        with pytest.raises(RuntimeError,match="live owner"):
            with RunLease(tmp_path):
                pass
    with RunLease(tmp_path):
        pass


def test_phase_timing_accumulates_without_changing_work():
    timing = {}
    with measure(timing,"collection_s"):
        time.sleep(.002)
    first = timing["collection_s"]
    with measure(timing,"collection_s"):
        time.sleep(.002)
    assert timing["collection_s"] > first > 0
    if os.name == "nt":
        commit = commit_memory()
        assert 0 < commit["used_gib"] <= commit["limit_gib"]
