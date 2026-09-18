"""Prepare and start one separately saved normal-speed game worker."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil

from prepare_runtime import prepare, ROOT


def launch(instance, source):
    if not 0 <= instance <= 16:
        raise ValueError("Game instance IDs must be 0..16")
    runtime = ROOT/"runtime" if instance == 0 else ROOT/"runtime"/f"worker-{instance:02d}"
    exe = (runtime/"game"/"isaac-ng.exe").resolve()
    for proc in psutil.process_iter(["exe"]):
        if proc.info["exe"] and Path(proc.info["exe"]).resolve() == exe:
            print(f"Instance {instance} already running as PID {proc.pid}")
            return
    manifest = prepare(source,instance)
    env = dict(os.environ,ISAAC_RL_PORT=str(manifest["port"]))
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    process = subprocess.Popen([str(exe),"--luadebug"],cwd=exe.parent,env=env,
        startupinfo=startup,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    (runtime/"process.json").write_text(json.dumps({"pid":process.pid,"port":manifest["port"],
        "process_started":psutil.Process(process.pid).create_time(),"exe":str(exe)},indent=2))
    print(f"Worker {instance}: game PID {process.pid}, localhost:{manifest['port']}",flush=True)
    subprocess.run([sys.executable,str(ROOT/"scripts"/"game_window.py"),"--instance",str(instance),"--start"],check=True)
    if instance != 0:
        subprocess.run([sys.executable,str(ROOT/"scripts"/"game_window.py"),"--instance",str(instance),"--hide"],check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("instance",type=int)
    parser.add_argument("--source",type=Path,default=Path(r"C:\Program Files (x86)\Steam\steamapps\common\The Binding of Isaac Rebirth"))
    args = parser.parse_args()
    launch(args.instance,args.source)
