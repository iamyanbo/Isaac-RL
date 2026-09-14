"""Inspect or send menu keys only to the private training game's window."""
import argparse
import ctypes
from ctypes import wintypes
from pathlib import Path
import time

import psutil
from PIL import Image

ctypes.windll.user32.SetProcessDPIAware()


def runtime_path(instance=0):
    runtime = Path(__file__).resolve().parents[1] / "runtime"
    return runtime if instance == 0 else runtime / f"worker-{instance:02d}"


def window(instance=0):
    game = (runtime_path(instance) / "game" / "isaac-ng.exe").resolve()
    pids = []
    for proc in psutil.process_iter(["pid", "name", "exe"]):
        if proc.info["exe"] and Path(proc.info["exe"]).resolve() == game:
            pids.append(proc.pid)
    windows = []
    user = ctypes.windll.user32
    callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def collect(hwnd, _):
        pid = wintypes.DWORD()
        user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value in pids:
            length = user.GetWindowTextLengthW(hwnd)
            if length > 0:
                windows.append(hwnd)
        return True
    user.EnumWindows(callback(collect), 0)
    if not windows:
        raise RuntimeError(f"No training game window (processes {pids})")
    return windows[0]


def capture(hwnd, output):
    user, gdi = ctypes.windll.user32, ctypes.windll.gdi32
    user.GetWindowDC.restype = wintypes.HDC
    user.GetWindowDC.argtypes = [wintypes.HWND]
    user.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
    user.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    gdi.CreateCompatibleDC.restype = wintypes.HDC
    gdi.CreateCompatibleDC.argtypes = [wintypes.HDC]
    gdi.CreateCompatibleBitmap.restype = wintypes.HBITMAP
    gdi.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
    gdi.SelectObject.restype = wintypes.HANDLE
    gdi.SelectObject.argtypes = [wintypes.HDC, wintypes.HANDLE]
    gdi.GetBitmapBits.argtypes = [wintypes.HBITMAP, ctypes.c_long, ctypes.c_void_p]
    gdi.DeleteObject.argtypes = [wintypes.HANDLE]
    gdi.DeleteDC.argtypes = [wintypes.HDC]
    rect = wintypes.RECT()
    user.GetWindowRect(hwnd, ctypes.byref(rect))
    width, height = rect.right-rect.left, rect.bottom-rect.top
    dc = user.GetWindowDC(hwnd)
    memory = gdi.CreateCompatibleDC(dc)
    bitmap = gdi.CreateCompatibleBitmap(dc, width, height)
    old = gdi.SelectObject(memory, bitmap)
    user.PrintWindow(hwnd, memory, 2)
    buffer = ctypes.create_string_buffer(width * height * 4)
    gdi.GetBitmapBits(bitmap, len(buffer), buffer)
    Image.frombuffer("RGB", (width, height), buffer, "raw", "BGRX", 0, 1).save(output)
    gdi.SelectObject(memory, old)
    gdi.DeleteObject(bitmap)
    gdi.DeleteDC(memory)
    user.ReleaseDC(hwnd, dc)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--key", choices=["enter", "escape", "space", "up", "down", "left", "right"])
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--hide", action="store_true")
    parser.add_argument("--instance",type=int,default=0)
    parser.add_argument("--start", action="store_true",help="Start through normal menus, stopping once a run is observed in the game log")
    args = parser.parse_args()
    if args.start:
        root = Path(__file__).resolve().parents[1]
        import json
        installation = json.loads((runtime_path(args.instance)/"installation.json").read_text())
        logfile = Path(installation["saves"])/"log.txt"
        private_exe = (runtime_path(args.instance)/"game"/"isaac-ng.exe").resolve()
        game_started = max(proc.create_time() for proc in psutil.process_iter(["exe"])
            if proc.info["exe"] and Path(proc.info["exe"]).resolve() == private_exe)
        for attempt in range(40):
            try:
                hwnd = window(args.instance)
            except RuntimeError:
                time.sleep(1)
                continue
            ctypes.windll.user32.ShowWindowAsync(hwnd,4)
            if logfile.exists() and logfile.stat().st_mtime >= game_started and "SpawnRNG seed:" in logfile.read_text(errors="replace"):
                print("Normal run started")
                raise SystemExit(0)
            if attempt >= 5:
                ctypes.windll.user32.PostMessageW(hwnd,0x100,13,0)
                time.sleep(0.15)
                ctypes.windll.user32.PostMessageW(hwnd,0x101,13,0)
            time.sleep(1)
        raise RuntimeError("Normal menu startup timed out; inspect the private game window")
    hwnd = window(args.instance)
    user = ctypes.windll.user32
    if args.show:
        user.ShowWindowAsync(hwnd, 4)
    if args.hide:
        user.ShowWindowAsync(hwnd, 0)
    if args.key:
        key = dict(enter=13, escape=27, space=32, left=37, up=38, right=39, down=40)[args.key]
        user.PostMessageW(hwnd, 0x100, key, 0)
        time.sleep(0.15)
        user.PostMessageW(hwnd, 0x101, key, 0)
    if args.capture:
        capture(hwnd, args.capture)
