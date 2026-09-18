"""Run ownership and measured phase timings; no process stopping or retry loop."""
from contextlib import contextmanager
import ctypes
import os
from pathlib import Path
import time

import torch


class RunLease:
    """OS-released lock: one learner per run, even before any port is bound."""
    def __init__(self, run):
        self.path = Path(run)/"trainer.lock"
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.stream = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as error:
            self.stream.close()
            self.stream = None
            raise RuntimeError(f"Run already has a live owner: {self.path.parent}") from error
        return self

    def __exit__(self,*_):
        if self.stream:
            self.stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(),fcntl.LOCK_UN)
            self.stream.close()
            self.stream = None


@contextmanager
def measure(timing, key, device="cpu"):
    cuda = str(device).startswith("cuda")
    if cuda:
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        yield
    finally:
        if cuda:
            torch.cuda.synchronize(device)
        timing[key] = timing.get(key,0.) + time.perf_counter()-started


def commit_memory():
    """Windows commit capacity, which is NOT the same as free physical RAM."""
    if os.name != "nt":
        return None
    class PerformanceInfo(ctypes.Structure):
        _fields_ = [("cb",ctypes.c_ulong)] + [(name,ctypes.c_size_t) for name in
            ("CommitTotal","CommitLimit","CommitPeak","PhysicalTotal","PhysicalAvailable",
             "SystemCache","KernelTotal","KernelPaged","KernelNonpaged","PageSize")] + [
            (name,ctypes.c_ulong) for name in ("HandleCount","ProcessCount","ThreadCount")]
    info = PerformanceInfo()
    info.cb = ctypes.sizeof(info)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info),info.cb):
        raise ctypes.WinError()
    scale = info.PageSize/2**30
    return dict(used_gib=info.CommitTotal*scale,limit_gib=info.CommitLimit*scale,
        available_gib=(info.CommitLimit-info.CommitTotal)*scale,
        percent=100*info.CommitTotal/info.CommitLimit)
