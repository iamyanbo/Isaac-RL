"""Process identity and phase are separate: stale telemetry is not liveness."""
import time

import psutil

TERMINAL = {"stopped", "error", "finished"}


def identity(state):
    if not state or not state.get("pid") or not state.get("process_started"):
        return None
    return int(state["pid"]), float(state["process_started"])


def same_identity(first, second):
    """Windows launch metadata has millisecond rather than submillisecond time."""
    return first is not None and second is not None and first[0] == second[0] and abs(first[1]-second[1]) < .01


def process_liveness(key):
    if key is None:
        return "unknown"
    try:
        process = psutil.Process(key[0])
        if abs(process.create_time()-key[1]) > .01 or not process.is_running():
            return "dead"
        return "dead" if process.status() == psutil.STATUS_ZOMBIE else "alive"
    except psutil.NoSuchProcess:
        return "dead"
    except psutil.AccessDenied:
        return "unknown"


def lifecycle(state, now=None, stale_after=15, liveness=None):
    now = time.time() if now is None else now
    live = process_liveness(identity(state)) if liveness is None else liveness
    phase = state.get("status", "waiting") if state else "waiting"
    age = max(0., now-state.get("time", now)) if state else 0.
    if live == "dead":
        return "DEAD", phase, age
    if live == "unknown":
        return "UNKNOWN", phase, age
    if phase in TERMINAL:
        return "EXITING", phase, age
    if phase in {"connecting", "idle", "waiting"}:
        return "IDLE", phase, age
    if age > stale_after:
        return "STALE", phase, age
    return "ALIVE", phase, age


class SessionWatch:
    """Bind once to PID AND creation time; never silently follow a replacement."""
    def __init__(self, key=None, follow=False, startup_timeout=30):
        self.key, self.follow = key, follow
        self.started = time.monotonic()
        self.startup_timeout = startup_timeout
        self.last_state = None

    def observe(self, state, now=None):
        candidate = identity(state)
        if self.key is None and candidate:
            self.key = candidate
        if candidate and not same_identity(candidate,self.key):
            if self.follow:
                self.key = candidate
            elif process_liveness(self.key) == "dead":
                return self.last_state, True, "Session ended; replacement learner not followed"
            elif self.last_state is not None:
                return self.last_state, True, "Telemetry replaced by a different session"
            else:
                state = None  # New launcher PID has not published its first state.
        if state and same_identity(identity(state),self.key):
            self.key = candidate  # Canonicalize the full-precision native identity.
            self.last_state = state
        live = process_liveness(self.key)
        if live == "dead" and not self.follow:
            return self.last_state, True, "Learner exited; monitor exiting (games may remain idle)"
        elapsed = (time.monotonic() if now is None else now)-self.started
        if not self.last_state and elapsed >= self.startup_timeout and not self.follow:
            return None, True, "No telemetry within startup timeout; monitor exiting"
        return self.last_state, False, ""
