"""Versioned control timing. Physical units are referenced to eight native ticks.

Legacy checkpoints retain decision-based semantics. physical_v1 changes temporal
resolution while preserving discount/GAE decay, time cost, visitation intensity,
episode budgets, and nominal optimizer-update frequency per simulated second.
"""
from dataclasses import dataclass
import math


NATIVE_HZ = 30
REFERENCE_FRAMES = 8
BASE_GAMMA = 0.995
BASE_LAMBDA = 0.95
TIMING_PROFILES = ("legacy_v1", "physical_v1")


@dataclass(frozen=True)
class ControlTiming:
    frames: int = REFERENCE_FRAMES
    profile: str = "legacy_v1"

    def __post_init__(self):
        if isinstance(self.frames, bool) or not isinstance(self.frames, int) or not 1 <= self.frames <= 30:
            raise ValueError("Action frames must be an integer between 1 and 30")
        if self.profile not in TIMING_PROFILES:
            raise ValueError(f"Unsupported timing profile: {self.profile}")

    @property
    def scale(self):
        return self.frames / REFERENCE_FRAMES if self.profile == "physical_v1" else 1.0

    @property
    def gamma(self):
        return BASE_GAMMA ** self.scale

    @property
    def lam(self):
        return BASE_LAMBDA ** self.scale

    def steps(self, reference_steps):
        return math.ceil(reference_steps / self.scale)

    def manifest(self):
        return dict(profile=self.profile, frames=self.frames, native_hz=NATIVE_HZ,
                    reference_frames=REFERENCE_FRAMES, decision_interval_ms=1000*self.frames/NATIVE_HZ,
                    gamma=self.gamma, gae_lambda=self.lam, time_cost_scale=self.scale,
                    visit_count_scale=self.scale)


def checkpoint_timing(checkpoint=None):
    saved = checkpoint or {}
    return ControlTiming(saved.get("frames", REFERENCE_FRAMES), saved.get("timing_profile", "legacy_v1"))


def configure_training(args, saved=None, existing=False, previous_config=None,
                       default_rollout=128, default_batch=128):
    """Resolve explicit CLI values, checkpoint inheritance, and time-unit defaults.

    Durations cannot change within an existing run. Schedule fields are stored in
    checkpoints so a plain resume cannot accidentally return to eight-tick limits.
    Older in-place resumes recover their custom schedule from config.json.
    """
    parent = checkpoint_timing(saved)
    frames = args.frames if args.frames is not None else parent.frames
    profile = getattr(args, "timing_profile", None) or parent.profile
    control = ControlTiming(frames, profile)
    changed = saved is not None and control != parent
    if existing and changed:
        raise ValueError("Changing action timing requires a new run directory; preserve the baseline")
    previous = previous_config or {}
    if existing and previous and checkpoint_timing(previous) != control:
        raise ValueError("Checkpoint timing differs from the existing run configuration")
    source = (saved or {}).get("training_schedule", previous)
    defaults = dict(rollout=default_rollout, batch_size=default_batch,
                    max_episode_steps=3375, idle_limit=450)
    schedule = {}
    for key, reference in defaults.items():
        requested = getattr(args, key, None)
        if requested is None:
            if key in source:
                factor = parent.frames/frames if changed and profile == "physical_v1" else 1.0
                requested = math.ceil(source[key]*factor)
            else:
                requested = control.steps(reference)
        if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
            raise ValueError(f"{key} must be a positive integer")
        schedule[key] = requested
        setattr(args, key, requested)
    args.frames, args.timing_profile = frames, profile
    return control, schedule, changed


def configure_evaluation(args, saved):
    control = checkpoint_timing(saved)
    for name, reference in (("max_episode_steps", 3375), ("idle_limit", 900)):
        value = getattr(args, name, None)
        value = control.steps(reference) if value is None else value
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        setattr(args, name, value)
    return control
