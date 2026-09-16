# Native combat-clear qualification — September 16, 2026

## One authorized learning mechanism

Fork the live enriched-history learner's **final** checkpoint into
`runs/ppo-native-clear-v9`, with six workers and reward `native_clear_v5`.
User explicitly approved this one-time checkpointed handover. It is not an
automatic restart policy. The final checkpoint, not the preparation snapshot,
must be preserved byte-for-byte and used to initialize the fork.

Only change: a room qualifies for a combat-clear bonus after it has been observed
with native `room.clear == false` and `room.enemies > 0`. Previously, a positive
NPC count alone qualified a room, including already-cleared rooms. The existing
+10 completion bonus and all other coefficients remain unchanged. A genuine
native clear still qualifies with residual nonblocking NPCs; requiring zero NPCs
would discard real native clears. Existing death/survival semantics are unchanged.

The observation, four-snapshot/action history, architecture, model weights, Adam
state, RNG states, four-tick hold, physical-time conversions, entropy .002, PPO
settings, six ports, CPU/two threads, seed, rollout256/env, batch256,
episode6750/idle900 and uncapped training remain unchanged. No unit conversion or
hyperparameter tuning. New-run reward/loss telemetry windows are cleared because
returns across reward profiles are not comparable; optimizer state is not reset.
Games start fresh episodes at handover; their mid-episode state is not checkpointed.

## Evidence, not a claim of the sole bottleneck

The completed observability comparison has **0/100 native boss wins in each arm**.
Replaying all 132,029 native states exactly reconstructs every legacy and strict
counter. Fourteen disagreements split into seven false room-entry credits and
seven real native clear transitions excluded by the strict zero-NPC definition.
The false credits all involve Type218 wall-hugger NPCs in already-cleared rooms.
Do not describe all fourteen as false rewards or infer an unobserved NPC cause.

The prior strict living-clear endpoint is 22/100 parent vs33/100 treatment;
delta +11 percentage points, conservative95% interval [-6.51,+27.69] points.
A post-hoc native Room.IsClear endpoint allowing residual NPCs is24/100 vs37/100;
delta +13 points, interval [-5.05,+30.09]. Neither establishes boss competence.
The frozen treatment was5,797,932 steps,583,296 later than the planned review.
The original fixed-budget verdict is unassessable. Preserve the original
comparison/result.json with native_counters_valid=false and comparison=null;
the independent audit does not retrospectively turn that guard into a pass.

Artifacts: `runs/observability-review-20260916/native-counter-audit.json`,
`audited-outcomes.json`, and raw comparison traces. The counterfactual replay in
`runs/native-clear-intervention-20260916/reward-replay.json` checks all131,829
actions in200 episodes through the real reward code: **exactly seven +10 payments
removed**, all other components, idle counters and episode endings identical.
Trace SHA256:7f27d737767b42dd7d451803b73f9babb5e0c8a27f8ff8e3d6500d2a98686f66.
This is a demonstrated objective-label error, but seven events do not prove it
explains the whole plateau. Correcting it precedes speculative further changes.

## Fixed-budget test and explicit falsification

Retain the first optimized checkpoint at or above **final parent +1,200,000
aggregate decisions**. Full rollouts may overshoot by at most1,535 decisions;
record actual exposure, never substitute a later favorable snapshot. Opt-in
`snapshot_steps.json` retention is an operational artifact safeguard, not another
learning mechanism; it does not stop or evaluate training. Archive failure is
reported in telemetry without crashing the learner.

Compare the frozen activation parent and retained treatment on100 new matched
held-out native full-floor seeds selected before scoring, excluding both models'
training seeds and all earlier evaluation seeds. Counterbalance policy order,
reset the same per-seed action RNG for each arm, and inherit the exact hold and
episode/idle limits. Primary outcome: independently verified living first-floor
boss completion. Secondary: observed uncleared-to-native-clear room transitions
while alive. Preserve raw native evidence and audit counters; shaped return,
recategorized clear counts and movement cannot establish learned improvement.

**Falsification:** reject the operational claim that this correction alone yields
at least a10-percentage-point boss-win improvement at this budget if the upper
conservative95% bound on treatment-minus-parent boss-win probability is below
+0.10. Invalid native evidence, missing target weights, or a bound still spanning
+0.10 makes this test inconclusive, not a success. A positive parent comparison
cannot isolate reward correction from additional training; a matched unchanged-
reward training control would be needed for strong causal attribution. No new
evaluation is launched as part of this handover.

## Verification and activation record

175 tests pass, including real single/six-env PPO fork/resume with exact initial
model/Adam/RNG inheritance, unchanged observation inputs and milestone retention.
The full native reward replay passed. `scripts/verify_reward_handover.py` archives
and verifies actual initial/optimized checkpoints without controlling any game.
Runtime metadata, exact final baseline hash, process identities and first optimizer
progress are recorded in `runs/native-clear-intervention-20260916` and the current
OPERATIONS.md section. See the activation addendum after the switch completes.

## Activation completed — September 16, 16:01 Toronto

Parent learner30608 exited0/cleanup complete at **6,667,974 steps,6,232 updates,
16,784,992 counted native ticks**. Final SHA256
**a1295f92033ebd3bff8a8106e81482f5afda12e93388907dac31e17c80f20621**;
old latest.pt, intervention parent-final.pt and new parent.pt are byte-identical.
No rollback to the preparation baseline6,636,588 or any earlier model.

New learner **37144 /1789588746.8131385**, session-1789588750943756600,
reuses all six native game identities. Exact initial model/Adam/RNG inheritance
verified in initial-verification.json. All recorded configuration constants
match; there is no observation, optimization, timing, curriculum or worker change.
Initial checkpoint SHA256
**d16d13b0b8f72d86ce1e96a0afe8915bfbefe57b2c8fc74da78a168b86dcddd8**.

First optimized checkpoint **6,669,510 steps /6,233 updates /16,791,129 ticks**,
SHA256 **da2fe7f78b5ab3f4d2ce31862e6839db8a268a28b7cb8df179710feb21f2103e**,
has finite changes in all16 tensors and unchanged optimizer groups/settings.
Collection38.8427s, PPO1.1237s; second update6,671,046 also completed.
Six established learner-owned sockets and empty stderr confirmed. These verify
actual learning execution, not competence. No evaluation was launched.

Actual retention target **7,867,974**, expected next full-rollout save7,869,126
(+1,201,152 decisions). `snapshot_steps.json` and experiment manifests written
before launch. Charts http://127.0.0.1:8770/; old8769 displays DEAD history.
Implementation commit **f15388a**. Approval consumed; no automatic restart/stop.
