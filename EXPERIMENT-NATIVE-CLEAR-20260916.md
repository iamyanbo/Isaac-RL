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
