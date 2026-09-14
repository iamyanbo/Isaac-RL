# Completion-reward training intervention — September 14, 2026

## Decision and current state

Implement **one mechanism change**: stop paying independently for enemy damage
and individual kills (`confirmed_v3` → `completion_v4`). Preserve combat-room
completion (+10), verified first-floor boss completion (+100), and every other
reward/control/learning setting. This is a categorical ablation of positive
within-combat proxy rewards, not a search for better coefficients.

**Implemented; NOT launched.** The existing six-worker learner20984 remains live
in `runs/ppo-control-v6`. Its Python imports keep the original reward profile.
No learner stop request, signal, game restart, or new evaluation was issued.
One checkpointed handover needs fresh user approval under the never-stop rule.
The user's latest instruction is a training change, not another evaluation;
completion of the already-running comparison is not a gate for this change.

Verification: **113 tests passed**, including five new tests covering the exact
reward delta, unchanged observations/endings/idle accounting, retained completion
rewards, native signal requirement, and both trainers' fork/resume paths. The
six-environment integration uses real environment/collector/PPO code with simulated
bridge replies, preserving initial weights/Adam/RNG exactly and then changing
finite weights through an actual optimizer update. It is not native gameplay
validation or a production training launch. PowerShell launcher parsing, CLI
profile exposure, frozen-baseline/manifest checks and `git diff --check` passed.

## Evidence and limits

The four-tick run's completed episode-log prefix through **3,420,864 steps** has
**1,387 episodes, zero wins, 32 boss encounters, 980 deaths, 407 idle endings**,
and731 combat clears. Prefix780830bytes, SHA256
`6acdc76b86db6e7f384ba7743cc2699a2fa09aeb2f322912d5dc307f051e2c5d`.
This is still insufficient floor competence despite the verified control fix.

The completed older learned-vs-random comparison (`runs/baseline-20260914`)
already showed more paid damage without a verified boss win for either arm.
Re-reading its stochastic transition records this turn found **327 sampled
health-loss transitions; 47 had positive immediate reward**. Removing exactly
damage and kill payments makes43 of those47 nonpositive. The full stochastic
trace paid430.15 for damage/kills versus -178.5 in hurt costs. These totals are
not a calculation of an optimal policy or a causal estimate of the intervention.
Transitions SHA256:
`d7d35f9e031f7a57e13a0420243832c5025d15cdead13079f67688bff0f07cbd`.

Concrete recorded example: seed3V2TZACA, repetition0, step22: 3.5 enemy damage,
one hurt unit, no kill/clear/pickup; reward+0.19 at eight ticks. The same event
combination at the current four-tick timing pays+0.195 under confirmed_v3 but
**-0.505 under completion_v4**. Tests reproduce this through IsaacEnv, with
identical observations and endings. Native HP-delta attribution is not limited
to damage caused by the player; eliminating that payment also removes its direct
contribution to the learning target without concealing its telemetry.

Hypothesis: positive intermediate combat payments let the learner earn useful
return without preserving enough health to finish combat and the floor. Removing
them makes room completion the first positive combat milestone. This does not
prove that the current behavior is globally reward-optimal or that this is the
sole bottleneck. Navigation idling, missing observations and memory may remain.

Why this before a recurrent/structured encoder: the payment mismatch is visible
in actual transitions; velocity and continuous entity positions already exist,
and the extent to which missing history limits current behavior is less directly
established. No claim is made that feed-forward observations are sufficient.
Why not remove all combat completion reward: there are still zero boss wins;
retaining a reached room-completion milestone avoids simultaneously eliminating
the only frequently observed combat outcome signal. Remaining shaping is **not**
all potential-based and does not guarantee invariance of the boss-win objective.

## Preserved baseline and constants

`runs/completion-intervention-20260914/baseline.pt` is a byte-for-byte copy of a
saved live-parent checkpoint: **3,419,340 steps / 9,478 total historical episodes /
4,116 updates / 3,795,560 controlled native ticks**. Native tick origin remains
2,470,092 steps. SHA256:
`e1498fbcade44ff8f562220735ff4a94bc933534bd9fed6c98ad3a120c76e0b9`.
All model tensors are finite. Model, Adam state, RNG state, seen training seeds,
and original metadata are retained. Copied config and observed status are adjacent;
status was copied separately and is not asserted to be simultaneous with the checkpoint.

Initialize the treatment from **this frozen baseline**, not silently newer weights.
If a handover is approved, archive the old learner's final checkpoint separately.
The learner's intervening training is neither deleted nor merged into the fork.

Constants: six instances0/1/2/4/5/6 on9999/10000/10001/10003/10004/10005; CPU/two
Torch threads; four ticks/physical_v1; gamma0.9974968671630001;
GAE lambda0.9746794344808963; rollout256/env; minibatch256; epochs4; lr3e-4;
Adam eps1e-5; clip0.2; target KL0.025; gradient clip0.5; entropy0.002;
max6750/idle900 decisions; seed20260912; architecture1; terrain_v2; action heads
9/5/4; native bridge/physics unchanged. Damage/kills still reset the idle timer,
are still logged, and still participate in identifying completed combat rooms.
No critic reset, Adam reset, reward normalization, or mechanical time conversion.
Only the reward/loss reporting windows clear on profile fork, as existing code
already requires for incomparable reward units. Old value estimates will initially
reflect the old target; that transient is acknowledged, not tuned away this turn.

## Launch after one-time checkpointed handover approval

Verify exact current learner/game identities before any native action. Cooperatively
save/exit the authorized learner, archive its final state, verify its exit and
port release, then reuse the same six games. Do not create six additional games
or steal sockets. Preserve the independent evaluator on10002.

```powershell
.\scripts\launch_parallel.ps1 -Run runs/ppo-completion-v7 `
  -Instances 0,1,2,4,5,6 -Device cpu -Steps 0 `
  -Initialize runs/completion-intervention-20260914/baseline.pt `
  -RewardProfile completion_v4 -DashboardPort 8768
```

The launcher makes an exact `parent.pt` copy. Omitted timing, schedule, entropy,
and observation options inherit the checkpoint. Its existing occupied-port guard
prevents launching over the live learner; do not bypass it. Do not run this command
before handover approval and port-release verification. Current charts stay8767.

Verify the first real six-worker PPO update: identical parent digest; correct
profile in config/checkpoint/status; zero damage/kill reward components with
unchanged raw counters; finite changed weights; all six frame deltas4; original
Adam parameter groups/schedule; empty stderr; correct chart run. No success claim
follows from this implementation check.

Assess adaptation at +2.4M controlled native ticks (600k new decisions), not by
comparing shaped returns across profiles. Require evidence of improved combat
completion per simulated time, health retained, boss reach and ultimately native
verified unseen first-floor boss wins. Reduced damage totals or lower return alone
are neither success nor failure. A collapse in combat completion without improved
survival would count against the hypothesis. Do not queue another evaluation now
or make evaluation completion a launch prerequisite.
