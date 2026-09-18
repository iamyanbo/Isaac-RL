# Completion-reward training intervention — September 14, 2026

## Decision and current state

Implement **one mechanism change**: stop paying independently for enemy damage
and individual kills (`confirmed_v3` → `completion_v4`). Preserve combat-room
completion (+10), verified first-floor boss completion (+100), and every other
reward/control/learning setting. This is a categorical ablation of positive
within-combat proxy rewards, not a search for better coefficients.

**LIVE as of September 15.** The user's "make the changes" authorized one
checkpointed handover. Learner46824 (creation1789476850.9710472) runs
`runs/ppo-completion-v7` on the same six games. Old learner20984 saved/exited
cleanly at5,377,722; its final checkpoint is separately archived. The new run
starts from the declared frozen3,419,340 baseline, whose bytes match `parent.pt`.
This one-time approval is consumed; never stop the new learner without further
approval. No new evaluation was launched. Current charts: http://127.0.0.1:8768/.

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

## Executed handover and launch

Exact learner/game identities and baseline hash were verified before the cooperative
stop. The old learner saved/exited normally and all six ports were released before
launch. Final checkpoint/status/states and stop marker were archived. No extra
games were created. Worker0 reached Game Over during the gap; one guarded normal
Space restart restored its connection after screenshot/native-log verification.
The other five games needed no input. Reserved instance3 remained untouched.

```powershell
.\scripts\launch_parallel.ps1 -Run runs/ppo-completion-v7 `
  -Instances 0,1,2,4,5,6 -Device cpu -Steps 0 `
  -Initialize runs/completion-intervention-20260914/baseline.pt `
  -RewardProfile completion_v4 -DashboardPort 8768
```

The launcher made an exact `parent.pt` copy. Omitted timing, schedule, entropy,
and observation options inherited the checkpoint. This command has already run;
do not repeat it against the live learner. The existing occupied-port/run guards
must not be bypassed. Old charts8767 remain baseline history.

Verified first real update4117 at3,420,876:1,536 transitions,6,140 native ticks,
collection38.241356s/PPO0.406807s,37.3187 decisions/s. The archived checkpoint
`first-observed-treatment.pt` has all16 model tensors finite and changed; original
Adam parameter groups and schedule retained. Parent digest, profiles, source hashes,
six established connections/four-tick deltas, zero proxy reward terms, retained
raw damage counters, empty stderr and chart run identity verified. The observed
checkpoint was already post-update, not a pre-update model/RNG snapshot.

Archived final parent SHA256:
`31740cdf0a3a99fbdf8415dcbaaf62b2240e9483233ab2c8cc54899420661f49`.
Archived first treatment update SHA256:
`1121ec697dedd94063a3ab52cfda126ee5c916160e570b8efe2c29e7074f192a`.
Both are under `runs/completion-intervention-20260914`. The original baseline is
unchanged. See OPERATIONS.md and the local experiment manifest for identities and
handoff evidence. No competence improvement is claimed from a successful launch.

Assess adaptation at +2.4M controlled native ticks (600k new decisions), not by
comparing shaped returns across profiles. Require evidence of improved combat
completion per simulated time, health retained, boss reach and ultimately native
verified unseen first-floor boss wins. Reduced damage totals or lower return alone
are neither success nor failure. A collapse in combat completion without improved
survival would count against the hypothesis. Do not queue another evaluation now
or make evaluation completion a launch prerequisite.
