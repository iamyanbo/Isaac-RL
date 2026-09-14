# Four-tick control intervention

The user authorized implementation and the next run after native verification of
the eight-tick control hold, and clarified that **hyperparameter tuning in general
must not be the default first intervention**. The rationale is control feedback,
not an entropy/return hill climb.

## Verified mechanism

The gameplay counter runs at 30 Hz. In the completed native diagnostic,
27,206/27,244 actions advanced eight ticks; the other 38 ended early on death.
Median observation cadence was 266.95 ms. Across 1,946 persistent, constant-velocity
projectile transitions, displacement matched eight times native velocity to less
than 0.001 world units. Median/p95 projectile speeds were 7/10 units per tick,
giving 56/80 units of travel between decisions. Source and all six runtime bridges
matched. The bridge freezes simulation while waiting for Python, so inference/PPO
wall time is not additional simulated reaction latency.

This limits feedback, but does not prove every hit unavoidable or that timing is
the sole bottleneck. Continuous entity positions/velocities already exist in the
observation. Native hit-source and per-tick first-visibility attribution remain
missing. Full-floor boss wins remain the goal.

## Intervention and explicit time-unit conversions

Fork the CURRENT lower-entropy checkpoint, not the earlier 2.417M ancestor.
Target `runs/ppo-control-v6`; archive `runs/control-intervention-20260914`.
Do not overwrite the older baseline or entropy experiment.

| Quantity | Prior | New | Preserved meaning |
| --- | ---: | ---: | --- |
| Native ticks/action | 8 | 4 | Intervention: 267 ms to 133 ms feedback |
| Gamma | 0.995 | 0.997496867163 | Same discount decay per native tick |
| GAE lambda | 0.95 | 0.974679434481 | Same trace decay per native tick |
| Time cost/full action | -0.01 | -0.005 | Same cost per simulated second |
| Episode action limit | 3375 | 6750 | 900 simulated seconds |
| Idle action limit | 450 | 900 | 120 simulated seconds |
| Rollout actions/worker | 128 | 256 | 1024 native ticks, about 34.13 seconds |
| PPO minibatch transitions | 128 | 256 | Six minibatches/epoch with six workers |
| Visit-map count multiplier | 1 | 0.5 | Same saturation rate per dwelling time |

Time cost uses actual elapsed ticks when death ends an action early. Novelty still
pays once per first sampled 40-unit cell. Finer sampling can observe cells/events
that an eight-tick observation skips; it is not exact trajectory equivalence.
The larger minibatch preserves nominal SGD steps per simulated time, not identical
gradient noise or early-KL-stop behavior. No claim of identical optimization is made.

Unchanged: six native instances 0/1/2/4/5/6, CPU/two Torch threads, native physics,
model architecture and weights initialization from parent, action heads 9/5/4,
terrain_v2 entity features, confirmed_v3 event-reward coefficients, entropy0.002,
Adam3e-4/eps1e-5, four epochs, clip0.2, targetKL0.025, gradient clip0.5, seed20260912,
and uncapped training. No recurrence, shooting expansion, or combat-reward removal.

## Reproducibility and evaluation

`physical_v1` versions timing semantics. Old checkpoints retain legacy semantics.
Checkpoints/config/status/update logs record timing and schedule; an ordinary
resume inherits them. Changing timing within an existing run is rejected before
opening game ports. Forking clears the old recent reward/loss telemetry window.
Evaluation inherits timing and preserves its existing 900-second episode and
240-second diagnostic idle budgets. The already-running baseline evaluator is
unchanged and must not be interrupted.

Native tick counters begin at the preserved parent's aggregate step count, with
`native_frames_origin_steps` documenting where exact accounting started. They
exclude operational resets; historical native ticks are not fabricated. The
aggregate steps axis mixes pre-fork eight-tick and post-fork four-tick decisions.
Review at +2,400,000 controlled native ticks, equivalent to 300k old eight-tick
actions (roughly 600k new decisions), then +4,800,000 ticks. Report boss wins/reach,
combat clears and health retention on held-out seeds, not just shaped returns.

## Handover status

Completed after **108 passing tests**, both PowerShell launcher parse checks and
Git diff checks. Implementation commit **f17378b**. The old entropy learner21432
saved/exited normally at **2,470,092 steps / 8,094 episodes / 3,498 updates**,
exit0/cleanup complete. Its final checkpoint is the baseline, SHA256
**2e4830a744508c4a98ba157e4806eedb7d5000c99e026cc6115f6fc84fd33b50**;
new `parent.pt` matches byte for byte. Parent final state/config/status, pre-handover
checkpoint and authorization marker remain in the archive. No files deleted.

New learner **20984**, creation1789400666.5494964, session
`session-1789400670817434800`, launched11:44 Toronto on the same six games without
restarting any native process. All training sockets belong to it and all six
workers report four-tick deltas. The old stop.request remains in the old run as a
guard. The one-time handover authorization is consumed; preserve the new learner.

First full update3499 at **2,471,628**: 1,536 transitions,6,141 actual native ticks,
collection37.391s/PPO0.660s/wall40.199s,38.21 aggregate decisions/s. Early death
accounts for the three-tick shortfall from nominal6,144. Commit96.67% with3.21GiB
available at that update. Training remains collection-dominated. These timings are
execution verification, not evidence of learning success.

The immutable `first-observed-control.pt` is update3500 at **2,473,164**, SHA256
**6dd8cc9a0e247f67add3cc370b46f6cc91682f4b9ff75dc1f3e69375243c4ce4**.
It contains12,282 controlled ticks; all16 parameter tensors changed and are finite,
Adam parameter groups match the parent, and all checkpoint/config control settings
match the declared conversions. Source fingerprints and runtime bridge hashes
verified. Empty stderr at verification. Current charts
**http://127.0.0.1:8767/**, dashboard52820, HTTP verified correct run/learner.
Older dashboards8766/8765 retain baseline history. Independent evaluator6968 was
left running unchanged on10002.

The launch used the equivalent direct trainer invocation below after verifying all
six games were already alive and their ports released. The standard launcher also
supports these arguments and reuses existing games:

```powershell
.\scripts\launch_parallel.ps1 -Run runs/ppo-control-v6 -Instances 0,1,2,4,5,6 `
  -Steps 0 -Device cpu -Initialize runs/control-intervention-20260914/baseline.pt `
  -Frames 4 -TimingProfile physical_v1 -DashboardPort 8767
```
