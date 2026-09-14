# First four-tick intervention review — 2026-09-14, 18:00 Toronto

## Outcome and decision

The learner is alive, but first-floor competence remains unproven. At the recorded
training-log prefix: **1,233 completed treatment episodes, zero wins, 31 boss
encounters**. The faster control loop works; it has not established improved
dodging, aiming, navigation, or boss completion. No production source/settings,
game processes, or learner signals were changed during this review.

The previous goal turn was PROGRESS: implemented and launched the four-tick
intervention. This review also makes PROGRESS: fully audited the now-completed
older random-control evaluation and launched a frozen current-checkpoint evaluation
on the available reserved worker. Neither job completion nor the goal is inferred
from telemetry files alone; exact process identities were checked.

Keep the six-worker treatment running. Do not tune hyperparameters by default or
interpret a lower entropy/higher return as competence. Use the new held-out native
traces to distinguish objective/proxy optimization, navigation failure, and actual
combat behavior before recommending another mechanism intervention. No new
training handover is authorized by this review.

## Training evidence, normalized by simulated time

The preserved parent is2,470,092 steps. The new evaluation snapshot is3,327,180:
**857,088 new decisions / 3,427,089 controlled native ticks**, approximately428k
old eight-tick actions of exposure. The first review threshold (+2.4M native ticks)
has been crossed. Aggregate steps alone mix two different action durations.

| Completed training episodes | Entropy parent run | Four-tick treatment |
| --- | ---: | ---: |
| Episodes in recorded comparison | 199 | 1,233 |
| Wins | 0 | 0 |
| Boss encounters | 4 | 31 |
| Combat clears/episode | 0.437 | 0.545 |
| Mean simulated episode seconds | ~68.93 | 92.22 |
| Combat clears/simulated minute | ~0.381 | 0.355 |
| Idle endings | 17.09% | 30.17% |
| Mean shaped return | 4.263 | 7.673 |

These are descriptive, non-paired training samples with different seeds and
training ages, not a causal estimate of the timing intervention. Parent duration
uses action_count*8 and slightly overcounts early-death holds; treatment duration
uses exact native ticks. Longer survival or more clear-room wandering can increase
per-episode clears and reduce per-episode hurt while not improving useful progress.
Neither this table nor its per-minute denominator isolates combat exposure.

Last100 recorded PPO updates: entropy2.524, explained variance0.790, KL0.01144;
collection37.116s, PPO0.528s, wall40.475s,38.23 decisions/s mean. Optimization remains
a small fraction of wall time. The six workers still report four-tick advancement;
all production source hashes match the launch configuration. No new code tests
were needed for this read-only review; the prior implementation passed108 tests.

## Completed older held-out comparison

`runs/baseline-20260914` naturally finished120/120 and evaluator6968 exited with
code0. Its separate cleanup record confirms an unscored fresh-room reset. The
frozen checkpoint was **2,372,664**, hash
`17a91da77688e2e74aedc1feea058c77b10ec94ac127d5a627c08fc49a0898bf`.
This is an OLDER eight-tick checkpoint, not the four-tick treatment result.

Twenty unseen seeds, three repetitions each, learned stochastic versus uniform
random, same native action space and eight-tick hold. All120 episode behavior
summaries were reproduced exactly from39,288 recorded transitions. Report:
`runs/baseline-20260914/analysis-20260914-1800.json`.

| Metric | Learned | Uniform random |
| --- | ---: | ---: |
| Episodes | 60 | 60 |
| Wins / boss encounters | 0 / 0 | 0 / 0 |
| Deaths / idle endings | 55 / 5 | 55 / 5 |
| Combat clears/episode | 0.25 | 0.10 |
| Enemy HP damage/episode | 32.429 | 16.033 |
| Kills/episode | 2.000 | 1.067 |
| Mean shaped return | 1.416 | -6.429 |
| Episodes entering combat | 60 | 52 |

For each seed, first average its three repetitions within each arm, then subtract
random from learned. A descriptive seed-level percentile bootstrap (10,000 draws,
Python Random(20260914), sorted seed order, indices250/9749 of sorted bootstrap
means) gives combat-clear difference **+0.15, interval[-0.033,+0.350]**. Shaped-return
difference is+7.845[4.433,11.312]; enemy-damage difference+16.396[7.775,28.537].
Twenty seeds, not120 episodes, are the independent comparison units. Intervals are
exploratory, not multiplicity-corrected claims or proof of equality when including0.

The learned policy increases paid combat proxies, but this sample does not establish
a reliable combat-clear advantage and contains no boss reach at all. More combat
entry and firing can also increase damage without superior aiming. Actual sampled
HP loss was354 units learned versus340 random; the damage-taken callback counts
357 and341 respectively. No source/tick attribution proves which hits were avoided
or avoidable. The full-floor success gate has NOT passed.

## New current-checkpoint evaluation LIVE

Output **`runs/control-eval-20260914-1800`**, evaluator **PID10544**, creation
**1789423443.6716738**, reserved port **10002**, existing native game **47392**,
creation1789356107.3601575. Old evaluator identity confirmed dead and10002 unowned
before attaching. No new game, game restart, training interruption, or menu input.
Preflight commit95.76%,4.10GiB available; about150GiB free disk.

Frozen **3,327,180** checkpoint hash
**`7a1b5cdd49ed2abdcbbc0b3ccc2a2bf18d9bb58901a72284ed8b60bd5357e74f`**.
Preserved inside evaluation output; hash/finite weights/timing/entropy verified.
Twenty unseen seeds x three repetitions x stochastic/uniform_random =120 cases.
Both arms use four ticks, physical_v1, max6750 actions (900s) and idle1800 actions
(240s, the same physical evaluation idle budget as the older comparison). No
optimizer. First seed6Z1RA7J3 verified absent from the frozen training-seed set;
native transitions report four-frame deltas and correct time/potential rewards.
Evaluator and learner identities both alive with advancing counters and fresh
heartbeats. Stderr empty at startup verification. A first partial case is not an
evaluation result.

On next resume: read this evaluator's exact identity/status/results before starting
anything. Let it finish; do not restart on stale telemetry or a read timeout. If
complete, audit all transitions and compare within-seed means at four ticks.
Do not directly compare raw episode action lengths or the old '<1 unit per action'
stationary fraction across holds: their physical units differ. Use frame_delta
and actual HP changes. Different old/new seed sets also preclude a paired
before/after timing effect estimate. This diagnostic is not a replacement for the
100-unique-seed full-floor consistency gate once meaningful wins exist.

## Evidence boundaries

The saved analysis includes exact hashes of completed native trace/transition/
episode/manifest files, plus completed-byte prefixes of the growing training logs.
The treatment table uses its episode prefix **694149 bytes**, SHA256
`e9bb3cc9d3d410771075e4a167db734d8a24f33c5bc07b6001eb17e09eae14ae`, ending at
aggregate step3,326,058. Update prefix **678714 bytes**, SHA256
`02fe9ac87a55f4a5c37938226a5807acda98365e2915a00132bde9880dbc14fe`, ends at3,325,644.
Subsequent live counters can exceed these frozen report boundaries.

Reproduce the transition audit with the existing read-only-game analysis command
and a fresh output filename:

```powershell
py -3.10 -m isaac_rl.behavior_report runs/baseline-20260914 `
  --output runs/new-baseline-review.json --training-run runs/ppo-control-v6
```

The generic movement-ablation prose in evaluator manifests describes an optional
arm. Both actual active-arm lists here are stochastic/uniform_random; neither
suppresses movement.
