# Current handoff — 2026-09-16

## CURRENT: September 16, 09:40 Toronto — six-worker training LIVE; frozen comparison LIVE

Training remains **learner30608 / creation1789528910.5677547**, session
**session-1789528914816534000**, `runs/ppo-observability-v8`, same six
instances0/1/2/4/5/6 and ports9999/10000/10001/10003/10004/10005, CPU/two threads,
four ticks, combat_history_v3/combat_v2. **Do not stop/reconfigure/reconnect this
learner.** This turn made no training mechanism, setting, checkpoint or native
training-process change. Current native training identities remain those below.
On review: PID+creation verified live, all six established sockets, heartbeat<1s,
empty current stderr. Update5664 saved5,796,396; later5677 also completed. Windows
commit52–56%,28–31GiB available before reserved comparison; no overnight crash.

Preceding goal turn was a VERIFIED WAIT on the same live learner; its user-facing
answer was interrupted, but no process action was in flight. This resumed turn
completed the scheduled review and launched new bounded evidence collection.
Archived2681 treatment episodes show **zero recorded boss wins**,26 encounters,
813 episodes with a living combat clear. Last500: zero wins,4 encounters,165
living-clear episodes,204 idle/296 deaths. These are training logs, not held-out
competence or a causal verdict about observability.

**Protocol deviation:** planned review5,214,636 passed overnight, but trainer
retains only latest.pt, no checkpoint at that threshold. No historical target
weights can be reconstructed from metrics. Immediately froze the first available
review checkpoint at **5,797,932 decisions /5665 updates /13,306,165 native ticks**:
`runs/observability-review-20260916/treatment.pt`, SHA256
**d4d6421021683a3d8dd51c2323ff6a75fb7714b5e969f5b03ba1db9125f3c7f6**.
Parent4,014,636 copied byte-for-byte into the same directory, SHA256
**58a28f1524b62dd64432bda0ae63a19ac3db4ce312c13a9401d31ce1f429db46**.
Actual additional exposure1,783,296; overrun583,296. Selection preceded held-out
results. **The original fixed-budget falsification is not assessable as
preregistered.** Late results must remain explicitly exploratory; do not silently
reset the budget or substitute an even later favorable checkpoint.

New **comparison34416 / creation1789565921.058696**, output
`runs/observability-review-20260916/comparison`, exclusively owns reserved10002.
Reserved native **13848 / creation1789565921.170414** uses existing private
worker03 executable/resources and bridge0.1.6. Hashes checked; only bridge enabled,
no fixture. Native metadata saved in native-process.json and runtime process.json.
No training instance/Steam/personal save was touched. The comparison is finite:
100 native seeds selected BEFORE scoring,200 full-floor episodes total,
counterbalanced parent/treatment order, same per-seed stochastic action RNG.
Exclude both frozen checkpoints' seeds, all archived training episodes, observed
status seeds and the previous uncheckpointed crash tail. Seeds/exclusions saved.
Hold4, max6750, idle900 inherited exactly; generic evaluation's longer idle
default is NOT used. Models frozen, no optimizer, no action ablation/curriculum.

Comparison implementation **35e6007**; **163 tests pass**, including full fake
native main-loop integration, one shared reserved bridge, exact pairing, no Adam
step, seed exclusions, independent native-clear evidence and budget guard.
Raw native trace and action transitions are gzip-compressed without dropping
states. Each episode stores initial/final states and before/after room-clear
evidence; dead/uncleared/enemies-remaining states cannot count as living clears.
Counter disagreement with derived telemetry blocks a comparison verdict.
Complete100-seed results report conservative Wilson delta bounds and boss gate.
Because budget_compliant=false, original_fixed_budget_hypothesis_rejected is
ALWAYS null, even if late-checkpoint bounds exclude a10-point improvement.
Parent comparison cannot isolate representation from additional training time.

At first live verification the comparison had selected33/100 native seeds,
its exact process and10002 socket were live, training had advanced to5,817,660,
and both stderr files were empty. Inspect comparison/status.json plus actual
PID+creation before any action. Do not reconnect/restart a live evaluator; finite
completion will perform an unscored fresh-room reset, close10002, and exit.
Final handoff check: all100 seeds selected, unique and disjoint from exclusions;
parent pair0 underway at920 decisions (no complete episodes yet). Native actions
are advancing, not merely a queued process. Training saved5,824,044/update5682;
seven established sockets have correct separate owners, both heartbeats<2s,
both stderr files empty, memory56.7% commit/27.7GiB available. Reserved window
hidden after normal menu startup. `live-verification.json` records the checks.
No automatic restart policy. On next resume inspect result.json/episode evidence
only after actual completion; do not interpret partial results as final.
Training charts remain **http://127.0.0.1:8769/**. Goal competence UNPROVEN.

## CURRENT: September 15, 23:27 Toronto — recovered six-worker learner LIVE

**Learner30608 / creation1789528910.5677547**, session
**session-1789528914816534000**, trains `runs/ppo-observability-v8`, uncapped,
CPU/two Torch threads, instances0/1/2/4/5/6 on9999/10000/10001/10003/10004/10005.
**Preserve live training. No new stop or automatic restart authority.** This
manual goal resume found learner2152 genuinely terminal; repair/recovery did not
stop a live learner. Older LIVE/recovery-preparation sections below are historical.

Repair commit **a3f0aa3** replaces nonunique InitSeed observation association
with entity-lifetime GetData track_id. Bridge **0.1.6/combat_v2**, SHA256
**f9d28a8eb4c18267c62c2601984fc6908ce8e121615ee78d45470474c8894c76**.
All six new initial native states confirmed schema/version/normal Isaac/floor1.
Only metadata changes in config: observation_layout, origin and source_hashes.
No changes to input shape, learned weights, optimizer, reward, entropy, timing,
rollout/episode/idle schedules or six-worker count. No new learning intervention.

Byte-identical recovery checkpoint **4,406,316 decisions / 4,759 updates /
7,741,857 counted native ticks**, preserved under
`runs/identity-recovery-20260915-2303/checkpoint.pt`, SHA256
**e97537991fadc3d7c2e3d2ef232f5281df3aa85bfa516ef9cd7729ecbf3cd3be**.
Parent-final4,014,636 and original5,377,722 control checkpoints remain intact;
their SHA256 values were reverified. No rollback to an older baseline.
Resumed-initial snapshot **722e18d281dd4973cc37ee93fb09f0844b365b091ce8d8e4de2c7bf4312202cc**
matches model/Adam/RNG/counters/recent/loss/settings exactly. Training seed set
only adds the six fresh initial seeds. Current source hashes match. Metadata
upgrade accepts only the exact old architecture2 manifest, not arbitrary layouts.

First actual new update **4760 / 4,407,852 decisions / 7,748,001 native ticks**:
collection38.1404s, PPO1.5533s, wall41.9117s,36.6485 aggregate decisions/s. CPU
remains appropriate. Archived first-observed-optimized.pt is the SECOND update
4761 /4,409,388 /7,754,142, SHA256
**15b0cc7ec6d217ce3e20630805934ba2b5e6e67135032725c48ce0cc308a8f3d**.
All16 parameter tensors changed and remain finite; settings/Adam groups unchanged.
Update4763/4,412,460 also completed; six established sockets owned30608 and
dashboard8769 lifecycle ALIVE verified. New stderr232150 is empty. Windows
commit observed61–62%,24GiB available; not a guarantee against long-run leaks.

Native identities:0=19508/1789528905.762303;1=17580/1789528905.833454;
2=31784/1789528905.846059;4=6852/1789528905.8606613;
5=38476/1789528905.8764212;6=7680/1789528905.8921103.
Existing private executable hashes verified; only bridge main/metadata replaced.
Instance0 had already exited with a clean shutdown log before scoped cleanup
(cause unknown). The other five old games and reserved probe35996 received
exact-identity WM_CLOSE and exited0. Probe fixture moved to recoverable
deactivated-native-fixture outside runtime. Reserved3 STOPPED; no training
fixture, personal-save changes, Steam changes or game resource recopy.

**156 tests pass**, including both real trainer migration paths; native fixture
passes16 checks. Six native snapshots reproduce the original validator error;
25 collision-free four-history inputs match the archived encoder bit-for-bit.
Actual historical crash entity remains unknown. Future exceptions now save raw
failure states without overwriting the optimized checkpoint. Damage accounting
still uses seed keys and is a separately documented debt, not changed here.

Before the crash:581 logged treatment episodes, zero boss wins,6 boss encounters,
174 episodes with at least one living combat clear (203 total). These are not
held-out evidence. Only391,680 optimized treatment decisions had completed,
32.64% of the1.2M budget; no intervention verdict yet. Review threshold remains
**5,214,636 cumulative decisions**, not reset by recovery. Future held-out seed
selection must exclude both checkpoints' sets AND recovery's
observed-training-seeds.json (uncheckpointed crash tail). Logged episode counters
can overlap after recovery; distinguish session_id. Manifest/experiment metadata
record this. Charts **http://127.0.0.1:8769/**, dashboard33564; old8768 is baseline
history. This turn made actual training PROGRESS. Goal competence unproven. PAUSE.

## CURRENT: September 15, 23:20 Toronto — identity crash repair; recovery in progress

Prior learner2152/session-1789512773700976500 is TERMINAL, error/exit1 and
cleanup complete at1789523412.6773946. PID absent and all six ports free were
verified, not inferred from an old heartbeat. Exception: Invalid combat_v1
state: entity identity mismatch or duplicate. Saved4,406,316 decisions,
4,759 updates,7,741,857 counted native ticks; last status4,407,822. Unoptimized
post-checkpoint collection is not counted as learned exposure. The run completed
255 treatment updates (391,680 decisions,32.64% of the planned1.2M). Logs contain
581 treatment episodes, zero boss wins. No held-out competence claim.

Evidence: runs/identity-recovery-20260915-2303, including byte-identical checkpoint
SHA256 e97537991fadc3d7c2e3d2ef232f5281df3aa85bfa516ef9cd7729ecbf3cd3be,
source-before.zip (36401e7), logs/status/config and all six native logs. The
actual failing raw frame was not retained, so its precise entity is UNKNOWN.
Native fixture confirms separate projectiles can share InitSeed: six snapshots
reject with the old validator. Repair assigns entity-lifetime GetData track_id;
bridge0.1.6/combat_v2. No numeric IDs enter the policy. Ordinary collision-free
inputs match bit-for-bit across25 native snapshots; model/Adam load exactly.
156 tests and reserved native fixture16 checks pass. Existing dimensions,
weights, reward, optimizer, control, schedules and six workers are unchanged.
New failure-state capture preserves raw responses without saving partial rollout
counters over the optimized checkpoint. No recurrent mechanism added.

This manually resumed goal is recovering verified stopped training, not stopping
a live learner or authorizing automatic restarts. Old native identities below
will be closed only after exact PID/creation/exe checks. Reserved3 is only a
fixture probe and must be closed/deactivated before launch. Only bridge files
are replaced in private runtimes; do not recopy game resources, change personal
saves, or touch Steam. Launch same run with -Resume, CPU, instances0/1/2/4/5/6,
Steps0, dashboard8769. First review remains5,214,636 (do not reset experiment
origin to the recovery checkpoint). Update this current section after actual
six-worker optimizer progress is verified. No automatic stop/restart policy.

## CURRENT: September 15, 18:52 Toronto — enriched-history SIX-WORKER training LIVE

**Learner2152 / creation1789512768.613361**, session
**session-1789512773700976500**, trains `runs/ppo-observability-v8`, uncapped,
CPU/two Torch threads, six instances0/1/2/4/5/6 on9999/10000/10001/10003/10004/10005.
**The user-approved one-time handover is COMPLETE. Never stop this live learner
without new approval. This is not an automatic restart policy.** Older pending
approval and completion-run sections below are historical.

Only learning direction: native combat enrichment plus four snapshots/three
intervening executed actions, observation `combat_history_v3`, architecture2.
Bridge **0.1.5**, SHA256
**d2272113015894565d810e4934b3ebfa692196e6961c6b29130d6a81822c512d**.
All six initial native states confirmed combat_v1/0.1.5/normal Isaac/floor1.
Worker telemetry confirmed four ticks/action, four valid snapshots/12 ticks,
zero omitted hazards and zero invalid laser geometry in the observed sample.
Reward completion_v4, entropy .002, timing physical_v1, gamma/GAE, rollout256/env,
batch256, Adam/groups, hidden layers/action heads, episode6750/idle900 and all
other training settings remain unchanged. No evaluation/performance claims.

Parent learner31448 saved/exited with code0 at **4,014,636 decisions / 10,397
historical episodes / 4,504 updates / 6,175,744 counted native ticks**. Its six
native games received scoped WM_CLOSE and exited0. Final parent is preserved
byte-identically in `runs/observability-intervention-20260915/parent-final.pt`
and `runs/ppo-observability-v8/parent.pt`, identical to completion-v7/latest.pt:
SHA256 **58a28f1524b62dd64432bda0ae63a19ac3db4ce312c13a9401d31ce1f429db46**.
The older3,970,764 preparation snapshot was NOT used for this fork. Previous
5,377,722 control checkpoint and all archived baselines remain intact.

Initial activation23940 exited error/cleanup complete before any PPO update:
native bomb omitted nonexistent ExplosionCountdown. Failure at4,014,666
reported decisions; saved checkpoint remained4,014,636. Evidence retained in
activation-failure and old session/stderr files. Verified learner absent and
ports released before reloading scoped games. Corrected export uses native
**bomb age FrameCount, not remaining fuse**. No fuse getter is available in
vanilla. Same vector width, no optimized weight used the invalid slot. Commit
**4e15780** fixes that implementation error; initial implementation75e5614.
**142 tests pass**, plus expanded native bomb/geometry/cooldown fixture's12
checks over24 actions. Reserved probe29000 was closed (exit0); fixture moved
to deactivated-bomb-fixture, outside runtime mods. Reserved instance3 is STOPPED.
No training fixture, personal-save changes, Steam changes or gameplay assistance.

Corrected native game identities (existing executables/resources retained;
only bridge main.lua/metadata.xml replaced, old copies/logs archived):
0=25328/1789512764.3619049;1=13632/1789512764.3801577;
2=12396/1789512764.3941047;4=16968/1789512764.4084795;
5=32288/1789512764.4234695;6=31552/1789512764.4394803.
Only bridge scripts loaded; native process metadata updated. Native-after.json
records the failed-start generation; native-final.json is the CURRENT generation.

First activation initial snapshot3f589bf39316b85608fcaadfed501345f582191696ec95df1a3fa0ba8fe314f4
matches the final parent's model/Adam after zero expansion; original moments,
optimizer steps, RNG and counters exactly retained. Corrected-start initial
snapshot `treatment-resumed-initial.pt`, SHA256
**5676db50c5e0dbcc0ca258be76c3c485bdc4950220ad8886c42634cbf57cac35**,
exactly matches that model/Adam/RNG/settings and uses current source hashes.
Verification artifacts initial-verification.json/resumed-initial-verification.json.

First REAL optimized update **4505 / 4,016,172 decisions / 6,181,886 native ticks**:
1,536 decisions,6,142 native ticks; collection38.0885s/PPO0.3693s/wall40.7005s,
37.739 decisions/s. CPU remains appropriate; PPO early stopping unchanged.
Frozen `first-observed-optimized.pt` is after the SECOND update4506,
4,017,708 decisions/6,188,027 native ticks, SHA256
**23955e872338aa98e316e78c5bd529184edaab9771f2f2c0fa6035447745a3df**.
All16 parameter tensors changed and finite; new input columns/channels acquired
nonzero weights; Adam groups/source hashes match. See optimized-verification.json.
Third update4507/4,019,244 also completed. New stderr185248 is empty; old184638
stderr is preserved failure evidence, not current failure. Windows commit about
49–50%,31–32GiB available. This does not prove the long-run allocation risk fixed.

**Current charts http://127.0.0.1:8769/**, dashboard33564. API lifecycle ALIVE,
serves observability-v8/new learner. Old8768/dashboard30664 remains the completion
baseline's read-only history. No current stop.request. First review threshold:
**5,214,636 cumulative decisions** (+1.2M from final parent); nominal cumulative
native ticks10,975,744, report exact early-terminal ticks. This is a manual review
threshold, not an automatic training cap/stop/restart. Manifest and experiment.json
record the seed-held-out falsification condition. Goal competence still unproven;
leave detached training running. This turn made actual training PROGRESS. PAUSE.

## CURRENT: September 15, 18:38 Toronto — observability prepared; old six-worker run STILL LIVE

Learner **31448 / creation1789509120.8056352**, session
**session-1789509125636090700**, is still training `runs/ppo-completion-v7`
with the same six native workers and its already-loaded original code/bridge.
Last check **4,001,532 observed decisions / 4,495 updates**, saved4,001,484;
heartbeat0.92s, all six established sockets owned31448, empty stderr, no
stop.request. Commit49.92%,32.01GiB headroom. Charts remain8768/dashboard30664.
**No live learner/native training worker was stopped, reconnected or reconfigured.**

Latest requested intervention is implemented as `combat_history_v3`: enriched
native combat state plus four snapshots/three intervening actions. See
EXPERIMENT-OBSERVABILITY-20260915.md for mechanism, exact unchanged settings,
limits and explicit falsification at+1.2M decisions on100 held-out full-floor
seeds. No GRU/LSTM, reward change, control change or optimizer tuning. New input
weights/moments zero-pad the existing policy. Architecture2 records input-width
growth; new schema must not be hot-swapped into architecture1 training.

**A one-time checkpointed handover was requested through the user-input tool;
no answer has been received and no handover occurred.** Preserve this learner
until that approval. If approved, capture its FINAL current checkpoint as
parent-final.pt, hash it, and fork that checkpoint. Do not roll back to the
preparation baseline simply because it is already archived. No automatic restart
policy is authorized. Update/restart only the six private games after their
learner is cleanly stopped; confirm schema and process identities first.

Preparation artifacts: `runs/observability-intervention-20260915`.
Byte-identical then-current baseline **3,970,764 steps / 4,475 updates /
6,000,323 native ticks**, SHA256
**a413289a52b31b6e71ef6a761e1da6e7d6575b7eb23f8e9bb338d4c05c337af5**;
source archive from40266ae. Existing5,377,722 checkpoint and all prior baselines
remain intact. `manifest.json` records constants and falsification threshold.

Verification: **140 tests passed**, including actual Lua export and real
single/six-environment PPO fork/resume tests with fake bridges. Native fixture
smoke on reserved instance3, **not a performance evaluation**, passed24 actions:
four native ticks/hold,133.364ms median; true firing cooldown, hitbox, damage
invulnerability, enemy state, projectile variants and line/ring geometry observed.
Four-snapshot history spans12 ticks. `GetSamples()` returned NaNs for native
circles; exporter now uses radius for rings, endpoints for lines and explicit
validity for sampled curves. Native curved attacks remain unverified; Lua
sampling/invalid-geometry regression tests pass. Raw failed and passing evidence
is retained under native-smoke through native-smoke-4.

Disposed of own probe games28704 and30488 after verifying exact identities.
**Reserved instance3 is now STOPPED**, no listener on10002. Its bridge file is
0.1.4; its temporary fixture was moved out of runtime mods to the artifact
directory's `deactivated-native-fixture` (recoverable). Never copy that fixture
to any training installation. No other Steam/game/save state was changed.

Offline native-input audit: exact zero initial logit/value difference from the
frozen parent. Disposable synthetic-copy CPU PPO timings .448s ->1.374s;
six-env inference .640ms ->1.058ms; rollout observations28.55MiB ->155.33MiB.
These are resource/migration checks, not optimized native treatment checkpoints.
The requested learning intervention is READY but **has not trained any live
steps**. Git tracks source changes; live learner continues using loaded old
source despite changed working-tree fingerprints. Goal competence unproven.

## CURRENT: September 15, 17:52 Toronto — recovered stopped run; six workers LIVE

**Current learner31448**, creation **1789509120.8056352**, session
**session-1789509125636090700**, continues `runs/ppo-completion-v7` unchanged:
six workers0/1/2/4/5/6, CPU, four ticks/physical_v1, completion_v4, entropy0.002,
original architecture/observations/optimizer/schedules. **Never stop this live
learner.** This was recovery after a verified terminal failure and subsequent
machine reboot, not a stop of live training or an automatic restart policy.

On resume, old learner46824 was absent, status error/exit1/cleanup complete at
**3,897,150 observed steps** with a reset receive timeout. Its final update showed
**99.594% committed memory,0.393GiB headroom**. Worker1's native log recorded failed
texture allocations (1–8MiB), then an exception/minidump. The traceback timed out
inside reset_done → IsaacEnv.reset → Bridge.request. The last idle episode was
on10000. All native games and Steam were absent after a later reboot; boot time
1789508333.4582417. Current commit before recovery was31.50%,43.79GiB headroom.
No restart decision was based merely on observation timeout or a stale status file.

Failure evidence preserved in **runs/recovery-20260915-1747**: native logs for all
six workers, learner stderr/stdout/status/config, complete pre-resume episode/update
logs and a byte-identical checkpoint. Saved checkpoint **3,897,036 steps / 10,220
historical episodes / 4,427 updates / 5,705,526 native ticks**, SHA256
**b08dc2f3d73a66585824bee7ddf7894afbdfed51f786e93e0630fccc0e0ff94a**.
The114 later transitions were not saved into policy/optimizer state. The two later
episode log records are retained; new session IDs distinguish resumed counters.
No old logs/checkpoints were deleted and the original3,419,340 baseline is unchanged.

Before recovery:744 completed treatment episode records,0 wins,7 boss encounters,
282 combat clears,530 deaths/214 idle. Episode-prefix419340bytes, SHA256
**43e98ed50bed4c76d5156f0803a12d1036b914711a4874a01fc77739d5ee17fa**.
Treatment exposure was about1.91M native ticks, below the planned+2.4M review point;
no competence conclusion or second learning intervention was introduced.

Started the normal Steam client, then reused existing private executables/resources
without recopying from the original installation. Each executable matched its
installation hash; each bridge matched the unchanged0.1.3 hash; only the bridge mod
was enabled. Native startup logs confirmed only game base scripts and the bridge,
normal runs, and no allocation/minidump errors. New native identities:
0=31892/1789508991.6291175;1=9764/1789509064.0075495;
2=31108/1789509064.6309474;4=31920/1789509065.3370583;
5=30516/1789509066.16111;6=25992/1789509066.9162357.
Normal menu startup completed; workers1/2/4/5/6 hidden afterward. Reserved instance3
was not relaunched. No personal saves, original game files, pagefiles, or unrelated
applications were changed or closed. No new evaluation launched.

Resumed via launch_parallel.ps1 -Run runs/ppo-completion-v7 -Resume
-Instances0,1,2,4,5,6 -Device cpu -Steps0 -DashboardPort8768.
Archived **resumed-initial.pt** SHA256
**1056b681e56bce895612412a13c954da1f131d58fde07ef32280d0c29affb788**:
model, Adam, Torch/NumPy RNG, counters, recent/loss windows, schedules, timing,
reward/observation profiles and source hashes exactly match the saved checkpoint.
Only config resume path/origin provenance changed. No production code changed.

First resumed PPO update **4428 / 3,898,572 steps**,1,536 transitions/6,142 native
ticks; collection42.770351s/PPO0.536960s/wall45.966643s,33.4155 decisions/s.
Frozen **first-observed-resumed.pt** is after the SECOND update4429 at3,900,108,
native ticks5,717,809; SHA256
**e13529d5e7d4b2baab999a97f669923aeef43427d5009e06339d3069f1d21039**.
All16 model tensors changed and finite; Adam groups unchanged; source hashes match.
Verified live beyond3,900,636, all six established sockets owned31448, all six frame
deltas4/profiles correct, zero damage/kill payments, empty new stderr
**runs/ppo-completion-v7/stderr-20260915-175200.log**. First update memory42.72%
committed/36.62GiB available, later44.05%/35.77GiB. This is current headroom, not a
claim that the underlying risk of future system-wide memory exhaustion is fixed.

**Charts remain http://127.0.0.1:8768/**, new dashboard **30664**. API verified to
serve this run and learner31448. Old dashboard46344 disappeared with the reboot.
Next review threshold remains cumulative **6,195,560 native ticks**; do not reset
experiment exposure to zero on session recovery or compare rewards across profiles.
The previous completed goal turn was PROGRESS (reward handover); the interrupted
continuation issued no actions. This turn is PROGRESS (verified recovery and actual
optimized training). Leave detached training running; goal remains unproven; PAUSE.

## CURRENT: completion-reward six-worker training LIVE; approved handover complete

The user said **"make the changes"** in response to the one-time handover question.
That handover is now complete. **New learner46824**, creation
**1789476850.9710472**, session **session-1789476854270681600**, run
**runs/ppo-completion-v7**, CPU/two Torch threads, uncapped, six instances0/1/2/4/5/6
on9999/10000/10001/10003/10004/10005. **Never stop this learner without new approval**;
the consumed exception is not an automatic restart policy. Previous pending-approval
and blocked notes below are historical. This turn is PROGRESS: native launch and
first real optimized checkpoint verified; full-floor competence remains unproven.

Only training mechanism changed: **confirmed_v3 → completion_v4**, damage0.2→0
and kill0.5→0. Combat-clear10, boss100, all other rewards, raw event counters,
idle semantics, architecture1/terrain_v2, action heads9/5/4, four ticks/physical_v1,
gamma/GAE lambda, rollout256/env/minibatch256, max6750/idle900, entropy0.002,
optimizer and seed remain unchanged. Full tests rerun: **113 passed**.

Old learner20984 saved/exited normally at **5,377,722 steps / 12,386 historical
episodes / 5,391 updates / 11,625,971 controlled native ticks**, exit0 and cleanup
complete. Exact identity absent and all six ports released before the new launch.
Final checkpoint preserved in `runs/completion-intervention-20260914/parent-final.pt`,
SHA256 **31740cdf0a3a99fbdf8415dcbaaf62b2240e9483233ab2c8cc54899420661f49**.
Final status/states, pre-handover checkpoint/status and authorized stop marker
archived alongside. Old run's stop.request intentionally remains as a resume guard.
No checkpoints or training history deleted.

Treatment intentionally starts from the previously frozen **3,419,340** baseline,
not the overnight final checkpoint. `baseline.pt` and new run's `parent.pt` both
SHA256 **e1498fbcade44ff8f562220735ff4a94bc933534bd9fed6c98ad3a120c76e0b9**.
Native counter starts at inherited3,795,560, retaining origin2,470,092 steps.
The old branch's additional training remains archived, not merged into this fork.

First full update **4117 / 3,420,876 steps**:1,536 transitions,6,140 actual native
ticks (early deaths shorten some holds), collection38.241356s, PPO0.406807s,
wall41.158942s, **37.3187 aggregate decisions/s**. Frozen after-update checkpoint
`first-observed-treatment.pt`, SHA256
**1121ec697dedd94063a3ab52cfda126ee5c916160e570b8efe2c29e7074f192a**:
all16 model tensors changed and finite; Adam parameter groups and inherited
settings unchanged. A first attempted initial-state assertion correctly failed
because the copied checkpoint had already optimized; the artifact was renamed
and verified as this first-update checkpoint, not misreported as pre-update state.

Verified beyond3,423,420/update4118: all six established sockets owned46824,
all six frame deltas4, completion_v4 in worker telemetry, zero damage/kill payments
while raw damage remained recorded. Source fingerprints match the launch manifest.
Learner stderr `runs/ppo-completion-v7/stderr-20260915-085410.log` empty. Native
game PIDs unchanged:0=6996,1=13120,2=34576,4=24640,5=39876,6=25388; all six bridge
hashes unchanged. Worker0 reached Game Over during the gap; screenshot and native
log confirmed it. With exact game/learner identities, listener46824 and no9999
connection verified, one normal Space restart input restored it. No game process
or new learner restarted/killed; no other menu input or visibility change.

**Current charts http://127.0.0.1:8768/**, dashboard **46344**, HTTP/API verified to
serve ppo-completion-v7 and learner46824. Old8767 remains baseline history; earlier
8766/8765 likewise. First update commit95.66%,4.20GiB available. Do not add games
or modify unrelated processes/saves. Reserved evaluator10544 already finished;
instance3 remains untouched. **No new evaluation launched.**

Next manual review at **+2,400,000 treatment native ticks**, i.e. cumulative
**6,195,560** (roughly4,019,340 aggregate decisions; early-death holds alter this).
Compare actual combat completion, health and boss outcomes, not shaped returns
across profiles. Do not introduce another mechanism or default to tuning before
reviewing this run. Revalidate exact identities on resume. Leave detached training
running and end bounded work with PAUSE. See EXPERIMENT-COMPLETION-20260914.md.

## CURRENT: September 15, 08:50 Toronto — overnight check; handover still pending

Learner **20984**, creation **1789400666.5494964**, identity verified LIVE with
0.18s heartbeat at **5,369,868 steps / 5,385 updates**, six workers, four ticks,
confirmed_v3. No stop.request; stderr remains empty. Prepared completion_v4 run
**has not launched**. Commit3a7a41f and its frozen3,419,340 baseline remain the
planned intervention; do not silently replace that baseline with newer weights.

Training episode prefix through **5,370,498 steps**: 4,280 control-run episodes,
0 wins,96 boss encounters,2,011 combat clears;3,059 deaths/1,221 idle endings.
Since the previous3,432,780-step check: **2,872 episodes,0 wins,64 boss encounters**,
2,062 deaths/810 idle. Last500:0 wins,7 boss encounters,205 combat clears,
347 deaths/153 idle;0.256 clears per simulated minute. These are descriptive
training outcomes, not a causal comparison. Prefix2,410,152bytes, SHA256
`4340a48ee45565466777bce785eedd3f265239b07f5733d60d805537280f66df`.
Last100 updates averaged collection36.765s versus PPO0.362s; retain CPU.

Existing evaluator10544 is now **FINISHED120/120, exit0, exact identity absent**.
It reports normal unscored cleanup/parking. Episode records: stochastic0/60 wins,
0 boss encounters,23 combat clears,35 deaths/25 idle; uniform random0/60 wins,
0 boss encounters,9 combat clears,42 deaths/18 idle. No transition re-audit or
statistical improvement claim was made this turn. Episodes SHA256
`9e3bc4929606495b9b065262419f341ec352a7612d081d004a5c537a19fc45b9`.
**No new evaluation launched**, no native game/process signalled or restarted.

Next action remains the implemented reward ablation, not another diagnostic or
hyperparameter change. Reusing the six occupied training ports requires the
previously requested **one-time checkpointed handover approval**. A goal resume
alone does not override the explicit never-stop rule. The prior goal turn was a
verified live-process wait ending blocked on that approval; this resumed run
starts a fresh blocked audit (first occurrence), not an immediate blocked update.
The original full-floor success criterion remains unproven. Current charts8767.

## CURRENT: training reward change implemented; one-time handover approval pending

Latest user direction: **"Next turn is a training change, not another evaluation."**
This supersedes the older next-review/evaluation-first recommendations below.
Do not launch another evaluator or wait for the current evaluator as a prerequisite.
The previous completed goal turn was PROGRESS (completed comparison review/new
evaluation launch); the later acknowledgement alone made no progress. This turn
implements a concrete training mechanism change, not another diagnosis-only review.

**Sole factor: confirmed_v3 → completion_v4**, eliminating damage and kill payments
while retaining room-clear10, boss100 and all other reward/control/learning settings.
Damage/kills remain logged and reset idle exactly as before. No changes to observation,
architecture, native bridge, optimizer, entropy, four-tick timing or six-worker count.
See `EXPERIMENT-COMPLETION-20260914.md` for evidence, risks, launch and verification.

Frozen parent **3,419,340 steps / 4,116 updates / 3,795,560 controlled native ticks**:
`runs/completion-intervention-20260914/baseline.pt`, SHA256
**e1498fbcade44ff8f562220735ff4a94bc933534bd9fed6c98ad3a120c76e0b9**.
Model tensors finite; original Adam/RNG retained. Config/observed status/experiment
manifest alongside. Target **runs/ppo-completion-v7 has NOT launched**.

Verification: **113 tests passed**; five new tests cover reward isolation and real
PPO fork/resume on fake bridges for single/six-worker trainers. Initial model,
Adam and RNG exactly preserved; updated weights finite and changed; profile
inheritance and in-place reward-change rejection verified. PowerShell parsing,
CLI choices, baseline digest, manifest/constants and diff checks passed. Only
env.py (error message) and rewards.py (new profile) differ from the live learner's
recorded Python/native source hashes. No native behavior benefit is claimed.

**Preserve live learner20984**, creation1789400666.5494964, six-worker
`runs/ppo-control-v6`; identity verified this turn at3,427,932 with fresh heartbeat,
all six established sockets owned20984, empty stderr, ~3.98GiB commit headroom.
Existing evaluator10544 identity also verified live,14/120 complete. Source edits are
not hot-applied to its imported modules. Never-stop rule still applies: ask for
fresh **one checkpointed handover** before signalling it. Earlier one-time approvals
were consumed; do not add duplicate games or steal their ports. No stop.request or
signal issued. Current charts remain127.0.0.1:8767; next run plans8768 only after
authorized handover. Existing evaluator10544/10002 remains untouched; no new job.

Training prefix through3,420,864:1,387 treatment episodes,0 wins,32 boss encounters,
980 deaths407 idle. Old completed stochastic traces had47 positive-reward health-loss
transitions;43 become nonpositive when subtracting damage/kill payments. This supports testing
the reward mechanism, not claiming a proven unique bottleneck or improvement.

## CURRENT REVIEW: 18:00 Toronto; training unchanged, current-checkpoint evaluation live

Previous goal turn classified PROGRESS (four-tick implementation/verified launch).
This turn is PROGRESS: audited the completed older120-case comparison and launched
a frozen current-checkpoint evaluation on the now-free reserved game. See
`AUDIT-CONTROL-20260914.md` and
`runs/baseline-20260914/analysis-20260914-1800.json`.

**Learner20984 creation1789400666.5494964 remains LIVE**, six workers/four ticks,
`runs/ppo-control-v6`, observed beyond3.333M aggregate steps,3.45M controlled native
ticks. Production source hashes/settings unchanged; no learner/game stopped or
signalled. First review threshold+2.4M ticks crossed. Recorded treatment prefix:
1,233 episodes,0 wins,31 boss encounters;0.545 combat clears/episode versus0.437
in short parent run, but mean episode time92.22s versus~68.93s and idle30.17% versus
17.09%. Clears/minute0.355 versus~0.381. This is descriptive, non-paired evidence,
not a causal improvement/regression estimate. Goal remains unproven. Last100
updates: collection37.116s/PPO0.528s; entropy2.524 is not competence.

Old evaluator6968 naturally FINISHED120/120, exit0, exact identity DEAD. Recomputed
all120 behavior summaries from39,288 transitions. Older2.372M learned and random
both0/60 wins and0 boss encounters. Learned damage32.43 versus16.03, but combat
clears0.25 versus0.10 with exploratory seed-bootstrap difference interval[-0.033,
0.350]. No robust boss or survival competence follows from the proxy improvement.

**New evaluator PID10544**, creation **1789423443.6716738**, output
**`runs/control-eval-20260914-1800`**, reserved **10002**, native **47392** unchanged.
Frozen **3,327,180** / native ticks3,427,089, SHA256
**7a1b5cdd49ed2abdcbbc0b3ccc2a2bf18d9bb58901a72284ed8b60bd5357e74f**.
20 unseen seeds x3 repeats x stochastic/uniform_random =120. Four ticks,
physical_v1, max6750/idle1800 (900s/240s evaluation budgets); no optimizer.
Startup verified advancing beyond880 actions, fresh heartbeat, empty stderr,
first seed absent from checkpoint training seeds, correct native four-tick deltas.
Preflight port unowned/game identity alive,4.10GiB commit headroom. No game restart
or extra native process. Stdout/stderr adjacent to the evaluation output folder.

Next manual resume: confirm exact learner/evaluator identities and inspect this
evaluation, not the completed older one. Let it finish; do not infer terminal state
from observation timeout. Compare within-seed outcomes and physical time, not raw
four-tick/eight-tick episode lengths or displacement-threshold fractions. Preserve
the current single-mechanism training run; no hyperparameter tweaks, reward changes,
or automatic handover authorized. Current charts remain127.0.0.1:8767. End bounded
review with PAUSE, leaving training running.

## CURRENT: four-tick six-worker training LIVE; handover completed

The user said **"make the changes"** after the verified action-hold audit. That
one checkpointed handover is now COMPLETE. **Current learner PID20984**, creation
**1789400666.5494964**, session **`session-1789400670817434800`**, run
**`runs/ppo-control-v6`**, CPU/two Torch threads, six workers0/1/2/4/5/6, uncapped.
Never stop this new learner without further user direction; no auto-restart policy.
See `EXPERIMENT-CONTROL-20260914.md` for the timing experiment and exact conversions.

Four ticks/action (133 ms), physical_v1, gamma0.997496867163, lambda0.974679434481,
time cost-0.005/full action, max6750 actions/idle900, rollout256/env/batch256,
visit count multiplier0.5. These preserve physical-time quantities and nominal
SGD steps per simulated time. Keep entropy0.002, event rewards, architecture,
action heads, observation layout, optimizer settings and native physics unchanged.

Old learner21432 saved/exited normally at **2,470,092 steps / 8,094 episodes /
3,498 updates**, exit0/cleanup complete, exact old identity absent and six ports
released before launch. Final baseline preserved as
`runs/control-intervention-20260914/baseline.pt`, SHA256
**2e4830a744508c4a98ba157e4806eedb7d5000c99e026cc6115f6fc84fd33b50**.
New run's `parent.pt` matches exactly. Parent config/final status/final states and
authorized stop marker archived alongside. Older entropy baseline retained.
Old `runs/ppo-entropy-v5/stop.request` intentionally remains to prevent accidental
resume. No files deleted. All native games reused without restarts or menu input:
0=6996,1=13120,2=34576,4=24640,5=39876,6=25388; runtime bridge hashes unchanged.

First full update **3499 / 2,471,628 steps**, 1,536 transitions, 6,141 actual native
ticks (three fewer than full holds because of early death), collection37.391s,
PPO0.660s, wall40.199s, **38.21 decisions/s aggregate**. Independently archived
optimized checkpoint after update3500 at2,473,164, native ticks12,282, SHA256
**6dd8cc9a0e247f67add3cc370b46f6cc91682f4b9ff75dc1f3e69375243c4ce4**,
`first-observed-control.pt`: all16 model tensors changed and finite, Adam groups
unchanged, metadata/schedule/entropy correct. Source fingerprints match. All six
ports owned20984; all six worker frame deltas observed4. Empty learner stderr
`runs/ppo-control-v6/stderr-20260914-114426.log`. These are bounded observations,
not claims of continual liveness or improved competence. Tests **108 passed**.

**Current charts http://127.0.0.1:8767/**, dashboard **PID52820**, HTTP verified to
serve the new run and learner. Old8766 entropy and8765 damage dashboards retain
history and show old learners dead. Independent baseline evaluator6968 continues
on10002, last checked56/120 episodes; do not interrupt or wait for it as a gate
to this already-live intervention. Windows commit96.67%,3.21GiB available at the
first update; avoid adding native games or large diagnostic processes.

Review after **+2,400,000 controlled native ticks**, then+4,800,000. This is roughly
600k/1.2M new decisions, equivalent to300k/600k old eight-tick actions. Exact
`native_frames` accounting starts at `native_frames_origin_steps=2470092`, excludes
operational resets, and does not invent historical tick counts. Compare held-out
boss outcomes, combat clears and health; the mixed-duration aggregate steps axis
alone is not a fair exposure comparison. Goal remains unproven. End bounded reviews
with PAUSE while leaving training running.

User clarification: **hyperparameter tuning in general is not the default first
intervention**. Prioritize mechanism-backed control, observation, objective, data,
and learning-structure fixes. This is also recorded in AGENTS.md.

The lower-entropy trial is being superseded by explicit user direction, not declared
successful or failed. Preserve its final checkpoint and retain entropy0.002 in the
next run; do not combine the control change with reverting entropy. Older sections
below are historical and do not block this specifically authorized handover.

## Historical: lower-entropy six-worker training handover

The user explicitly answered **"yes"** to one checkpointed stop of the existing learner and launch on the same six workers. Handover completed. **Current learner PID21432**, process creation **1789397760.954275**, session **`session-1789397765416817800`**, run **`runs/ppo-entropy-v5`**, CPU, six workers0/1/2/4/5/6, uncapped. **Never stop this learner**; the one-time approval is consumed, not an auto-restart policy.

Only training factor changed: **entropy coefficient0.02→0.002**. Rewards, observations, architecture, optimizer hyperparameters, rollout128 per worker/minibatch128/epochs4, eight-update actions, episode limits, seed setting and six-worker count verified unchanged against the baseline configuration. All six runtime bridge files retain the original hash. New checkpoint metadata records0.002 and inherits it on resume.

Preserved parent **2,417,208 steps**, SHA256 **eeffc748b50552e5947e3575585d92e1294f60dadc6b0f8e94996ac026a294ae**, remains `runs/entropy-intervention-20260914/baseline.pt`; the new run's `parent.pt` matches exactly. Old learner51580 saved/exited normally at **2,425,314 steps / 7,916 episodes / 3,440 updates**, exit code0, cleanup complete, exact PID identity absent and ports released before launching. Archived old final checkpoint hash **d9759e25ca650403db30d4e6750397e30f1272bf1a51e0526d332bcf4fca8170**, plus final status/states/authorized stop marker in the intervention folder. Old `runs/ppo-damage-v4/stop.request` intentionally remains to guard accidental resume. No files or old learned weights were deleted. New aggregate counters start at the earlier frozen parent intentionally.

Native games reused unchanged:0=6996,1=13120,2=34576,4=24640,5=39876,6=25388. Instance6 reached Game Over during the gap and was the only unconnected worker. Captured/inspected its screen, guarded on listener21432 plus no established10005 connection, then sent its normal Space restart input. It reconnected and started normal training; no game process or new learner was killed/restarted. Temporary window visibility restored to hidden. Screenshot evidence in the intervention folder.

**First real optimized checkpoint verified:2,417,976 / update3430**, full768 transitions, coefficient0.002 in config/checkpoint/update/status, model weights finite and changed from parent, Adam parameter groups unchanged. Archived `first-observed-treatment.pt` hash **c5b2e7d5ba224810b410529743faa2fb0c22419c66ef534e86526a30eb47e306**. Collection37.021s/update0.357s/wall39.699s/19.35 transitions/s. Second update3431 also completed; learner observed alive above2,419,416. All six established sockets owned21432, stderr `runs/ppo-entropy-v5/stderr-20260914-105600.log` empty. Windows commit~94.98%,~4.79GiB available. These are bounded execution observations, not perpetual liveness or learning-success claims.

**New charts http://127.0.0.1:8766/**, dashboard **PID52888**, start1789397761.659211, verified HTTP serves the new run. Old dashboard6604/8765 remains read-only baseline history and shows old learner dead. Existing independent diagnostic6968/10002 continues separately; **do not wait on it before assessing the active training intervention or revert to diagnosis-only work**.

Next review target: **2,717,208 total steps (+300k treatment)**, then3,017,208 (+600k), without stopping training. Inspect newly collected episode records and full-floor evaluation; entropy/return/movement alone are not competence. Do not add a second intervention before assessing this one. Implementation tests84passed in preceding turn; no source changes beyond launch documentation this turn. Goal remains consistent unseen full first-floor boss clears, unproven. End bounded reviews with PAUSE.

## Historical: user-directed adaptation implemented before handover approval

Latest user instruction supersedes waiting for diagnostic completion: **stop diagnosis-only work, choose one high-value adaptation, keep everything else constant, preserve the current checkpoint, launch next training**. Chosen sole training change: PPO entropy coefficient **0.02→0.002**. See `EXPERIMENT-20260914.md`. Do not continue the earlier pattern of waiting for120 diagnostic episodes before this intervention.

Implemented explicit coefficient configuration in both trainers/launchers, legacy-safe default0.02, checkpoint inheritance/persistence, per-update/status/config reporting, finite/nonnegative validation, and rejection of changes within an existing run. No reward, observation, architecture, action-duration, PPO clipping/epochs/lr, or worker-count changes. Source edits do not hot-change the already-running learner, which retains its imported0.02 code.

Frozen baseline **`runs/entropy-intervention-20260914/baseline.pt`**, steps **2,417,208**, episodes7895, updates3429, SHA256 **eeffc748b50552e5947e3575585d92e1294f60dadc6b0f8e94996ac026a294ae**. Exact model/Adam/RNG/seeds retained and model tensors finite. Original config/status copied alongside; experiment manifest records the single factor and constants. New target **`runs/ppo-entropy-v5`** has NOT launched yet.

Standing user instruction still forbids stopping live training. Learner51580 creation1789334361.456794 remains alive on its six ports; no signal or stop.request was issued. Ask for **one checkpointed handover** to reuse those six workers. Do not infer it from the automatic goal continuation alone. Windows capacity check95.69% committed/4.11GiB headroom, CPU60.5%; six additional native games/another learner risk allocation failure. Do not substitute fewer workers, duplicate existing ports, or change unrelated apps/memory settings. This is a new authority question, not the old verified-wait state.

After explicit approval: checkpointed exit of exact old learner, preserve final old checkpoint/status, verify process exit/port release, then launch the command in the experiment document. Initialize from the preserved2,417,208 baseline, not a silently newer parent. Keep all six instances0/1/2/4/5/6, CPU, original hyperparameters, new coefficient0.002, uncapped. New dashboard8766; existing baseline dashboard8765 remains read-only. Verify first real update and metadata before handoff. The original full-floor goal remains unmet.

Verification before asking for handover authority: **84 tests passed**, both PowerShell launchers parsed, and `git diff --check` passed. Preserved checkpoint digest and held-constant configuration matched the experiment manifest. Live learner identity reverified at2,422,872 steps/six workers. Active stop.request absent; new treatment status absent. Implementation is complete; native launch is the remaining action requiring the one-time exception.

## CURRENT: completed behavior audit; repeated random-control comparison live

Previous goal turn classified **VERIFIED WAIT**: exact learner/evaluator process identities were confirmed live and counters advanced. This turn is **PROGRESS**: analyzed the naturally completed 60-episode diagnostic, reproduced every episode behavior summary from transitions, recorded exact evidence hashes/training-log byte prefixes, added repeatable offline analysis and uniform-random controls, and launched the next diagnostic without touching live training.

See **`AUDIT-20260914.md`** and **`runs/review-20260914.json`**. Through recorded 2,374,002 episode steps: 4,665 corrected-window episodes, zero wins, 80 boss encounters. Last partial bucket (2M onward): 0.439 combat clears/episode, 2.15% boss reach. Last 100 PPO updates average 0.285s versus 37.01s collection. Source hashes recorded by the learner still match; no learner settings, training mods, ports, or processes were changed.

The old evaluator **39916** naturally finished **60/60**, exit code0, and its exact identity is now dead. Frozen1.630M stochastic:0/20 wins,1 boss,19 deaths/1 idle,8.85% stationary same-room combat transitions,17.7% move-zero combat actions. Deterministic:0 wins,15 idle,56.3% stationary combat. Only7/20 movement pairs eligible; the others first diverged in native entities despite identical reported resets. Do not promote all-arm differences to exact causal effects or claim all current stochastic behavior is passive shooting. Small selected movement subset shows no win advantage and does not prove dodging.

**Active evaluator: PID6968**, process creation **1789394939.3799973**, reserved native game **47392** creation1789356107.3601575, port **10002**. Output **`runs/baseline-20260914`**; stdout/stderr alongside folder. Frozen checkpoint **2,372,664**, SHA256 **17a91da77688e2e74aedc1feea058c77b10ec94ac127d5a627c08fc49a0898bf**. Twenty unseen seeds × three repetitions × learned stochastic/uniform random =120 episodes. Same native action space/8-update holds/full-floor reset; counterbalanced arm order; no optimizer. Verified live evaluating beyond88 actions with a fresh heartbeat and empty stderr. Read status and process identity next time; this is not a perpetual liveness claim.

**Trainer remains PID51580**, creation1789334361.456794, six CPU workers0/1/2/4/5/6, run`runs/ppo-damage-v4`, uncapped, advancing above2.37M. Never stop or signal it. Dashboard6604 remains at http://127.0.0.1:8765/. No game restart or extra native process was needed: the previous evaluator parked reserved game3 and released its port. Before attaching the new evaluator, port10002 was confirmed unowned; game3 identity/path were verified; ~6.1GiB Windows commit headroom and149GiB disk space were available.

On next manual resume, check both exact identities and evaluator outcome. Do not restart on a stale heartbeat or observation timeout. Let120 episodes finish. Compare learned versus random combat clears using **seed-level mean differences**, not120 independent observations; report uncertainty, boss/wins, health and endings. Damage alone is not the primary competence measure. The next controlled training recommendation depends on those results; do not silently rewrite the live control. Goal remains unproven. End bounded reviews with PAUSE.

Final verification this turn: **76 tests passed**; `git diff --check` passed. Learner alive at **2,379,000 steps**, evaluator alive at **3/120 completed**, next stochastic repetition advancing. All six training sockets still owned51580; evaluator alone owns10002. The first native uniform-control action was[2,4,1], with exact uniform-head entropies ln9/ln5/ln4 and eight-frame advancement. Frozen checkpoint bytes match the manifest hash. Both active stderr files are empty. Dashboard API still serves the correct run as PID6604. No conclusion is drawn from the first repeated seed. The manifest's generic movement-intervention text describes the optional no-movement arm; the active `arms` list contains only stochastic/uniform_random, and neither suppresses movement.

## CURRENT: fresh behavior diagnostic launched beside live training

First paired seed **HB0Y9PK9** completed all3 arms, with exact matching prefixes and combat entry. Stochastic policy:103 steps,41 damage,6 damage-taken units; combat movement suppressed:32 steps,21 damage,7 damage-taken units. Both died, no combat clears or wins; n=1 is not an aggregate conclusion. First three result records and pair eligibility verified, evaluator still live and moving onto seed2. Learner simultaneously advanced to **1,635,480 steps**. This is direct current-checkpoint behavioral evidence, unlike the earlier1M trace. Player HP before/after is retained; damage-taken callback units are not automatically equivalent to net health loss.

Previous goal turn classified **PROGRESS**: inspected policy/logs, documented the plateau and missing behavioral data, and created local Git commits30234e6/e4e812e. This continuation makes the next experiment concrete. The primary causal bottleneck remains unresolved; `AUDIT-20260913.md` now distinguishes measured poor boss coverage from hypotheses about reward, architecture and optimization. Do not infer competence or a required movement reward from event counts. Do not feed arbitrary replay transitions to vanilla on-policy PPO.

Live learner remains **PID51580**, creation1789334361.456794, six CPU workers, unchanged active core training source, uncapped. Observed advancing above1.63M steps; no learner was stopped or signalled. Goal completion remains unproven. The reserved idle evaluation game3 was normally closed and refreshed to bridge0.1.3 while port10002 was unowned; training games and personal/original game files were untouched.

**Active evaluator PID39916**, creation **1789356160.140311**, owns reserved port **10002**, native game **47392**. Output **`runs/behavior-20260913-v2`**. Frozen checkpoint **1,630,008 steps**, SHA256 **e29a7e274f0cfb8f2d82a0468d05acd2e8306dbcee855b6811bb17e98f4c653c**. Requested20 unseen seeds ×3 arms =60 episodes: deterministic, stochastic, stochastic-no-movement-during-combat. Same per-seed sampling RNG for the stochastic pair, rotating order, exact observation prefix hash through combat entry. No-movement does not suppress navigation in empty rooms. Proposed and executed actions, per-head entropy/probabilities, reward components, positions, raw event deltas and native traces retained. Same-room displacement excludes room-transition teleports. Successful startup verified normal seedHB0Y9PK9, advancing beyond53 actions and real combat, bridgehp_delta_v1, empty stderr. Check PID+creation and current counters next time; these observations are not perpetual liveness claims.

An initial evaluator **31536** naturally exited with error before its first scored episode: a canonical8-character seed had been passed without the native console's middle space. Evidence at **`runs/behavior-20260913`**, stdout/stderr alongside that folder. Fixed native formatting in the new evaluator; targeted evaluation tests **11 passed**. Only after terminal status and absent PID were verified was the idle evaluation game6668 closed normally and relaunched47392. The corrected evaluator uses the exact first attempt's frozen checkpoint, not newer training weights. Do not restart any job solely because observation timed out.

On next manual resume, read this evaluator's status/result/episodes/transitions before choosing a training intervention. Exclude unmatched/no-combat pairs from movement-effect conclusions; raw event/displacement associations do not attribute damage to the immediate action or player alone. One stochastic repetition per seed limits precision. This20-seed diagnostic cannot prove the full success gate. After completed evaluation, the evaluator parks its game in a fresh normal room and exits; it has no learner restart loop. Continue to leave training running and end manual reviews with PAUSE.

## CURRENT: explicit "6 workers now" authorizes one checkpointed restart

Verification PASSED: learner51580 creation **1789334361.456794**, session `session-1789334365873010900`, completed its first six-worker PPO update **1872** at **1,221,432 steps**, checkpoint saved. All SIX trainer-owned established sockets verified and all six worker states reported confirmed_v3/terrain_v2, advancing frames and steps. First measured rollout **19.2946 steps/s**, collection **37.2543s**, PPO update **0.36243s**, wall **39.8038s** per768 transitions. Commit **85.77% / 13.09GiB headroom**. Learner stderr remained empty. Existing worker2 had reached Game Over during the checkpointed transition and did not reconnect until one ordinary Enter menu input; no game process or replacement learner was killed/restarted. Chart service follows the new run session and shows six workers. The standing never-stop-live-training rule resumes after this authorized transition; leave this learner running when pausing agent work. Older pending-six-worker/memory-blocker sections below are superseded.

The user repeated the six-worker command after restart and memory risks were explained. This is approval for the one-time checkpointed transition, NOT an ongoing auto-restart policy. Prior learner51700 was asked to stop through its cooperative stop.request protocol; it optimized/saved and exited normally at **1,220,664 steps / 1,871 updates / 3,715 episodes**, exit_code0, cleanup_complete=true. Preserved final status, checkpoint and the consumed stop.request in `runs/restart-six-20260913`; nothing was deleted. Frozen checkpoint SHA256 **7b6b3c064d7c2a331b44e643ac6e30fb9ed43fff5631a2f0984b5fb1b383b532**, all model tensors finite. Original policy/optimizer/RNG, confirmed_v3 rewards and terrain_v2 observations retained. Current-run stop.request was moved into the archive after verified clean exit.

Launched detached CPU learner **51580** at17:19 using instances **0/1/2/4/5/6**, ports **9999/10000/10001/10003/10004/10005**, --steps0, same run `runs/ppo-damage-v4` resumed from latest.pt. Native games0/1/2 retained as6996/13120/34576; newly started4/5/6 as24640/39876/25388. Evaluation instance3 untouched. Existing localhost chart service6604 reused at **http://127.0.0.1:8765/**. At this restart memory pressure had fallen to **82.01% committed / 16.55GiB headroom** before extra game launches; old98% warnings below are historical, not the current measurement. No unrelated processes, pagefile settings, or original game files changed. New learner stderr `runs/ppo-damage-v4/stderr-20260913-171921.log`.

## CURRENT: localhost charts delivered; six-worker restart awaiting clarification

Final memory recheck worsened to **97.94% committed / 1.99GiB available**. Do not treat the earlier3.44GiB measurement as current expansion headroom.

Latest request supersedes the user's earlier three-worker preference: use SIX workers and HTML/localhost curves instead of a terminal. Dashboard is live at **http://127.0.0.1:8765/**, service **PID 6604**, module `isaac_rl.dashboard`. Both launchers now use it by default; separate `scripts/launch_dashboard.ps1` safely reuses an existing service for the same absolute run path. Loopback-only, read-only endpoints; no game port access, learner signals, or automatic learner starts. This history service intentionally stays alive after learner exit with an explicit DEAD trainer badge. Connection failure shows DISPLAY OFFLINE and unknown current learner health; paused chart refresh is explicitly labeled. TUI **8892** was identity-checked and closed after the chart service was verified. No live learner or game was stopped.

Learner remains **51700**, creation **1789318456.3176687**, CPU, three native workers **0/1/2**. Latest checked **1,169,082 steps**, all three established sockets owned by51700, empty learner stderr, approximately **10.98 rollout steps/s**. Dashboard returned **369 update records / 483 episodes**, earliest update at1,000,770 and latest1,168,962. It retains all available current-run JSONL history, not fabricated earlier metrics; raw/10/50-record means, range and axis controls, per-metric curves and separate PPO/collection times. Tests: **64 Python tests passed** under `py -3.10`; **7 synthetic chart checks** passed, including150,000 records andbillion-step counters. Default `python` is3.13 and lacks lupa; use3.10 for this project's full suite. In-app preview bootstrap failed at tool asset initialization; direct HTTP page/API/assets checks passed. No visual browser QA was claimed.

Six-worker expansion is NOT performed yet. The live process has a fixed collector pool and cannot load hot-added workers from source edits. Standing user instruction still forbids stopping live training. Ask whether one checkpointed restart is allowed to change to instances **0/1/2/4/5/6**; do not silently signal the learner or start a competing learner. Preserve evaluation instance3 (oldbridge0.1.2). First current check found **96.45% Windows commit**, **3.44GiB available**. Existing native games each use~0.45GiB private memory, but startup/system headroom fluctuates and previous97–99% commit caused failures. Recheck capacity before any authorized expansion; do not change pagefiles or kill unrelated apps. Historical three-worker and TUI instructions below are superseded by this section; learner never-stop and scratch-policy success criterion remain in force.

## CURRENT: user-approved three-worker resume at 12:54

The user explicitly chose **resume with three workers**. Learner **PID 51700**, creation time **1789318456.3176687**, is now detached in `runs/ppo-damage-v4`, session `session-1789318460011108300`, with **CPU**, ports **9999/10000/10001**, and **no step cap**. Monitor **27540** is bound to that exact learner identity. Initial native check verified all three trainer-owned established sockets, advancing steps from saved **1,054,530**, confirmed_v3 rewards and terrain_v2 observations, and empty new stderr. Never stop this live learner.

The final check caught a monitor-only startup bug: Windows rounded launch time to1789318456.317 while native telemetry used1789318456.3176687. Strict identity comparison rejected the same learner, so monitor48604 exited after30s. Matching now uses PID plus the existing0.01s creation-time tolerance, then canonicalizes the full timestamp. A regression covers the exact pair and still rejects PID reuse. Only the display was reopened as27540; learner51700 was never signalled or restarted. Latest suite: **61 passed**. Display stderr: `runs/ppo-damage-v4/monitor-20260913-1258.stderr.log`.

Reused native games: instance0 **6996**, instance1 **13120**, instance2 **34576**. Closed only unused IDLE training games4/5/6 (37128/36308/21256) normally before launch, after confirming no live trainer/port owner. Their private copies/saves remain. Reserved evaluation game3 **42656** remains idle and old bridge0.1.2. Windows commit at the first resumed check was **92%**, down from the capacity-test peak99.73%.

The resume uses the last SAVED checkpoint at 1,054,530 / 3,238 episodes, not the crashed session's uncommitted 1,054,626 / 3,239. Its final error status was preserved as `runs/benchmark-20260913/failed-trainer-status.json`. No completed logs were deleted; new records carry a session_id to distinguish resumed counters. The user's worker-count choice is resolved; all PENDING-relaunch notes below are historical.

Native production verification PASSED: by **1,055,346 observed steps / 3,241 episodes / 1,440 updates**, two new PPO updates had completed, model tensors were finite, all three sockets remained owned by51700, new learner stderr was empty, and the display was subsequently corrected as described above. Latest measured 384-transition rollout: collection **35.1747 s**, inference **0.4339 s**, reset **1.8005 s**, PPO update **0.15993 s**, save **0.01623 s**, total **37.6582 s**, rollout **10.197 steps/s**. Loss0.72090, no recent wins. Commit **93.04%**, headroom **6.733 GiB**. The real display rendered ALIVE, large step counts, uncapped target, phase timings and rolling reward/loss/throughput history in the terminal. These are handoff observations, not perpetual liveness proof; recheck processes and counters next time. No further work remains this turn; hand off PAUSE and leave training running.

## LATEST: throughput/display task at ~12:50 — supersedes old live-PID claims below

Previous learner **34280 has EXITED**, status error at 1,054,626 observed steps, last saved checkpoint **1,054,530** / 1,438 updates. Worker 2 native log had resource-decompression assertions/minidump; worker 0 later closed normally. The 20-seed baseline evaluator **25424 also finished**, **0/20 wins**, full report under `runs/review-20260913/evaluation-1m`. No learner was stopped this turn.

User now explicitly requested measured CPU/CUDA and worker-capacity benchmarking plus large-counter, rolling-history and process-lifecycle fixes. These are implemented; **60 tests passed**, with real native collection and real CPU update measurements. Report: `runs/benchmark-20260913/report.md`. CPU collection **18.3–20.0 steps/s** on six; **10.19 steps/s** on three. Per 768-transition rollout: 36.7–38.7 s game collection, CPU PPO update median **0.37775 s** over five identical warm measurements. CUDA failed three fresh-process full-update attempts with OOM; no valid GPU timing exists. Collection dominates; retain CPU.

Adding native instances 7/8 successfully created nine game processes under the existing local Steam session, but Windows committed memory hit **97–99.73%** and helper/CLR allocation failures. No usable eight-worker throughput was measured. Candidate ports10006/10007 were free. Both NEW IDLE test workers were normally closed; no additional workers should be launched under this resource pressure. Do not change pagefiles, close unrelated apps or bypass Steam licensing. Native copies/saves for7/8 remain for possible future use.

Production relaunch is PENDING the user's async worker-count choice: recommended three now vs six after memory relief. Do not claim it is running. Current six IDLE game PIDs: instance0 **6996**, 1 **13120**, 2 **34576**, 4 **37128**, 5 **36308**, 6 **21256**. Reserved eval game3 **42656** remains idle with old bridge0.1.2; do not use it for confirmed_v3 without updating it while idle. All benchmark sessions finished; do not re-poll old session IDs or relaunch captures into existing directories.

Stale display PIDs **44996,4508,30148 were closed** after their learner absence was verified. A new monitor against the real failed run displayed DEAD and exited successfully. New default monitors bind PID+creation time, exit with that session, and only follow replacements with explicit --follow. Launchers pass exact session identity. Both trainers use an OS-held run lease, bounded bridge cleanup, explicit terminal metadata and measured phase timings. The first updated native production update is still to be verified on relaunch.

This turn is PROGRESS (implemented/tested fixes and measurements changed capacity/device decision), not a blocked-goal declaration. Continue to honor never-stop-live-training and final PAUSE. Older active-run/PID/monitor assertions below are historical.

## Authoritative workflow (supersedes every historical instruction below)

The user manually pauses/resumes the agent goal, usually after 12–24 hours. On resume, review actual progress, explain decisions, continue the live detached run or set up the next run if needed, then output `PAUSE` on its own last line and STOP AGENT WORK. **Never stop a live training process.** Do not create stop.request, signal a learner, or close an in-use training game. The old `PAUSE_NOW`, automatic 100k checkpoint gates, intermediate-inspection restrictions, and 300k-per-worker freeze are superseded. No checkpoint watcher is currently armed. Do not relaunch old unified-exec sessions; those jobs are finished.

Goal remains unproven: consistent normal first-floor clears including boss. At the one-million-step review there were 43 training boss encounters and ZERO wins; frozen 100k and 200k evaluations each returned 0/20. The 90/100 unseen-seed gate with Wilson lower bound >= 0.8 remains the completion criterion. Never equate shaped reward, bridge tests, or a boss encounter with a win.

## Active detached parallel processes

- Run: `runs/ppo-damage-v4`; trainer PID **34280**, create time **1789310828.3609345**, monitor **4508**. Launch command uses `--steps 0` (uncapped), six native workers, confirmed_v3 rewards, terrain_v2 observations. Launcher default now points to this run. Resume only after independently verifying the trainer has exited; never start a duplicate.
- Training games: instance 0 **33968** / 9999, 1 **36708** / 10000, 2 **24408** / 10001, 4 **31108** / 10003, 5 **2432** / 10004, 6 **31220** / 10005. One shared PPO learner, six concurrent independent native game processes, 128 steps/game/update. All six confirmed bridge 0.1.3 / hp_delta_v1 through actual saved states and all six TCP connections belong to trainer 34280.
- Frozen baseline evaluation: evaluator **25424**, game instance 3 **42656**, port **10002**, output `runs/review-20260913/evaluation-1m`, stdout/stderr under the review directory. Requested 20 unseen seeds with full 3375-action/900-idle limits. It intentionally retains bridge **0.1.2** for the old one-million-step checkpoint; it is NOT a corrected-reward evaluation. Do not steal port 10002 or reload that game until the evaluation finishes naturally. Review its result/distribution on next manual resume. Upgrade only this idle reserved game before any future confirmed_v3 evaluation.
- Old monitors **44996** / **35896** point at the completed terrain run and were left untouched. Do not mistake their display for active training.

Latest verified audit: **1,004,610 aggregate steps / 1,373 updates**, six fresh PPO updates after the handoff, finite model tensors and loss **0.2911230413**, all six live trainer-owned sockets. Review artifact: `runs/review-20260913/review.json` (quantiles AND raw sorted samples). PIDs/counters here are historical evidence, not perpetual liveness proof; recheck exact processes/create times, sockets, advancing counters, stderr, and checkpoint finiteness on the next manual resume. No process may be restarted solely because an observation times out.

## This review's changes and provenance

The previous trainer PID 2016 had already exited naturally at **1,000,002 steps / 3,067 cumulative episodes / 1,367 updates** because of its configured one-million-step cap. Actual process absence, stopped status, empty stderr, and absent stop.request were checked. No live trainer was stopped in this review. Idle training games were normally reloaded to deploy the bridge; evaluation was not touched. Parallel training now defaults to no step cap; errors/user-requested shutdown still terminate normally.

Immutable parent `runs/review-20260913/checkpoint-1000002.pt` SHA256 **b782c555589357c366fb89a81f252aeb8a546a9d98749dfc03970c414cb538f2**; identical bytes copied into the new run's parent.pt. Model, optimizer, RNG, architecture, action duration, terrain observations, counters, and training seeds continued. Only the damage measurement is corrected; every numeric balanced_v2 reward weight remains identical in the new confirmed_v3 profile, with a fresh telemetry window.

Bridge 0.1.2 counted attempted hits that could be rejected by native invulnerability. Worst old episode earned ~1519 reward from 7761 reported damage with zero kills/clears. Native reproduction reported 320 damage against four enemies with only 40 total initial HP. Bridge 0.1.3 now observes HP across native updates; death settles any remaining observed HP once; disappearing entities alone earn no damage. Attempted hits remain diagnostic-only (`damage_attempted`). Reset requires `damage_signal=hp_delta_v1` for confirmed_v3, preventing silent mixed deployment.

Native regression `runs/review-20260913/damage-frame-observer/result.json`: PASSED, 20 actual versus 226.5 attempted damage, initial HP budget 40; positive control 8 damage and one kill. Earlier post-callback-baseline attempt FAILED and was not deployed to training; retained under `damage-after`, `damage-positive-control.json`, and `failed-postcallback-bridge.lua`. All scripted probes are unscored and never enter PPO or policy evaluation. Latest tests: **36 passed**. Distribution reporting now preserves quantiles/raw samples and clamps zero-win Wilson lower bounds to zero.

Original Steam installation, original saves, subscribed mods, and unrelated finance Python processes remain out of scope and untouched. C: free disk ~26.1 GiB at handoff. No more workers are needed now. Do not use debug stage launch, physics acceleration, or policy-side scripted navigation.

Read-only next-review commands:

```powershell
py -3.10 -m isaac_rl.monitor runs/ppo-damage-v4 --once
py -3.10 -m isaac_rl.analyze runs/ppo-damage-v4
Get-Content runs/ppo-damage-v4/stderr-*.log -Tail 20
Get-Content runs/review-20260913/evaluation-1m/result.json
Get-Content runs/review-20260913/evaluation.stderr.log -Tail 20
```

## Historical notes only — all conflicting workflow/PID/run hints below are superseded

# Working state — 2026-09-12 (archived inline)

## CURRENT USER CONSTRAINTS / AGENT PAUSE — supersedes older next-work notes

NEXT CHECKPOINT IS ALREADY ARMED: quiet gate unified-exec session **83844**, target **200,000 aggregate steps**, checkpoint **2**, 20 unseen-seed episodes on10002, output runs/checkpoint-002. Startup validated live learner2016 and its exact create time, without exposing/interpreting intermediate training progress. Do not launch a duplicate gate or evaluator. Receive this session's checkpoint event; do not poll training telemetry between milestones. Architecture/reward were not changed, training was not stopped, and checkpoint1 evaluation progress was not inspected in this arming turn.

At every checkpoint, output `PAUSE_NOW` on its own first line, then checkpoint number and CURRENT aggregate steps, and STOP AGENT WORK ONLY. Background training MUST CONTINUE. No intermediate progress checks unless fixing an actual bug. Evaluate at intervals of at least 100,000 aggregate steps using 20+ unseen-seed episodes; retain/report distributions, not only means. Detailed reporting interval is 500,000 aggregate steps. Architecture and reward are FROZEN: architecture1, balanced_v2 rewards, terrain_v2 inputs retained; no architecture/reward changes before evidence of 300,000 steps PER WORKER (aggregate counts are not per-worker counts).

Checkpoint **1** snapshot: **100,614 aggregate steps**, SHA256 350d705d2c9ee990ba81c5a76f5997c9695fd7d4fef7ec39760ccbfb3352d95e, runs/checkpoint-001/checkpoint.pt. The first version of scripts/checkpoint_pause.py clean-stopped training under the earlier wording 'stop all work'. The user immediately clarified agent-only pause. This was corrected: exact stop.request removed and same checkpoint/model/optimizer/RNG resumed, all SIX ports verified connected. Current learner **PID2016**, create time **1789246225.8261597** (2026-09-12 16:50:25.8261597-04:00), observed training at **101,046** aggregate steps / 232 episodes / 195 updates. Monitor44996 remains. Primary game is now **32048** (launcher found no live primary and started its normal private runtime); other training games remain24000/33176/25200/23928/41328, evaluation game42656. No architecture or reward changes. Future checkpoint_pause.py no longer writes stop.request or waits for trainer exit: it freezes exact latest.pt bytes while training runs.

Background checkpoint evaluation IS RUNNING: **20 episodes**, unseen seeds, full defaults3375 max actions /900 idle, evaluator **44828**, parent gate **36280**, unified-exec session **24592**. Output runs/checkpoint-001/evaluation; stdout/stderr separately logged under runs/checkpoint-001. Parent will save report.json with raw sorted samples, min/p05/p25/median/p75/p95/max, outcome counts and success statistics after completion. DO NOT cancel it, launch a competing evaluator on10002, or count the earlier3-episode diagnostics as satisfying the new constraint. The running parent retains the OLD initial-pause metadata in memory; its training_paused/manual_resume_required fields refer to the initial, since-corrected pause. See runs/checkpoint-001/pause-clarification.json for the current override. Future gates use corrected agent-only semantics. Session61720 (resume launcher) is CLOSED. Old watcher90522 was deliberately stopped before its obsolete3-episode diagnostic; it is CLOSED and diagnostic-terrain-100k was not run.

Do not continue agent monitoring/tuning after the PAUSE_NOW handoff. No claim of goal completion: most recently3 training boss encounters, zero wins. Episode201 at88,668 steps visited6 rooms, cleared3 combat rooms, killed11 enemies, then died in boss room5 with3 enemies remaining; raw evidence archived as runs/ppo-terrain-v3/boss-encounter-000201.json. All29 tests passed before the checkpoint-gate addition; gate distribution helper has not yet had a dedicated unit test. Earlier counts/process hints below are historical.

Goal ACTIVE and UNPROVEN: scratch-trained policy consistently clears floor 1 including the boss. Previous turn was a VERIFIED WAIT on live trainer 20596 and six native connections. Current turn made progress: training continued unchanged from 62,976 actions past 83,712, and a concrete evaluator cleanup defect was fixed and native-verified. CPU contention from unrelated work remains untouched. No blocker; zero floor wins.

## Current process and run evidence

Active run: runs/ppo-terrain-v3. Trainer PID **20596**, process_started **1789242778.9075906** (2026-09-12 15:52:58.9075906-04:00), monitor **44996**. Native training processes: primary **37732** (9999), worker 01 **24000** (10000), worker 02 **33176** (10001), worker 04 **25200** (10003), worker 05 **23928** (10004), worker 06 **41328** (10005). Evaluation worker 03 **42656** (10002) remains open and hidden. All six training connections were verified owned by trainer 20596. Original four games and monitor were left running; previous trainer 36088 checkpoint-stopped cleanly at 42,912 steps / 83 episodes / 119 updates before this handoff.

The corrected learner originally resumed at 25,089 steps / 40 episodes / 72 updates. Its 25,857-step checkpoint proved terrain_v2 observations, unchanged balanced_v2 rewards, finite model weights, and 337,058 parameter values changed through two fresh PPO updates. Death/reset cycles have completed since deployment. The prior reward-only run recorded one combat-room clear (episode 34, seed VALR GDNF, 43.5 damage / 4 kills / 3 rooms, then death). These are not first-floor clears or proof of consistency.

First boss encounter: episode **77**, at **39,897** total steps, seed **FXMN ZBQY**, port 10000. Recorded boss_seen=true, boss_defeated=false, success=false, 38.5 enemy damage, 3 rooms visited, then death after 370 actions. No boss win has been observed. Episode 78 cleared two combat rooms and then idled; positive reward does not imply floor success.

Final six-worker read: **54,432 steps / 114 cumulative episodes / 134 updates**, ~20.08 actions/sec. Fresh loss -0.004217 and entropy 4.80469 were finite. Trainer 20596/create time, monitor 44996 and all six established training connections were revalidated. Each worker has completed new episodes. Hardware after expansion was ~28% CPU, ~18 GB free RAM and ~4.5/8 GB GPU memory; C: free disk 29.49GB after all copies and diagnostic. stderr remained empty. Training remains active. Status counters are historical evidence only; read live state again before relying on them.

Previous monitoring turn ended at **62,304 steps / 134 cumulative episodes / 144 updates**, session-average ~17.56 actions/sec, loss 0.13949 and entropy 4.944 finite, mean recent reward -0.49, no wins. Same trainer PID/create time and monitor alive. From that turn's initial 56,304-step read, 16 episodes completed: 11 deaths, 5 idle truncations, 6 episodes with combat-room clears, 377.42 total enemy damage. Across its 94 terrain_v2 episodes there remained just ONE boss encounter and ZERO boss wins. Positive rewards/partial combat clears do not prove competence. stderr empty.

Read-only watcher unified-exec session **91316 exited 0 and is CLOSED** after nine samples over about six minutes, each checking the exact process handle and all six established trainer-owned connections. It recorded 57,024 -> 62,064 steps (+5,040), then the final monitor advanced to 62,304. No game, trainer or monitor was restarted, and no learning settings were changed. A separate immutable read of the 61,344-step / 143-update checkpoint verified all model tensors finite, all six ports and both profiles retained, and 373,050 parameter values changed since the six-worker handoff; SHA256 668dd29727959da14ad042a75e2f8cff5489ad0dad415ccc1ecc9cb52b9281ea (historical latest.pt bytes, not a separately archived file).

Throughput dropped because machine CPU reached **100%**, largely in unrelated finance-evaluation Python workers under parent 36004 (and separate 42808), identified from process metadata only. These are NOT Isaac workers and were not modified or stopped. GPU remained ~11%, 4550/8192MiB, and ~16.6GiB RAM free. Do not add game copies under this load or kill unrelated work. The learner still advances; CPU contention is not a terminal condition or a reason to restart.

The queued three-episode frozen diagnostic completed successfully as an experiment (NOT as a floor-clear goal), output runs/diagnostic-terrain-50k. Watcher/evaluator session **8171 exited 0 and is CLOSED**, evaluator 43784 exited; port 10002 is available again. Previous watcher session 13252 exited as expected when the three-worker trainer was checkpoint-stopped, and is closed. No pending unified-exec session remains. Neither a short diagnostic nor the first boss encounter meets the completion gate.

CURRENT PENDING SESSION: watcher **90522** holds exact trainer 20596/create time, samples live ports/counters every 45 seconds, and will run a frozen diagnostic at >=100,000 steps on port 10002, output runs/diagnostic-terrain-100k. Three episodes, max450 / idle180, same limits as the 50k diagnostic. Last observed 83,712 steps / 188 cumulative episodes / 172 updates, all six ports connected, zero recent wins. Poll this SAME session; never launch a competing evaluator. Evaluation cleanup probes are finished and port 10002 is free. The older 'no pending session' statements above refer to completed earlier turns.

Evaluator cleanup defect: after the last 50k diagnostic episode died and disconnected, native worker03 reached the game-over screen and MC_POST_UPDATE stopped reconnecting. A 10-second connection probe timed out; process 42656 stayed alive. Screenshot runs/evaluation-idle-100k-visible.png plus native Game Over log confirmed actual UI state. Sent Space ONLY to worker03's window to restart normally; no process was killed/restarted, no training game changed, then hid the window again. evaluate.py now does a separately logged UNscored normal reset AFTER all requested results are persisted, then closes. cleanup.json records previous/fresh states and failure independently; it cannot alter episode counts/results. Exceptions/timed-out scored transitions do not enter cleanup, and reset timeouts are not retried. All **29 tests passed**.

Native regression evidence: runs/evaluation-cleanup-native-death-fine passed. Ordinary scripted 1-frame movement into an existing native fire caused six half-hearts of damage/death after 198 actions; cleanup reset episode18 -> 19 to normal Isaac/stage1/difficulty0/alive, and a second TCP connection to the SAME game42656 succeeded. Every probe artifact is explicitly unscored/operational; no controller actions feed the learner/evaluator. Earlier retained probes runs/evaluation-cleanup-regression, -native-death and -native-death-450 did not reach death before their limits, so their strict death-case result remains failed even though cleanup/reconnection succeeded. Fine-grained scripted movement avoided the 8-frame fire-orbit problem. A replay of seed8D4H6JQ2 with the same frozen policy had identical initial encoded inputs but native collision outcomes diverged at action11 (same chosen action), ending idle instead of death; do NOT claim universal seeded combat determinism from the earlier empty-room movement probes. No physics, native Lua or training settings were changed.

PID hints/status files are NOT continuing proof. Inspect actual executable paths, connections and advancing counters. Trainer is detached, not a unified-exec session. If observation times out, inspect the same process; never restart solely for an observation timeout.

Useful read-only commands:

    Get-Content runs/ppo-terrain-v3/status.json
    Get-ChildItem runs/ppo-terrain-v3/*stderr* | Sort-Object LastWriteTime | Select-Object -Last 1 | Get-Content -Tail 20
    Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -in @(20596,44996,37732,24000,33176,42656,25200,23928,41328) } | Select-Object ProcessId,ExecutablePath,CommandLine
    Get-NetTCPConnection -State Established | Where-Object { $_.LocalPort -in @(9999,10000,10001,10002,10003,10004,10005) }
    py -3.10 -m isaac_rl.monitor runs/ppo-terrain-v3 --once
    py -3.10 -m isaac_rl.analyze runs/ppo-terrain-v3

## Provenance and evidence

- runs/ppo-floor1: discarded debug-launch integration run, 623 steps. Never used its weights for normal training.
- runs/ppo-normal-v1: random initialization seed 20260912, stopped at 3,255 steps / 9 episodes / 14 updates. Its stop.request remains intentionally.
- runs/ppo-parallel-v1: continued through immutable parent.pt; stopped cleanly this turn at **17,148 steps / 21 episodes / 51 updates**. Trainer 31128 exited after checkpoint, and obsolete monitor 37196 was closed only after verifying its exact monitor command line. Its stop.request remains intentionally.
- runs/ppo-parallel-v1/before-kill-counter-fix.pt: 8,184 steps / 15 episodes / 27 updates, before bridge 0.1.2. IsActiveEnemy(false) in MC_POST_NPC_DEATH excluded dead enemies; changed to true, excluding friendly/background NPCs. Episode 16 verified 3 real kills / 15 enemy damage. Damage rewards worked before the fix; old kill data were missing/zero.
- runs/diagnostic-policy-v1: **completed frozen-policy diagnostic**, checkpoint at 13,560 steps, SHA256 94816ca1a519c08ff9545435bdc6e79ebe2dce3be699928b0746e9096650d5ae. Three unseen normal Isaac seeds, deterministic policy, 450-action max / 180-action idle limit. All three idle, no enemy damage, no boss. Seeds 344R 6MAC (1 room, reward -0.298), M6CM PBSR (2 rooms, +6.548), QXDV 1KMY (2 rooms, +6.658). The latter two earned positive reward from empty-room clears. Trace showed repeated wall-bound movement. Full action/state traces, exact frozen checkpoint and initial/final states are retained. This is diagnostic-only, not a 100-run consistency evaluation.
- runs/ppo-balanced-v2/parent.pt: immutable 17,148-step parent, SHA256 c3bd1dd8c4b6f4f5544d3c61a5dd88f07c4fc0f967cc14b36b74ae44e287827b. Preserved model/optimizer/RNG/counters/training seeds; cleared only the old reward telemetry window because rewards changed.
- runs/ppo-balanced-v2 stopped cleanly at 25,089 steps / 40 episodes / 72 updates. Its stop.request remains intentionally. Across its 19 completed episodes: mean reward -5.283, one confirmed combat-room clear, zero bosses.
- runs/ppo-terrain-v3/parent.pt: immutable handoff checkpoint, SHA256 49d959b8bdd7300b6e5152f2f69a487b8fbea325a8b22ba02d1b59c8613202e4. Preserved model/optimizer/RNG/cumulative counts/seeds. Reward remains balanced_v2; observation changes from legacy_v1 to terrain_v2; telemetry window cleared. Original manifest session-1789241095438781200.json records both profiles and source hashes. Initial states use bridge 0.1.2 and normal Isaac.
- runs/ppo-terrain-v3/before-six-workers.pt: immutable 42,912-step / 83-episode / 119-update checkpoint, SHA256 b11c8dd8a533b8867fd26a97e1874a0c1fa4f7cba178f36df2b38cd1b3c72ef6. Six-worker session-1789242782208914600.json and its initial-states file record the resumed six-port setup. Model/optimizer/RNG/counters/seeds and both profiles were preserved. Only this active run's exact stop.request was removed for resume.
- **26 tests passed** after adding six-environment cases to concurrent stepping/reset isolation and PPO weight-update tests. Includes installed native-enum contract, profile compatibility, env routing and reward tests. Both trainer CLIs previously passed --help smoke checks. All SEVEN native games passed seeded timing/pause/replay probes (runs/probe, probe-10000 through probe-10005). New worker 04/05/06 logs showed only the RL mod plus native Lua, without Lua errors.
- runs/diagnostic-terrain-50k: completed frozen **50,592-step** checkpoint, SHA256 9754a8a6a44b7d37bdca3da7fc5becf7db9972163ff034a70e443f72f602c96f. Deterministic, terrain_v2 / balanced_v2, 450 maximum actions / 180 idle, **0/3 floor wins**. Seed 64Q6 W3RK: 180 actions, idle, 1 room, no damage. Seed 2B30 BFB9: 249 actions, **one combat-room clear, 6 kills, 30 enemy damage, 1 half-heart damage taken**, then idle, 2 rooms. Seed 8D4H 6JQ2: 57 actions, death, 2 rooms, no damage dealt. No bosses. All three seeds were separately rechecked as absent from the frozen checkpoint's training seeds; all weights finite. Exact checkpoint, 1.44MB action/state trace and initial/final states retained. Episode 2 raw final state confirms normal stage 1, ordinary cleared room, zero enemies, no boss flags. Earlier 13,560-step diagnostic dealt no damage on any of its three seeds, but these tiny different-seed samples cannot establish a reliable improvement or attribute it to parallelism.
- runs/terrain-probe-1789241046763727700: real separate-game diagnostic on port 10002, seed 36MM BX9Y. Four open doors all had native collision class 5, were falsely solid in legacy encoding, and correctly passable in terrain_v2. Script crossed selected bottom door in 11 one-frame actions. Trace and raw initial/final states retained. This is scripted diagnostic code, NEVER policy training/evaluation; no model or optimizer receives these actions.

## Current learning implementation

Observation profile terrain_v2: player-solid collision classes are 2/3/4, not 5 (COLLISION_WALL_EXCEPT_PLAYER). Potential terrain hazards are grid kinds 8/9/12 (SPIKES/SPIKES_ONOFF/TNT), not 14/15 (POOP/WALL). Verified against runtime/game/resources/scripts/enums.lua and native door traversal. Spatial/vector dimensions and all game physics/actions/rewards are unchanged. Existing checkpoints missing observation_profile explicitly resolve to legacy_v1, preserving frozen evaluation behavior; new training defaults to terrain_v2. Both trainers and evaluator propagate profile to IsaacEnv on reset and step. Explicit changes require a new run directory. Single-game trainer also now persists/inherits reward profile rather than accidentally reverting a parallel checkpoint to legacy rewards.

rewards.py defines legacy_v1 (archived) and balanced_v2 (current). v2: action -0.01; first cell +0.01; first room +2; first clear of a room with previously observed enemies +10; enemy damage +0.2/HP capped 200 per transition; enemy death +0.5; player damage -0.5/half-heart; death -5; boss clear +100; collectibles +2 and heal +0.3/half-heart unchanged.

v2 adds bounded visible-door potential Phi in [0,2]. Navigation reward is GAMMA * Phi(next) - Phi(before), with GAMMA shared with PPO (0.995). Phi is zero in combat and at true terminal states, otherwise proximity to least-visited open/unlocked visible destinations. It never chooses actions. Potential accounting spans room transitions; tests verify discounted round trips cannot farm reward and terminal potential is zero. Combat/death reweighting is a separate experimental change, not a claim of policy invariance. Normal game damage, health and physics are unchanged.

New cells no longer reset v2's 450-action idle clock; actual room visits/clears/damage/kills/pickups do. Empty-room clear reward is gone; ordinary room exploration remains rewarded. Environment tracks combat_clears separately from raw clears. Source: env.py and rewards.py. Reward definitions are in manifests and checkpoints. Explicit profile changes require a new run directory; resumes inherit the checkpoint profile.

train_vector.py currently uses SIX concurrent native environments, 128 steps/game (768 transitions/update), batch128, four PPO epochs. ThreadPoolExecutor overlaps native game requests; one shared actor-critic batches inference and learns from all six games, not six independent learners. Per-game GAE computed before flattening, terminal values before reset; independent futures collected completely before raising. Default 1M total steps, CPU two Torch threads. Atomic checkpoints after updates; bounded live-states.json snapshots support audits without a second client stealing a port. Models still choose movement x shooting x utility through MultiDiscrete([9,5,4]); no expert policy, demonstrations, aiming or pathfinding controller.

analyze.py separates reward profiles instead of averaging incomparable rewards; missing old kills/combat-clear values are not invented as zero. Two reporting regression tests were added after deployment; the running trainer need not restart for those offline edits. Monitor now shows reward profile plus kills and combat clears.

Evaluation hardening: --port supports the separate worker; load_snapshot hashes, deserializes and archives the SAME checkpoint bytes despite concurrent latest.pt replacement; seed uniqueness/exclusions normalize case/spacing; manifests label short diagnostics and record limits. Default normal evaluation still uses 3375 maximum actions / 900 idle. It honors the checkpoint reward profile, but completion depends on actual boss state, not reward.

## Next work

Let the six-worker learner continue, next useful frozen comparison around 100,000 total steps. The 50,592-step diagnostic is fully collected and audited; do not rerun/overwrite it or poll the closed session. Assess navigation/combat/death statistics cautiously; three diagnostic seeds are insufficient to establish improvement or consistency. Prefer a verified wait on the actual live trainer over inventing another hyperparameter change. Fix concrete evidenced defects, but do not keep resetting experiments just because early performance is poor. Need actual competent learning, not further infrastructure for its own sake.

The evaluator command is:

    py -3.10 -m isaac_rl.evaluate runs/ppo-terrain-v3/latest.pt --port 10002 --episodes 100

A shorter diagnostic can use --episodes 3 --max-episode-steps 450 --idle-limit 180, but cannot prove the objective. Evaluation freezes the policy, excludes checkpoint training seeds, logs all attempts and checks initial/final normal first-floor boss states. Concrete completion gate: >=90 wins /100 unique unseen seeds, Wilson 95% lower bound >=0.8. No full consistency evaluation has been run; no boss win has been observed. Goal is not complete.

## Safe control

To checkpoint-stop current training, use apply_patch to create runs/ppo-terrain-v3/stop.request, then wait for the actual trainer process to exit. It finishes the current step batch, optimizes partial rollout and saves. Only remove that exact control file when ready to resume:

    .\scripts\launch_parallel.ps1 -Resume -NoMonitor

The launcher defaults to runs/ppo-terrain-v3. Resume without -Games/-Instances inherits prior config ports, retaining all six training games. Explicit six-game selection is -Instances 0,1,2,4,5,6; this reserves instance 3 for evaluation. Fresh launch without a selection still defaults to three. -Games N selects consecutive IDs; it cannot be combined with -Instances. Keep the existing monitor. Launcher leaves live game processes alone. Do not clear archived run stop.requests accidentally. For intentional profile changes, use a new -Run, -Initialize pointing to a stopped/frozen checkpoint, and explicit -RewardProfile and/or -ObservationProfile.

## Runtime isolation and native-game constraints

Original Steam game: C:\Program Files (x86)\Steam\steamapps\common\The Binding of Isaac Rebirth. Original executable hash unchanged when rechecked: 04469d0c3d3581936fcf85bea5f9f4f3a65b2ccf96b36310456c9626bac36dc6. Original saves/subscriptions/mods are untouched.

| Instance | Runtime | Port | Save folder under C:\Users\yanbo\Documents\My Games |
| --- | --- | --- | --- |
| 0 | runtime/game | 9999 | Isaac RL Training__________ |
| 1 | runtime/worker-01/game | 10000 | Isaac RL Worker 01_________ |
| 2 | runtime/worker-02/game | 10001 | Isaac RL Worker 02_________ |
| 3 | runtime/worker-03/game | 10002 | Isaac RL Worker 03_________ |
| 4 | runtime/worker-04/game | 10003 | Isaac RL Worker 04_________ |
| 5 | runtime/worker-05/game | 10004 | Isaac RL Worker 05_________ |
| 6 | runtime/worker-06/game | 10005 | Isaac RL Worker 06_________ |

Each folder's log.txt is authoritative for Lua/errors. Installation manifests record hashes/paths. Preparation modifies exactly one same-length save-directory literal in a private executable copy, not gameplay or licensing. Private SteamCloud=0 isolates saves.

- Installed executable is Repentance 1.7.9b, despite separate Repentance+ saves.
- NEVER --set-stage=1: debug unlocks/D6 and ignores requested seeds. Launch only --luadebug and normal menus.
- Keep disable.it markers in private subscribed-mod copies; deleting dirs causes Steam to recreate enabled mods. All seven logs showed only the RL mod's Lua plus native scripts. Recheck after preparing new workers/subscriptions.
- Game:Update() acceleration changed movement per frame and was removed. Do not restore without fidelity proof.
- Hidden workers simulate normally and pass probes. Startup briefly shows windows without activation; extras hide afterwards. Primary visible. No need to minimize.
- Preparation derives actual save parent from original savedatapath.txt. Tool Documents lookup differed; an unused initial C:\Documents\My Games\Isaac RL Training__________\options.ini remains, not active.
- Ryzen 5600X (6 cores / 12 threads), 48GB RAM, RTX3060Ti 8GB. Seven-game read after expansion: CPU ~28%, ~18GB free RAM, GPU ~4.5GB / 11% utilization. Inspect disk before adding copies or unbounded traces.
- Never kill unrelated Python processes or alter original game data. No AGENTS.md found; worktree files are untracked, no commits made. No subagents used/permitted.
