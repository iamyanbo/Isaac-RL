# Current handoff — 2026-09-14

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
