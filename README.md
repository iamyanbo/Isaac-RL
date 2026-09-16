# Isaac RL

A randomly initialized PPO agent learning in the actual Windows game. The Lua mod streams structured state over localhost TCP and applies the agent's movement, shooting, bomb, and item inputs. The game waits at each action boundary while Python performs inference or optimization.

The project is **not yet at the requested success criterion**. Training must demonstrate consistent first-floor boss clears on unseen seeds. Passing the bridge or unit tests does not establish that the agent can play well.

## Run

### Current training — enriched history, September 16

**Live: `runs/ppo-observability-v8`, learner30608, six workers, CPU, four ticks/action.**
Overnight review:5.80M saved decisions, zero recorded wins in2681 treatment
episodes. Training remains unchanged. A frozen parent/treatment comparison is
running separately on reserved instance3:100 held-out seeds each, native evidence,
no optimizer. The5,214,636 review checkpoint was not retained by latest-only
checkpointing; the frozen5,797,932 treatment is583,296 decisions late. Results
will be explicitly exploratory, not a fixed-budget falsification verdict.
Artifacts: `runs/observability-review-20260916`; process IDs in OPERATIONS.md.

`combat_history_v3` adds native combat fields and four snapshots with intervening
actions. Input widening preserves the parent's initial policy and Adam state.
The approved handover is complete, using the **final4,014,636-step checkpoint**,
preserved byte-for-byte. The first real PPO update completed at4,016,172; finite
changed weights and gradients into new inputs are verified. All other learning
settings remain unchanged. **156 tests and native smoke checks pass**. Bridge0.1.6
exports actual bomb age, not an unavailable fuse countdown, and stable entity
lifetime IDs: InitSeed is not unique. After an identity validation crash, the
manually resumed goal recovered the saved **4,406,316-step checkpoint** with exact
model/Adam/RNG inheritance. First new optimizer update4760 reached4,407,852;
input shapes and all learning settings are unchanged. Crash evidence and the
byte-identical checkpoint are in `runs/identity-recovery-20260915-2303`. See the
[observability experiment](EXPERIMENT-OBSERVABILITY-20260915.md) for the exact
change, unchanged settings, evidence and predeclared falsification condition.

**Charts: http://127.0.0.1:8769/**. Training is uncapped and must not be stopped
without new approval. No competence
claim follows from passing tests or loss curves. OPERATIONS.md has current IDs.

### Historical baseline — completion rewards, September 15

**Stopped after authorized observability handover: `runs/ppo-completion-v7`.**
`completion_v4` removes damage/kill bonuses, retaining combat-room and boss-clear
rewards and all other settings. Baseline3,419,340 is preserved byte-for-byte and
used to initialize the new run; the old run's final5,377,722 checkpoint is archived
separately. Authorized handover complete; learner46824's first real update verified
at3,420,876 with finite changed weights. **113 tests passed.** No new evaluation
launched; improved boss competence remains unproven. Do not stop this learner.

**Baseline charts: http://127.0.0.1:8768/**. Read-only history service:
`.\scripts\launch_dashboard.ps1 -Run runs/ppo-completion-v7 -Port 8768`.
See
[the completion-reward experiment](EXPERIMENT-COMPLETION-20260914.md).

September15 recovery: a native allocation failure at99.6% Windows commit caused
the first session to exit; a later machine reboot removed the game processes.
The unchanged six-worker run is now resumed as **learner31448** from its saved
3,897,036 checkpoint, with exact model/Adam/RNG inheritance verified and real PPO
updates completed. Current dashboard30664 uses the same8768 URL. Failure evidence
and recovery snapshots are preserved in `runs/recovery-20260915-1747`; see
OPERATIONS.md for current identities. No live learner was stopped during recovery.

### Historical baseline — four-tick control, September 14

**Stopped after authorized handover: `runs/ppo-control-v6`.**
Six workers, CPU, four native ticks/action (133 ms).
The user authorized this timing intervention after the native trace audit.
Discount/GAE decay, time cost, episode/idle budgets, rollout/minibatch sizes and
visit-map intensity were converted to preserve physical-time meaning. Architecture,
action heads, combat reward terms, native physics and entropy0.002 remain unchanged.
See [the control experiment](EXPERIMENT-CONTROL-20260914.md) for exact conversions.

Baseline **2,470,092 steps** preserved at `runs/control-intervention-20260914/baseline.pt`.
The old learner exited cleanly; new learner20984 reused all six native games. First
full PPO update verified at2,471,628,38.21 aggregate decisions/s. Optimized weights,
checkpoint inheritance and four-tick worker advancement verified;108 tests passed.
Training success is still unproven. Do not stop/relaunch a live learner.

**Baseline charts: http://127.0.0.1:8767/**. Reopen only the history service with
`.\scripts\launch_dashboard.ps1 -Run runs/ppo-control-v6 -Port 8767`.
Plain resume inherits `physical_v1`, four frames, and converted schedules from the
checkpoint; no timing change is permitted inside an existing run. Old8766/8765
dashboards remain baseline history. See OPERATIONS.md for exact process identities.

Working rule: **hyperparameter tuning in general is not the default first
intervention**. Prioritize evidence-backed mechanism fixes; see [AGENTS.md](AGENTS.md).

### Historical entropy intervention — September 14

Selected and implemented one change: **PPO entropy coefficient0.02 →0.002**. All six-worker training settings, rewards, observations, architecture, optimizer, and eight-update actions stay unchanged. The baseline at2,417,208 steps is preserved in `runs/entropy-intervention-20260914/baseline.pt`. See [the training experiment](EXPERIMENT-20260914.md).

**Current live run: `runs/ppo-entropy-v5`, six workers, CPU, entropy coefficient0.002.** The user authorized one checkpointed handover: the old learner exited cleanly at2,425,314 steps and its final state was archived. New learner21432 starts from the preserved2,417,208 baseline; its first full update at2,417,976 was verified with finite changed weights and all six connections. No extra native processes or training-factor changes were introduced. The never-stop-live-training rule now applies to this new learner.

**Current charts: http://127.0.0.1:8766/**. The previous dashboard on8765 remains available for baseline history and correctly reports its learner dead. Older run descriptions below are historical. Reopen the new history service with `.\scripts\launch_dashboard.ps1 -Run runs/ppo-entropy-v5 -Port 8766`; do not relaunch a live learner.

The trainers and launchers now accept `--entropy-coef` / `-EntropyCoef`. Omission inherits the checkpoint value (legacy checkpoints default0.02); saved checkpoints and update/status/config telemetry record the coefficient. Changing it requires a new run directory.

### Frozen behavior comparison

`isaac_rl.behavior_eval` evaluates one immutable checkpoint on20 unseen normal seeds in three conditions: deterministic, stochastic, and the same stochastic policy with movement disabled only during combat. It records full native states plus actions, per-head entropy/probabilities, displacement, event deltas and reward components. Paired comparison requires matching observation prefixes through the first combat state. Unmatched prefixes and seeds that never reach combat are excluded from causal movement comparisons. This is a diagnostic, not the100-seed success gate.

```powershell
# Only use a free reserved evaluation port with a current bridge; never a training port.
py -3.10 -m isaac_rl.behavior_eval runs/ppo-damage-v4/latest.pt --output runs/new-behavior-evaluation --port 10002 --seeds 20
```

Completed movement diagnostic: `runs/behavior-20260913-v2`, frozen at **1,630,008 steps**, all 60 episodes finished with zero wins. Only 7/20 stochastic movement-ablation pairs matched through combat entry. See [the follow-up audit](AUDIT-20260914.md); movement and damage counts are not proof of aim or dodging.

Current detached diagnostic: **`runs/baseline-20260914`**, frozen at **2,372,664 steps**. It compares learned stochastic versus uniform-random actions on 20 unseen seeds with three repetitions each (120 episodes). Repetitions share a seed and are not independent statistical units. Uniform actions use the same action space and frame hold, without a heuristic controller. No production learner settings changed.

```powershell
# New runs only, after verifying that the reserved evaluation port is free:
py -3.10 -m isaac_rl.behavior_eval runs/ppo-damage-v4/latest.pt --output runs/new-random-comparison --port 10002 --seeds 20 --repetitions 3 --arms stochastic uniform_random
# Offline analysis of a finished diagnostic; output must not already exist:
py -3.10 -m isaac_rl.behavior_report runs/behavior-20260913-v2 --output runs/new-review.json --training-run runs/ppo-damage-v4
```

`status.json` identifies the evaluator and phase; `result.json` updates after each completed episode; `transitions.jsonl` and `trace.jsonl` preserve behavior evidence. The offline report verifies transition-derived episode summaries, records evidence hashes, and distinguishes command, displacement, and health-event metrics. The first attempt under `runs/behavior-20260913` failed before scoring because native seeded commands require a middle space; all failed and successful evidence remains preserved.

### Local chart dashboard — September 13

Open **http://127.0.0.1:8765/** for full recorded history: total/policy/value loss, reward, entropy, KL, explained variance, gradient norm, collection throughput, separate collection/PPO timing, committed memory, episode length, combat, damage and success curves. Choose raw/10/50-record smoothing, all/100k/1M-step ranges, and step/time axes. Hover for exact values. There is no one-million-step cutoff; missing historical metrics are not fabricated. Session changes and resumed counter rollbacks break curves and reset smoothing.

Both launchers now start this lightweight, read-only localhost service instead of a TUI. `-NoMonitor` suppresses it; `-DashboardPort` changes its port. Reopen independently with `.\scripts\launch_dashboard.ps1 -Run runs/ppo-damage-v4`; it reuses the existing dashboard for the same run. It never binds game ports or controls training. The service intentionally remains available for historical analysis after learner exit, explicitly showing **DEAD**; server connection loss shows **DISPLAY OFFLINE**, not a claim that the learner died. Closing the page does not stop either process. The terminal monitor below remains an optional diagnostic, not the default display.

The user explicitly authorized **six workers now** after the restart requirement was explained. The three-worker learner saved and exited cleanly at **1,220,664 steps**, then the same scratch-trained policy resumed on CPU with instances **0/1/2/4/5/6**, preserving evaluation instance3. Memory pressure had fallen to **82% committed / 16.5GiB available** at the restart. This was one authorized checkpointed transition, not permission for automatic restarts. No unrelated applications or memory settings were changed. The dashboard follows the run's explicitly identified new learner session.

### Throughput/display update — September 13

Measured six-worker collection runs achieved **18.3–20.0 steps/s**. Collection alone took **36.7–38.7 s per 768 transitions**, while five identical CPU PPO updates took **0.378 s median**. CUDA failed full-update attempts with out-of-memory errors, so no GPU speed claim is made. CPU remains the selected device because collection dominates. The eight-worker capacity test hit Windows allocation errors with committed memory at 97–99.7%; free physical RAM was misleading. No worker expansion is enabled. See `runs/benchmark-20260913/report.md` for all results and limitations.

The previous learner had already exited on a native game connection reset; it is not still alive merely because a display or game window exists. The user chose the measured three-worker fallback, resumed on CPU from **1,054,530 saved steps** at 12:54 on September 13. The new run continues detached without a step cap, with the updated session-bound display and timing logs. Unused idle training games were closed; no live learner was stopped.

The display now fits normal terminal sizes, supports large counters, and shows a rolling reward/loss/throughput history from file logs. `--window 100` changes the window. A monitor binds to one process identity and **exits when that learner exits**; use `--follow` only if you explicitly want it to watch future runs. Closing the display never signals the learner. `ALIVE` means a matching live PID and fresh heartbeat, `IDLE` means waiting/connecting, `STALE` means a live PID with old telemetry, `EXITING` means a live process has published a terminal outcome, and `DEAD` requires actual process absence or identity mismatch. `UNKNOWN` is used when liveness cannot be verified.

Each update now logs `timing.collection_s`, `inference_s`, `reset_s`, `update_s`, `save_s`, `wall_s`, `rollout_sps`, and Windows `commit_memory`. An OS-held `trainer.lock` enforces one learner per run; the file can remain after exit, but the OS lock cannot. Bridge cleanup has a one-second timeout. Idle native game workers remain reusable; they are distinct from a stopped learner or exited monitor.

```powershell
# Read-only view, including history and actual process liveness:
py -3.10 -m isaac_rl.monitor runs/ppo-damage-v4 --once --window 50
# CPU/CUDA comparison from saved data; does not touch any game ports:
py -3.10 -m isaac_rl.benchmark runs/benchmark-20260913-six/checkpoint.pt --ports 9999 10000 10001 10003 10004 10005 --from-capture runs/benchmark-20260913-six --output runs/new-device-benchmark --repeats 5
# Native benchmark ONLY when the selected ports are unowned and games are idle:
py -3.10 -m isaac_rl.benchmark runs/ppo-damage-v4/latest.pt --ports 9999 10000 10001 --output runs/new-native-benchmark --collection-only
```

Benchmark updates are disposable, never production weights. Run CUDA/extra-worker experiments only after restoring sufficient Windows commit headroom, not merely seeing free RAM. `launch_parallel.ps1 -Device cpu` makes the selected production device explicit (`cuda` is supported for a future verified trial).

This workspace has Python 3.10, PyTorch, and a Steam installation of Isaac Repentance 1.7.9b. A Steam-owned copy of Repentance or a supported later version with Lua mod support is required.

```powershell
py -3.10 -m pip install -e '.[dev]'
# Only if this run's trainer has exited; never start a duplicate learner:
.\scripts\launch_parallel.ps1 -Run runs/ppo-damage-v4 -Resume
```

The active setup uses six separately saved normal-speed game processes (instances 0/1/2/4/5/6), one shared CPU PPO learner, and the localhost chart dashboard. Instance3 is kept separate for frozen-policy evaluation. The first verified six-worker update after the authorized restart reached1,221,432 steps at19.29 steps/s; collection took37.25s and PPO optimization0.362s. The launcher enters normal Isaac runs through the menus; extra game windows are hidden after startup and continue simulating normally. It disables subscribed mods in the private copies. The original game, subscriptions, and personal saves remain outside the training installations.

`-Instances` selects explicit game IDs, which lets training skip the evaluation worker. Alternatively, `-Games N` uses consecutive IDs starting at zero; do not combine the two options. A fresh launch without either option defaults to three games. `-Resume` without either option restores the run's recorded worker set, currently instances0/1/2/4/5/6; it does not silently change worker count.

The private executable changes exactly one same-length save-directory string; it does not change gameplay code or Steam licensing. The original and private SHA-256 hashes and save directories are recorded in `runtime/installation.json`. `SteamCloud=0` is set in the private save directory. Steam may copy subscribed mods into the private installation; `prepare_runtime.py` creates `disable.it` markers in those copies. Check the game log after a new installation: only the RL mod's Lua should execute. If Steam adds a newly subscribed mod, close the private game and prepare again before training.

Do **not** use `--set-stage=1`. That launch option creates a debug run with unlocks and, on the installed version, causes seeded resets to produce a different seed. Normal menu startup is automated instead.

The current training run is `runs/ppo-damage-v4`. It continues the scratch-trained learner from **1,000,002 aggregate steps**, preserving model, optimizer, RNG state, and training seeds. The previous run, `runs/ppo-terrain-v3`, naturally reached its old one-million-step cap. Parallel training now defaults to `-Steps 0` / `--steps 0`, meaning **no automatic step cap**. It remains detached while the agent/user goal is paused. Process or bridge errors can still end a run and must be diagnosed, not silently restarted.

The new `confirmed_v3` reward profile retains every `balanced_v2` numeric coefficient but requires bridge 0.1.3's observed enemy-HP-loss signal. Previously, attempted hits on invulnerable enemies inflated damage rewards. Corrected measurements start a separate telemetry window; rewards before and after this correction are not directly comparable. Terrain observations, model architecture, actions, and normal game physics are unchanged. Immutable parent and review evidence are under `runs/review-20260913`; the new run also retains `parent.pt`.

Earlier normal training runs are `ppo-normal-v1`, `ppo-parallel-v1`, `ppo-balanced-v2`, and `ppo-terrain-v3`. There are no demonstrations or pretrained policies. `runs/ppo-floor1` is a discarded debug-launch integration experiment; its weights are not used in normal training. At the one-million-step review, the terrain run had 43 boss encounters and **zero wins**; the completed 100k and 200k frozen evaluations each had **0/20 wins**. This is not a competent first-floor agent yet.

To inspect or reopen telemetry:

```powershell
py -3.10 -m isaac_rl.monitor runs/ppo-damage-v4
py -3.10 -m isaac_rl.monitor runs/ppo-damage-v4 --once
py -3.10 -m isaac_rl.analyze runs/ppo-damage-v4
```

The user manually pauses/resumes the agent goal, usually after 12–24 hours. On resume, review actual processes, progress, and evaluation evidence, make a bounded decision, then finish with `PAUSE` on its own line. **The agent must never stop live training.** Old checkpoint-triggered agent gates and `PAUSE_NOW` notes are superseded. Pausing the agent goal does not pause the detached learner.

For user-operated shutdown only, `stop.request` or Ctrl+C requests a checkpointed exit. Do not relaunch until the actual trainer has exited; remove the user's stop request before resuming:

```powershell
.\scripts\launch_parallel.ps1 -Run runs/ppo-damage-v4 -Resume -NoMonitor
```

The trainers also run directly when the corresponding game instances are open:

```powershell
py -3.10 -m isaac_rl.train --run runs/another-run --steps 1000000
py -3.10 -m isaac_rl.train_vector --run runs/ppo-damage-v4 --ports 9999 10000 10001 --resume runs/ppo-damage-v4/latest.pt --steps 0 --device cpu
```

Focus loss and hidden windows are supported; minimizing is not needed. Port is `9999 + instance ID`: current training uses 9999, 10000, 10001; evaluation reserves 10002. Previously used training ports 10003–10005 are currently idle. One trainer/evaluator may own each port at a time. Launchers check for existing listeners and leave live processes in place. The bridge restores manual inputs on disconnect. A lost TCP connection is an error, not an invented episode result.

## Architecture

| Component | Implementation |
| --- | --- |
| Bridge | LuaSocket TCP on localhost, one port per game, newline JSON, monotonic sequence and request IDs |
| Environment | Gymnasium; normal Isaac; starts on floor 1; 8 game updates per action |
| Observation | 10 × 16 × 28 spatial grid plus 392 normalized scalar/entity/door features |
| Memory | Visited rooms, door destination visit counts, per-room cell visits |
| Action | `MultiDiscrete([9, 5, 4])`: movement × shooting × utility |
| Policy | CNN for the room grid, MLP for structured state, shared 256-unit layers, categorical actor and value critic |
| Algorithm | PPO with GAE, gamma 0.995, lambda 0.95, clip 0.2, Adam 3e-4, entropy coefficient 0.02 |
| Updates | Parallel: 128 steps per game (768 transitions for six games), 4 epochs, batch 128; single-game: 256 steps, batch 64; gradient clip 0.5 and KL early stopping |
| Episode end | Death or observed first-floor boss-room clear; time/idle limits truncate |
| Persistence | Atomic model/optimizer/RNG checkpoints, source hashes, episode and loss logs |

Movement: `0` stay; `1` left; `2` right; `3` up; `4` down; `5` up-left; `6` up-right; `7` down-left; `8` down-right. Shooting: `0` none; `1` left; `2` right; `3` up; `4` down. Utility: `0` none; `1` bomb; `2` active item; `3` pill/card. The learned policy chooses all three. There is no aiming or pathfinding controller.

The spatial channels represent the player, enemies, hostile projectiles, pickups, solids, pits, hazards, doors, player tears, and visit memory. Structured features include health, resources, weapon stats, room progress, eight doors, the nearest sixteen enemies, sixteen projectiles, and eight pickups. Hidden room contents and unrevealed map topology are not provided.

Observation profile `terrain_v2` corrects two errors found against the installed game's own `resources/scripts/enums.lua`: collision class 5 blocks entities other than the player, so an open doorway is **not** a player-solid tile; grid types 8/9/12 are spikes/toggling spikes/TNT, whereas 14/15 are poop/walls. The former belong in the potential-hazard channel; ordinary poop/walls do not. This is a label correction, not an automatic avoidance controller. Observation size, actions, rewards and game physics are unchanged. Old checkpoints explicitly use `legacy_v1` (including checkpoints without a profile tag), so their frozen evaluations remain reproducible. Both trainers and the evaluator honor the checkpoint profile; an intentional change requires a new run directory and `-ObservationProfile terrain_v2`.

`scripts/probe_terrain.py --port 10002` verified four real open doors were falsely marked solid by legacy encoding and passable by the correction, then crossed a door in 11 one-frame diagnostic actions. Evidence is in `runs/terrain-probe-1789241046763727700`. Scripted probe actions are never used for training or policy evaluation. Before this correction, the reward-only trial recorded one combat-room clear (episode 34, seed VALR GDNF), followed by death; it had no boss encounters or clears.

Reward components per transition, defined and versioned in `isaac_rl/rewards.py`:

| Event | Current `confirmed_v3` (same coefficients as `balanced_v2`) | Archived `legacy_v1` |
| --- | ---: | ---: |
| First visit to a 40-unit cell | +0.01 | +0.025 |
| First room visit | +2 | +2 |
| First room clear | +10, only if enemies were observed there | +5, including empty rooms |
| Enemy damage (max 200 HP/transition) | +0.2 per HP | +0.06 per HP |
| Enemy death | +0.5 | +0.25 |
| Player damage | −0.5 per half-heart | −2 per half-heart |
| New collectible | +2 | +2 |
| Health pickup | +0.3 per half-heart | +0.3 per half-heart |
| Player death | −5 | −15 |
| First-floor boss clear | +100 | +50 |
| Each action | −0.01 | −0.003 |
| Navigation shaping | `0.995 * Phi(next) - Phi(now)` | None |

The frozen 13,560-step baseline timed out idle on all three diagnostic seeds, dealt zero enemy damage, and earned positive reward on two runs merely by entering another empty room. `runs/diagnostic-policy-v1` preserves the checkpoint, manifest, every action/state, and all results. This evidence motivated the separate `balanced_v2` trial; it does not yet establish that the revision improves performance.

The frozen 50,592-step diagnostic in `runs/diagnostic-terrain-50k` also had **0/3 floor wins**. One unseen seed cleared a combat room (6 kills, 30 damage dealt) before idling; another idled without combat, and the third died. No boss was encountered. Its exact checkpoint, all attempts, initial/final states and action/state trace are retained. These short, different-seed diagnostics do not establish consistent improvement or first-floor competence.

After all requested evaluation results are saved, the evaluator performs a separately logged, **unscored** reset into a fresh normal starting room before disconnecting. This keeps the reserved game reconnectable: native game-over screens stop the update callback that polls for Python. `cleanup.json` records the operational reset; it does not add an episode or alter results. Failed/timed-out evaluation transitions are not retried for cleanup. `scripts/probe_evaluation_cleanup.py` verifies the death/reset/reconnection path using ordinary native fire damage in the reserved game; its scripted actions never feed training or performance evaluation.

Navigation uses a bounded 0–2 potential from distance to visible, open, unlocked, least-visited doors in clear rooms. It supplies reward feedback, not movement commands. Potential differences are applied across room transitions and use zero terminal potential, so round trips cannot farm positive discounted shaping return. This follows [potential-based reward shaping](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf). Combat/death reweighting is a separate experimental change, not a claim of policy invariance.

Room/cell novelty cannot be collected repeatedly by moving back and forth. The starting room is excluded from clear reward. In v2, a new cell no longer resets the 450-action no-progress clock; entering rooms, clearing them, damage, kills or pickups do. Time limits bootstrap the last state's value; death and success do not. GAE never crosses an episode reset. Resume inherits the checkpoint's reward profile; an explicit profile change requires a new run directory (`-RewardProfile balanced_v2` in the launcher).

The parallel collector issues all games' actions concurrently to independent native OS processes. It computes terminal observation values before resetting any finished game. GAE is computed separately along each game's timeline, then transitions are flattened for PPO optimization. Measured collection throughput is about 3.6 actions/second for one game, 10.6 for three, and 20.3 for six, without changing simulation timing. These are shared-policy rollouts, not six isolated learners duplicating training. Room-clear counts include empty rooms, so they are not a combat-clear or boss-clear metric.

The user-requested expansion to six workers was checkpointed at 42,912 steps / 83 episodes / 119 PPO updates. `runs/ppo-terrain-v3/before-six-workers.pt` preserves that handoff. Observations, rewards, model, optimizer and RNG state were retained; session manifests record the changed port set. The three added workers each passed live seeded bridge probes before joining training. By that handoff, one training episode had reached a boss (episode 77, seed FXMN ZBQY) but died without defeating it. No successful first-floor clear had been recorded.

Training runs at normal simulation speed. An experiment invoking `Game:Update()` between renders changed player displacement per update; that approach was removed. An RTX GPU is available, but this small policy defaults to CPU to avoid GPU transfer overhead and reserve resources for the games.

## Verification and evaluation

```powershell
py -3.10 -m pytest -q
# Probes must use an idle reserved worker, never a live trainer/evaluator port.
# Once the existing evaluation on 10002 has FINISHED naturally:
py -3.10 scripts/launch_worker.py 3
py -3.10 scripts/probe_bridge.py --port 10002
# Corrected-policy evaluation requires bridge 0.1.3 on the reserved worker:
py -3.10 -m isaac_rl.evaluate runs/ppo-damage-v4/latest.pt --port 10002 --episodes 100
```

The bridge probe checks a fresh episode reset, movement, firing states, exact frame increments, and reproduction of a seed. Tests exercise message fragmentation/staleness, observations, reward farming, terminal/truncation handling, GAE boundaries, actual PPO parameter updates, and the evaluation gate.

All seven live instances (six training and one evaluation) passed the probe, including hidden-worker runs. Their recorded empty-room movement trajectories are identical; this does not guarantee identical seeded combat collision outcomes. Thirty-six tests cover the core integration contracts plus three- and six-game concurrent collection, per-game advantages, terminal/reset separation, actual PPO updates, the Lua death callback, one-time kill/clear rewards, navigation-potential accounting, idle detection, immutable checkpoint loading, normalized seed uniqueness, honest cross-version reporting, corrected terrain labels, profile compatibility, evaluation cleanup, and the installed game's native enum definitions. New tests cover unlimited rollouts, actual-source HP observation, damage-signal compatibility, distribution reporting, and zero-win Wilson bounds. The native-enum test skips only when no private game installation exists.

Bridge 0.1.2 fixed the death callback's enemy filter to include dead enemies. Before that fix the kill counter stayed at zero; its attempted-hit damage rewards were later found unreliable against invulnerability. The legacy run resumed at step 8,184 with the corrected kill filter; episode 16 recorded three real kills. `runs/ppo-parallel-v1/before-kill-counter-fix.pt` preserves the pre-fix checkpoint. This restored legacy's intended +0.25 kill reward; v2 subsequently raised its own kill reward to +0.5.

Bridge 0.1.3 credits observed HP decreases across native updates, with confirmed death accounting, instead of attempted damage callbacks. `damage_attempted` remains diagnostic-only. A native Roundy regression recorded **20 actual damage versus 226.5 attempted**, within the room's 40 HP budget; its positive control recorded **8 damage and one real kill**. See `runs/review-20260913/damage-frame-observer`. All probe actions are unscored and never enter PPO. The reserved evaluation game intentionally retains bridge 0.1.2 until its already-running frozen one-million-step baseline finishes; do not reload or claim that baseline uses corrected rewards.

Completion requires at least **90 successes in 100 unique unseen normal first-floor runs**, with a 95% Wilson lower bound of at least 80%. The evaluator excludes recorded training seeds and previously evaluated seeds in its batch, freezes the model, and records every initial/final state and the complete action/state trace. Loading, hashing and archiving use the same immutable checkpoint bytes even while training replaces `latest.pt`. Seed comparisons ignore spacing and case. A success must show a boss encountered and defeated in a cleared first-floor boss room with the player alive. Checkpoint hash, source hashes, seeds, deaths, timeouts, and failures are retained. A short smoke evaluation cannot pass this gate; `--max-episode-steps` and `--idle-limit` allow clearly recorded, shorter diagnostic runs.

The trainer writes:

- `status.json`: live process, episode, reward, loss, steps and game state telemetry.
- `episodes.jsonl`: completed episodes with actual seed, rooms, damage and boss result.
- `live-states.json`: bounded snapshots after each update for behavior inspection without taking the game connection.
- `updates.jsonl`: PPO losses, entropy, KL, gradient norm and explained variance.
- `latest.pt`: model, optimizer, RNG state and training metadata after every update.
- `training_seeds.json`: seeds excluded by evaluation.
- `success-*.json`: raw terminal states for any training successes.

## API references

The bridge uses the game's [LuaSocket support](https://wofsauge.github.io/IsaacDocs/rep/tutorials/Standard-Library.html), [input and update callbacks](https://wofsauge.github.io/IsaacDocs/rep/enums/ModCallbacks.html), and [seed/reset commands](https://wofsauge.github.io/IsaacDocs/rep/tutorials/DebugConsole.html). PPO follows the [original clipped-policy objective](https://arxiv.org/abs/1707.06347).
