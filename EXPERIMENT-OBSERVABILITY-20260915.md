# Observability intervention: enriched four-snapshot/action inputs

## Decision and status

One direction: `combat_history_v3`, native current-state enrichment plus four
snapshots and their three intervening executed actions. No GRU or LSTM.
**Activated: six-worker `runs/ppo-observability-v8`, charts8769.**
The user approved the one-time checkpointed handover. Parent completion-v7 is
stopped and preserved at its final4,014,636 decisions. First optimized treatment
update completed at4,016,172; all other learning settings remain unchanged.
The handover approval is consumed; preserve the new live learner.

September15 identity repair: learner2152 later exited at4,407,822 observed
decisions, last optimized checkpoint4,406,316. Evidence is preserved in
`runs/identity-recovery-20260915-2303`; see OPERATIONS.md for recovery status.
Native entities can share InitSeed; the original uniqueness check therefore
rejected legitimate states. Exact crash-frame entities are unknown because that
raw response was not saved. A reserved native fixture reproduces the same error.
The association repair below changes no feature widths or learning settings,
and does not restart the experiment's exposure budget. New exception capture
retains current raw native states separately from the last optimized checkpoint.

This tests whether missing short-horizon combat information is a dominant
bottleneck. It does not assert that the entire game becomes Markov, or that a
policy using these inputs has learned to dodge.

## Evidence and choice

The original 392-vector encoder sorts entities independently by player distance
on each snapshot. Hazard slots contain position, velocity and size, but omit
kind, type/variant/subtype and InitSeed. Lasers become points. The raw player's
`fire_delay` is **MaxFireDelay**, not current **FireDelay**. The raw bridge had no
player hitbox, damage cooldown, invincibility or laser path. The feed-forward
policy has no previous observation/action input, and PPO shuffles individual
transitions after computing GAE along each environment's trajectory.

Enrichment is needed because stacking cannot reconstruct fields that are never
exported. Explicit stacking is the lowest-complexity temporal mechanism here:
every shuffled PPO sample carries its own ordered context; episode boundaries,
bootstrap observations and hidden states do not require a sequence sampler.
A GRU would require that sampler and hidden-state handling as well as enrichment.
It remains a candidate if this finite context is insufficient. We do not infer
that either recurrent architecture would automatically solve the plateau.

Four snapshots span three decision intervals: at the unchanged four-tick hold
and 30 native updates/s, that is **12 ticks / 400 ms**, with a 133.33 ms decision
interval. This is a deliberately bounded short-memory test, not a tuned optimum
or a claim about multi-second planning.

## Exact change

Native bridge 0.1.6 retains the existing 12-column entities and adds a versioned
`combat_v2` sidecar (originally combat_v1/0.1.5). It exports:

- Player Size/SizeMulti, current FireDelay, damage cooldown in **render frames**,
  invincibility effect and combined invulnerable flag, shooting availability,
  flight and collision class.
- Per-entity identity, collision ellipse and damage/class; NPC state, state
  frame and sprite frame; projectile height/falling speed/acceleration; bomb age.
  Bomb age is not a remaining-fuse estimate: vanilla exposes a fuse setter but
  no readable countdown. The originally attempted property was invalid.
- Laser kind, endpoint/angle for lines, center/radius for rings, up to eight
  points for sample lasers, timeout and explicit geometry validity.

The new encoder keeps the current terrain_v2 grid and 392-vector as an exact
prefix. Each of four snapshot vectors then adds 12 player values, 16 enemy
tracks of 16 values and 32 hazard tracks of 46 values. Projectile/bomb/laser kind
is one-hot; Type/Variant/SubType remain separate bounded fields. Extended
velocity encoding does not hard-clip at 15 world units/tick.

Hazard selection uses distance to beam/ring geometry, not only laser origin.
Entity-lifetime `track_id` aligns all ages of each enriched track within a sample;
ID numbers are not fed to the policy. A namespaced GetData value and monotonic
counter identify each native lifetime without consuming RNG. The original
InitSeed remains raw evidence and a distance tie-break, not a unique dictionary
key. Current entities have priority; spare slots
retain recently disappeared entities. Legacy prefixes retain their original
sort semantics. Three one-hot 9/5/4 actions, four validity flags and four native
ages accompany the history. No current-to-be-chosen action is included.

History is per environment, copied into immutable rollout inputs, zero-padded
on reset, and cleared on episode/stage/room changes. Terminal history is valued
before resetting the environment. The original visit memory remains available.
Missing required schema fields fail explicitly. Telemetry reports omitted
hazards/enemies, simplified curves and invalid geometry.

Input shape changes from `10x16x28 + 392` to `40x16x28 + 8590` (2,132 values per
snapshot plus 62 history/action values). Architecture version 2 changes only the
two input widths: hidden layers, CNN output channels, action heads and critic
remain the same. Added input weights and Adam moments start at zero; all existing
weights, moments, optimizer steps and RNG states are inherited. No auxiliary
loss, predictive controller, pretrained weights or recurrent layer is added.

The additional `combat_clears_alive` episode counter is measurement only; it
requires a native uncleared-to-cleared combat room and a living player at that
transition. It does not change reward or termination.

## Constants and mechanical changes

Six workers 0/1/2/4/5/6; CPU, two Torch threads; four ticks, physical_v1;
completion_v4; entropy .002; Adam lr .0003, eps .00001, betas .9/.999;
gamma .9974968671630001; GAE lambda .9746794344808963; rollout 256/environment,
batch 256, four PPO epochs, clip .2, target KL .025, gradient clip .5;
max episode 6,750 decisions and idle 900; action heads 9/5/4 all remain unchanged.

There are **no timing-unit conversions or independent hyperparameter changes**.
Input-width growth, new zero moments and validity masks are mechanical
requirements of the representation change. Parameter count increases from
390,755 to 2,493,763; this is recorded rather than described as the identical
architecture. New observation profiles require an explicit fork. The exact old
architecture2 manifest is accepted for the metadata-only seed-association repair;
all model/Adam tensors are retained bit-for-bit. Arbitrary layout changes and
unsupported reverse migrations are rejected. Native schema1 is rejected by the
repaired encoder; native games must reload schema2 before recovery.

This is an association implementation repair, not a second learning intervention.
The separate InitSeed-keyed damage bookkeeping is unchanged: modifying it could
alter idle termination and metrics despite zero damage reward, and needs a
separate audit. Historical damage counts remain unsuitable as competence evidence.
Native API references: [entity GetData lifetime](https://wofsauge.github.io/IsaacDocs/rep/Entity.html#getdata)
and [explicit-seed Game.Spawn](https://wofsauge.github.io/IsaacDocs/rep/Game.html#spawn).

## Preservation and activation

Preparation snapshot of the then-current live checkpoint:
`runs/observability-intervention-20260915/baseline.pt`, **3,970,764 decisions,
4,475 updates, 10,334 historical episodes, 6,000,323 counted native ticks**.
SHA256 **a413289a52b31b6e71ef6a761e1da6e7d6575b7eb23f8e9bb338d4c05c337af5**.
Its original source is archived as `baseline-source.zip` from Git `40266ae`.
Existing checkpoints, including the saved 5,377,722-step control run, are intact.

This preparation snapshot is for migration/cost checks. **At an approved
handover, preserve and fork the final current checkpoint, not this now-older
snapshot.** Record that final SHA, decision/native-tick origin, source hashes
and new run directory in experiment metadata. The live learner must save/exit
cleanly; verify PID+creation time and free ports before any new listener. Only
then update/restart the six scoped private games with the current bridge. Never install
the test fixture in training, modify personal saves, or enable an automatic
restart policy. Require all six native schema handshakes before the first update.

Fork command once the approved handover/preflight is complete:

```powershell
.\scripts\launch_parallel.ps1 -Run runs/ppo-observability-v8 `
  -Initialize runs/observability-intervention-20260915/parent-final.pt `
  -ObservationProfile combat_history_v3 -Instances 0,1,2,4,5,6 `
  -Device cpu -Steps 0 -DashboardPort 8769
```

Do not execute this again against occupied training ports. Current dashboard is
http://127.0.0.1:8769/;8768 remains the previous run's read-only history.

## Predeclared falsification condition

Budget: **1,200,000 additional aggregate decisions** from the final fork origin
(nominally 4.8M native ticks; report exact ticks, including early terminal holds).
This is a review threshold, not an automatic stop/restart policy. Freeze the
checkpoint nearest the threshold and do not keep changing the budget until a
favorable checkpoint appears.

Compare the frozen parent and treatment on the **same 100 previously unseen
normal full-floor seeds**, one stochastic episode per policy/seed with a fixed
per-seed action RNG initialization. Seeds must be absent from the union of both
training-seed sets. Use the same four-tick hold, normal Isaac, episode/idle budgets
and no gameplay assistance. Preserve native traces and room-clear evidence.

Primary mechanism outcome: probability of completing **at least one combat room
alive** during a full-floor episode, including every requested seed in the
denominator. A seed that never reaches or clears combat scores zero. This avoids
conditioning only on easier surviving encounters. Let delta be treatment minus
parent probability. Compute a conservative approximate 95% upper bound for delta
as treatment Wilson upper minus parent Wilson lower, each with z=1.96 (97.5%
one-sided marginal bounds; Bonferroni joint coverage). Keep the seed-paired raw
outcomes, even though this conservative bound does not exploit their correlation.

**Falsification: if that upper bound is below +0.10, reject the operational
hypothesis that this short-horizon information fix is the dominant bottleneck
and yields a meaningful >=10-percentage-point improvement at the fixed budget.**
An interval still admitting +0.10 is inconclusive, not success. A lower bound
above zero supports improvement; it does not uniquely establish causation.

Secondary outcomes: native-verified first-floor boss wins, boss encounters, total
combat rooms cleared alive and health lost per combat encounter. Report timeouts,
deaths and failure to enter combat. These explain *how* outcomes changed; they
cannot replace the predeclared primary endpoint after seeing results. Shaped
return, entropy, explained variance, movement, damage and kills are not competence.

Consistent held-out first-floor boss clears remain the project success criterion.
A frozen-parent comparison establishes improvement over that parent, not causal
separation from extra training time. A matched continued terrain_v2 control would
be needed for stronger attribution. Do not claim this run alone proves the old
observation was the only bottleneck.

## Verification and limits

Offline regression coverage includes schema failure, old-state aliases becoming
distinct, fast velocities, laser/ring geometry, identity crossings/disappearance,
action alignment, immutable histories, independent environments, room/episode
boundaries, truncation bootstrap inputs, exact inherited Adam moments/RNG,
single/six-environment trainer fork/resume and real gradients into new inputs.

Native smoke used ONLY reserved instance 3/10002 with a separately installed
test fixture creating known hazards. It is **not a performance evaluation or
training data**. The first attempt encountered a game-over screen. A later attempt
captured invalid native JSON: `GetSamples()` on a ring returned NaN coordinates.
The shape-specific fix is covered by Lua regression tests and a passing native
rerun. HasInvincibility alone was false during damage immunity; the exported
combined flag also uses positive native damage cooldown.

Passing 24-transition native trace: `native-smoke-4/` inside the experiment
artifact directory. All holds were four native ticks; median end-to-end step
133.364 ms. FireDelay changed between -1 and 10; damage cooldown decreased by
eight render frames over four native ticks. Both projectile variants, native
enemy phase, player hitbox, straight beams and rings were observed; the completed
history spans 12 ticks. Curved-beam sampling has Lua tests but has **not** been
verified on a native curved attack. There were no omitted hazards or invalid
geometries in this small smoke sample; this does not establish full-floor coverage.

The disposable native probe processes were closed; the fixture was moved out of
the runtime mods directory into `deactivated-native-fixture` in the artifact
directory. No production game was stopped or had its loaded bridge replaced.

`input-audit.json` compares the frozen parent on 25 captured native states:
**maximum initial logit/value difference = 0**. On synthetic repeated copies
only, median CPU PPO update cost was .448s -> 1.374s; six-environment inference
.640ms -> 1.058ms; encoding .444ms -> .879ms; rollout observation storage
28.55MiB -> 155.33MiB. These disposable updates were never saved as training.
Actual live treatment throughput and full-floor coverage remain to be measured.

Residual information limits: a finite 400ms context, 16 enemy/32 hazard tracks,
eight-point curve approximation, projectile behavior flags and some special
effect hazards and remaining bomb fuse not encoded, no persistent action history across rooms, no
multi-second memory and no claim of a fully Markov state. Overflow/invalid geometry
must be audited in actual training before blaming optimization for failure.

Native API references: [EntityPlayer](https://wofsauge.github.io/IsaacDocs/rep/EntityPlayer.html)
for cooldowns, [EntityLaser](https://wofsauge.github.io/IsaacDocs/rep/EntityLaser.html)
for shape-specific geometry, [Entity](https://wofsauge.github.io/IsaacDocs/rep/Entity.html)
for collision size and [EntityProjectile](https://wofsauge.github.io/IsaacDocs/rep/EntityProjectile.html)
for projectile motion fields. Installed-game traces take precedence over assumed
availability or interpretation of these APIs.

### Activation regression, September 15

The user approved the one-time handover. Final parent is **4,014,636 steps /
4,504 updates / 6,175,744 native ticks**, SHA256
**58a28f1524b62dd64432bda0ae63a19ac3db4ce312c13a9401d31ce1f429db46**.
The old learner and all six old private games exited with code0.

The first new learner23940 failed in collection at4,014,666 reported steps,
before any optimizer update: a real bomb omitted the nonexistent
`ExplosionCountdown` property. Initial checkpoint remains4,014,636 with exact
zero-padded parent model/Adam/RNG inheritance verified; no optimized training
was lost. Terminal status, traceback and bridge source are retained in
`activation-failure/`. No live learner was stopped in response to this error.

Correction within the same information mechanism: replace the unimplemented
fuse slot with actual native `Entity.FrameCount` bomb age; preserve the same
input width and all training settings. Bridge0.1.5, two added regression tests,
**142 tests passing**. An expanded reserved-instance native fixture passed all
12 checks over24 actions, now including changing native bomb age. Evidence:
`native-bomb-smoke/`; no probe transitions enter learning. This is completion
of the specifically authorized handover, not an automatic restart policy.
[EntityBomb API](https://wofsauge.github.io/IsaacDocs/rep/EntityBomb.html) documents
the setter; [Entity API](https://wofsauge.github.io/IsaacDocs/rep/Entity.html)
documents FrameCount. No hypothetical default fuse is supplied to the policy.

### Completed handover and review origin

Corrected learner **2152 /1789512768.613361**, session
session-1789512773700976500, resumed the untouched initial checkpoint. Snapshot
`treatment-resumed-initial.pt` SHA256
**5676db50c5e0dbcc0ca258be76c3c485bdc4950220ad8886c42634cbf57cac35**
has exact model/Adam/RNG/counter inheritance and the corrected source hashes.
All six native workers use0.1.5/combat_v1, four ticks and 12-tick history.

First update4505:4,016,172 decisions/6,181,886 native ticks; collection38.0885s,
PPO0.3693s,wall40.7005s,37.739 decisions/s. The archived
`first-observed-optimized.pt` is the SECOND update4506 at4,017,708 decisions,
SHA256 **23955e872338aa98e316e78c5bd529184edaab9771f2f2c0fa6035447745a3df**.
All16 tensors changed and finite; new inputs have nonzero learned weights; Adam
groups unchanged. Evidence is in the initial/resumed/optimized verification JSONs.
The first three optimizer updates completed without new stderr errors. No new
held-out evaluation was launched; competence remains unproven.

Fixed review threshold: **5,214,636 total decisions** (+1.2M); nominal counted
native ticks10,975,744, with exact early-terminal ticks reported separately.
This does not cap or automatically stop the detached learner. Only the specifically
approved handover and its startup correction were performed; no auto-restart
policy was introduced. Current native process IDs and failure history are in
OPERATIONS.md; runtime fixture3 is stopped and deactivated.
