# GRU replaces finite stacking — September 16, 2026

## Decision and activation status

User explicitly selected **recurrence, not another reward intervention**. Implemented
`combat_gru_v4`, architecture3: retain native combat enrichment, replace four-frame/
three-action stacking with one current snapshot and a single-layer256-unit GRU.
The immediately previous executed action (9/5/4 one-hot plus validity) conditions
the transition; no old observations or multi-action stack enter the policy.
Existing visit-map memory remains unchanged. No LSTM, auxiliary objective,
curriculum, control change or further reward adjustment.

**Prepared and tested, not activated.** Existing six-worker native-clear-v9 learner
37144 remains live and unchanged. A NEW one-time checkpointed handover approval
has been requested; old reward-handover approval cannot be reused. Next run will
be `runs/ppo-gru-v10`. At activation preserve and migrate the FINAL live checkpoint,
never roll back to the preparation snapshot. Do not automatically stop/restart.

## Why this mechanism

The enriched stack spans only12 native ticks/400ms and discards history at room
changes. It cannot carry an earlier attack phase or action outcome for seconds,
nor retain learned episode context across rooms. GRU makes that context learnable
with explicit recurrent state. The prior frozen parent/treatment comparison had
zero boss wins in100 seeds each; the short-context treatment did not establish
competence. Its late snapshot and broad interval did NOT formally isolate memory
as the sole cause. That uncertainty does not justify another reward tune instead
of the recurrence intervention the user selected.

GRU keeps one hidden vector instead of an LSTM hidden/cell pair. Selection is an
implementation-cost choice, not a claim that GRU universally outperforms LSTM.
The recurrent PPO requirements apply to either: ordered sequences, correct state
initialization, episode masks and terminal bootstrapping. See the primary
[PyTorch GRU reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)
and [recurrent PPO implementation](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/ppo_recurrent/ppo_recurrent.py).

## Actual network, sampler and boundaries

Input:10x16x28 grid +2,151-vector (same2,132 enriched current values,18 previous
action values, one action-valid flag). Native bridge remains0.1.6/combat_v2.
No current-to-be-selected action, future state, numeric track IDs or reward leak.
Current entities retain threat-distance ordering; unlike explicit stack alignment,
association over time must now be learned. This is a limitation, not a claim of a
fully Markov observation or an identity-preserving entity-slot recurrence.

Existing CNN/vector encoders and shared256 features feed a256-unit GRU. A linear
readout of its hidden output adds to the shared features before unchanged actor
and critic heads. Readout starts at zero: the fresh random recurrent weights do
not immediately perturb the projected current-frame policy. Readout learns on
the first minibatch; gradients then reach GRU weights (verified in tests and native
smoke). The residual path can learn to ignore memory; memory use is not assumed.
Parameter count decreases2,493,763 ->1,301,347 because the large stack input goes.

Each game owns independent hidden state. It carries across room transitions AND
rollout/update boundaries, and resets only on death, success, time/idle truncation
or fresh process/session episodes. No hidden state survives a native episode reset.
Process checkpoints preserve model/Adam/RNG, not native game state; resuming starts
fresh games and zero hidden states explicitly, not stale checkpoint memory.

Collector records the hidden state BEFORE each observation and episode-start mask.
The bootstrap uses the resulting transition's next observation and the hidden
state AFTER the action-selection observation, BEFORE any environment reset. This
value probe does not advance collector memory twice. True terminals bootstrap0;
time/idle truncations bootstrap their terminal observation, but GAE stops there.

PPO preserves the current rollout256/env and minibatch256 transitions. Normal
six-worker updates shuffle six complete256-step per-environment sequences; they
never shuffle individual timesteps. Backpropagation spans up to256 decisions
(34.13s at four ticks/decision); episode resets sever gradients. Forward memory
can persist longer than this gradient horizon. Short interrupted/tail sequences
are unpadded, grouped only with equal lengths, and every real transition is used
exactly once per epoch. No padding can enter losses or advantage normalization.

Sequence initial states are detached recorded behavior states; subsequent states
within the sequence are recomputed under current parameters each PPO pass. This
is truncated recurrent PPO, NOT exact full-episode backpropagation. Initial states
and collector carry can become stale as weights change; no burn-in or full-episode
refresh is claimed. This known approximation is recorded, not silently described
as exact hidden-state reconstruction under updated weights.

## Constants and explicit checkpoint projection

Rewards remain **native_clear_v5**, unchanged coefficients and qualification logic.
Six workers0/1/2/4/5/6; CPU/two threads; four ticks/physical_v1; entropy.002;
gamma.9974968671630001, GAE.9746794344808963; rollout256/env, batch256; four PPO
epochs, clip.2, KL.025, gradclip.5; Adam lr.0003/eps.00001/betas.9/.999;
episode6750/idle900; action heads9/5/4; seed20260912; uncapped training all inherit.
No unit conversion or independent hyperparameter tuning. Sequence organization,
new recurrent parameters and input projection are mechanical parts of the one
temporal-mechanism replacement, not hidden changes to batch/rollout sizes.

The baseline is byte-identical; the migrated policy is **NOT** functionally
identical to the old stack policy. The projection retains current-grid channels,
the current enriched vector and most-recent action input columns, discarding old
snapshots, older actions and stack validity/age columns. Other existing weights,
biases and action/critic heads inherit exactly. Adam moments use the same column
projection, preserving existing step counters and group settings. New GRU/readout
parameters start with fresh Adam state; no global optimizer reset. The new action
validity column starts with zero weight/moments. New-run recent reward/loss windows
clear, while historical counters and seed exclusions carry forward.

Preparation baseline: **7,637,190 steps /6,863 updates**, SHA256
**150b56fe9256e85a977f839e4d3fb02bf12c5fd5d5d70afab1f62915184579c4**,
`runs/gru-intervention-20260916/preparation-baseline.pt`. This is for disposable
verification only; final activation origin must be saved later. Original
5,377,722 control, observation-parent and reward-parent checkpoints are retained.

## Verification (not competence)

Regression coverage: native enriched-prefix equality; no retained stack; causal
executed actions across rooms; fixed-action reward/idle/end equivalence; online vs
ordered-sequence probabilities; per-worker resets; no cross-episode gradient;
gradients back beyond four frames; sequence token coverage/tails; no double-update
on value probes; actual GRU PPO gradients; model/Adam projection and exact resume;
real single/six-env trainer fork with multiple updates/truncations; frozen evaluation
memory carried within and cleared between episodes. Feed-forward regression suite
also remains intact. **188 tests pass**; full suite18.85s.

Recorded-native input audit25 states: all current enriched fields match exactly.
Synthetic repetition sizing at production256x6:38.85MiB rollout observations plus
1.5MiB stored hidden states; six-env action/value inference median3.42ms; two full
four-epoch CPU updates2.849s and2.834s. These are repeated inputs/synthetic rewards,
not native throughput, learning progress or policy comparison. Disposable weights
were never written to a training checkpoint.

Reserved-native smoke:64 real actions through combat_gru_v4, recurrent PPO update
with finite changed GRU weights, checkpoint unchanged. Native step median.1295s,
disposable update.106s. Same existing reserved game13848/1789565921.170414,10002;
only bridge enabled, no fixture/assistance, no training-port connection, no game
restart/Steam/save changes. Afterward an unscored fresh-room reset parked it and
the smoke process released10002. Native traces and audit results are under
`runs/gru-intervention-20260916`. No full-floor evaluation was launched this turn.

## Falsifiable review

At the authorized final-checkpoint fork, set retention threshold **parent final
+1,200,000 decisions**, first optimized save at/above it (max1,535 overshoot).
Record actual decision/native-tick exposure and never substitute a later favorable
model. Retention does not stop the live learner. Review this fixed snapshot against
the frozen activation parent on100 new matched native full-floor seeds, selected
before scoring, excluding both training sets and previous evaluations. Same
per-seed stochastic RNG, alternating policy order, hold4, episode6750/idle900;
verify every claimed boss completion against native living boss-room clear state.

Primary endpoint: native first-floor boss-win probability, all requested seeds in
the denominator. **Falsification: if the conservative95% upper bound on GRU-minus-
parent boss-win probability is below +0.10, reject the operational claim that this
memory replacement alone yields a meaningful >=10-point gain at this budget.**
If evidence or target checkpoint is missing, or the bound still admits +0.10,
the result is inconclusive. Lower shaped return, entropy, damage and room counts
cannot substitute for success. A same-checkpoint hidden-reset ablation is a future
memory-use diagnostic, not an additional training mechanism or current evaluation.

Parent comparison includes additional training and stack removal; it is not proof
of recurrence's isolated causal effect. Project completion still requires >=90/100
unique native held-out first-floor boss wins with95% Wilson lower bound>=.80.

## Launch only after new handover approval

Save the LIVE learner's final checkpoint and verify clean exit/free ports; preserve
its bytes as parent-final.pt. Use that FINAL parent to calculate the review target
and populate new-run snapshot_steps.json BEFORE launch. Reuse six owned games.
No Lua change or game reload is required. Example (not executed while awaiting approval):

```powershell
.\scripts\launch_parallel.ps1 -Run runs/ppo-gru-v10 `
  -Initialize runs/gru-intervention-20260916/parent-final.pt `
  -ObservationProfile combat_gru_v4 -Instances 0,1,2,4,5,6 `
  -Device cpu -Steps 0 -DashboardPort 8771
```

Omit RewardProfile/EntropyCoef/Frames so they inherit. Do not use the old
verify_reward_handover.py for this architecture change: it deliberately requires
exact same model shapes/reward-profile migration. Verify the declared GRU input/
Adam projection and actual new recurrent optimizer updates instead.
