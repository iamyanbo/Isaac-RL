# Next training intervention: lower entropy regularization

User direction: stop diagnosis-only work, choose and implement one adaptation, preserve the current checkpoint, and launch the next training run with everything else constant.

## Chosen change

Change PPO's **entropy coefficient from 0.02 to 0.002**. Do not change reward coefficients, observation encoding, model architecture, optimizer/hyperparameters, action hold, worker count, rollout size, episode limits, or game physics.

Rationale: all three action heads remain diffuse after approximately 2.4M steps. The learned stochastic policy samples movement but has not developed reliable aiming, damage avoidance, or boss completion. Reducing the sustained randomness incentive tenfold is the highest-value single optimization intervention chosen for this run. It allows existing reward differences to produce more concentrated action choices while retaining nonzero exploration. This is an intervention hypothesis, not a proven cause or guaranteed improvement. More concentrated bad behavior or increased idling is a real failure mode.

No additional diagnosis run is a prerequisite. The already-running random-action diagnostic is separate and does not decide whether to start this adaptation.

## Preserved baseline

- File: `runs/entropy-intervention-20260914/baseline.pt`.
- Saved training steps: **2,417,208**; episodes: **7,895**; PPO updates: **3,429**.
- SHA256: `eeffc748b50552e5947e3575585d92e1294f60dadc6b0f8e94996ac026a294ae`.
- Exact checkpoint bytes copied from the live run's atomically replaced `latest.pt`; all model tensors verified finite.
- Model, Adam state, Torch/NumPy RNG state, training-seed exclusions, and history retained. Baseline configuration/status copies are alongside the checkpoint; status can be ahead of the saved checkpoint and is not its authoritative step count.

## Controlled configuration

Six games, instances **0/1/2/4/5/6**, ports **9999/10000/10001/10003/10004/10005**. CPU with two Torch threads. Same 128 transitions per worker (768 total), minibatch128, four PPO epochs, Adam lr0.0003/eps0.00001, clipping0.2, targetKL0.025, gamma0.995, lambda0.95, grad clip0.5. Same architecture1, confirmed_v3 rewards, terrain_v2 observations, eight native updates per action, normal Isaac first-floor episodes, max3375 actions, idle450, and uncapped training.

New run: **`runs/ppo-entropy-v5`**, initialized from the preserved baseline. `--entropy-coef` is explicit, saved in config/checkpoints, and logged in status/update records. Old checkpoints without this field retain0.02; subsequent resumes inherit0.002. Changing the coefficient inside an existing run directory is rejected so the baseline history cannot silently become a treatment run.

Native episode seeds/states change across a process handover. This is a one-factor training-configuration intervention, not exact deterministic game replay. Inherited recent-episode and loss summaries initially describe the parent; only new run log records are treatment evidence.

## Launch and authority

**Prepared, not launched.** Existing learner PID51580 is still live. The standing instruction says never stop live training. Reusing its six workers requires one explicitly authorized checkpointed handover; no stop request has been issued. At the capacity check, Windows commit was95.69%, leaving4.11GiB; duplicating six games plus a learner risks another allocation failure. A one-worker substitute would change the experiment's worker count and update batch, so it is not being substituted.

After the user authorizes the one-time handover, preserve the old learner's final checkpoint/status, verify its exact process exit and released ports, and launch:

```powershell
.\scripts\launch_parallel.ps1 -Run runs/ppo-entropy-v5 -Instances 0,1,2,4,5,6 -Initialize runs/entropy-intervention-20260914/baseline.pt -Device cpu -EntropyCoef 0.002 -DashboardPort 8766
```

The launcher refuses occupied training ports. Do not launch a duplicate learner or stop training automatically. The old dashboard remains on8765; the new run's charts use8766.

## Verification and next review

Tests exercise old-checkpoint defaults, coefficient inheritance, invalid values, same-run mutation rejection, identical-rollout loss decomposition, actual fake-game PPO optimization, checkpoint persistence and resume. Both PowerShell launchers must parse. Before handing off the native launch, verify at least one optimized checkpoint, finite model weights, coefficient0.002 in config/checkpoint/update/status, all six trainer-owned sockets, and no change to the controlled settings above.

Implementation verification: **84 tests passed**; both launchers parsed; the baseline digest and held-constant configuration were checked against the saved manifest. These are implementation checks, not evidence that gameplay has improved. Native treatment training has not started while handover authority is pending.

Review after approximately **300k additional transitions**, then again near600k without automatically stopping training. Compare only newly collected episodes and a frozen-policy full-floor evaluation against the preserved baseline: combat clears, boss reach, verified wins, death/idle endings and health loss. Entropy reduction, shaped return, movement, or loss alone is not success. Do not add a second training intervention before judging this one. The original goal remains consistent first-floor clears including the boss.
