
# Notes for Isaac RL — Day 2
## Day 2 notes
1. **No lifecycle for persistent processes.** Games stay up and reconnect
 rather than exit. From the outside, idle / in-use / orphaned / dead
 look identical. There is no health check on connect, no status marker, no
 teardown on trainer exit. Two visible symptoms: live stats stop
 updating past 1M steps silently, and processes survive training stops.
2. **Doesn't measure before scaling.** Ran 6 workers until the machine
 saturated, then dropped to 3. The cycle-time split that should have
 been measured on day one. 35.3s collection vs 0.17s PPO update. only
 surfaced after being asked. Same pattern as note 6 again: benchmark
 follows intervention, not the reverse.
3. **Logged metrics don't measure competence.** After 500k clean steps
 with a flat policy, the agent diagnosed the bottleneck from logs
 alone: return correlates with combat clears (0.806), kills (0.715),
 and damage (0.687), so it ranked reward/credit-assignment as primary.
 The diagnosis was internally consistent. but the logs said "2.4
 kills, 0.40 combat clears per episode" and the actual gameplay shows
 the agent standing still and shooting while enemies walk into its
 tears. Kills are proximity, not aiming or dodging. The agent never
 asked what the kills represented. It measured the wrong thing and
 trusted the number.
## Human interventions log
**001 — Throughput, display, lifecycle** (~1.1M steps)
Trigger: Open items, not failures. Throughput and compute target were
flagged Day 1 (note 6) and never resolved. Display and process lifecycle
surfaced from observation.
Intervention:
"1. Benchmark throughput. GPU vs CPU — measure PPO update time separately
from collection. More workers — check CPU headroom, ports, Steam login
limits. If GPU helps, use it; if collection dominates, CPU stays. Don't
guess.
2. Display breaks past 1M steps and only shows current values. Processes
don't exit on training stop — they reconnect. Fix the cutoff, add a
rolling window, give the process a lifecycle so I can tell alive / idle /
dead."
Result: Benchmark answered the GPU question — collection 35.3s vs update
0.17s, ~99.5% environment-bound. GPU off the table. Worker count went the
other direction: 6 → 3, CPU-saturated. Display and lifecycle items open.
**002 — Independent audit of the flat run** (~1.58M steps)
Trigger: 500k clean steps after the reward fix with no policy movement.
Charts show stable KL (~0.013), high entropy (~4.6/5.19 and rising),
moderate explained variance (0.64 broad, 0.73 recent), 1,594 episodes /
20 boss encounters / 0 boss defeats. Agent's own diagnosis ranked
reward/credit-assignment primary, architecture second, but couldn't see
that the kills were passive.
Intervention:
"Independently audit the current Isaac RL training run, source, logs, and
gameplay behavior.
The policy has been flat for roughly 500k steps. Explained variance is
moderate, entropy remains high, and shaped returns are not improving
meaningfully. Gameplay suggests that the agent often stands still and
shoots while enemies walk into its tears, so recorded kills and damage
may not represent active aiming, movement, or dodging.
Determine the real bottleneck: reward design, optimization, architecture,
data, observability, or something else. Do not assume the logged metrics
measure competence. Use the available evidence, identify what is missing,
and recommend the highest-value next experiment or intervention."
