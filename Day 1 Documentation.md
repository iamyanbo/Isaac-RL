
## Notes for Isaac RL

1. **Doesn't survey prior work.** Two gaps: it doesn't check whether
   similar projects already exist, and it doesn't look for established RL
   architectures or best practices before implementing. Could be that
   these are baked into the model, or that it isn't trained for proper research process.
   Black box, so we can't distinguish.
   - It *does* search for Isaac-specific documentation, so some domain
     knowledge is clearly retrievable, and it knows when it doesn't know.
   - It cites policy-improvement literature, but the paper surfaced was
     Ng, Harada & Russell (1999),  and only after substantial
     implementation and testing. The research pipeline runs late and out
     of order.

2. **Disregards throughput optimization** Didn't parallelize or spin up
   concurrent instances on its own, despite it being a well-known
   prerequisite for sample-hungry algorithms. We had to suggest it.
   (See Interventions Log 001.)

3. **Over-probes and burns tokens.** It runs tests and diagnostic checks
   more often than the task justifies. The ideal would be to suspend
   cleanly and resume on a hook, but both paths have a cost: either the
   cache evicts during sleep and resumption is expensive, or the process
   stays warm and pays for a constant idle stream. Neither is clean.
   More consequential than expected. See Entry 002.

4. **Correctly identities the basic architecture should be PPO.** No concerns here, as it should.

5. **Handled menu navigation pragmatically.** Isaac requires several menu
   steps before gameplay starts. The agent learned to send the keystrokes
   directly. Not elegant, but correct for the problem. It is interesting to see that for domains it doesn't know, it tries to get a visual understanding by taking screenshots and analyzing. It learned to send the manual keystrokes after it determined there is menu navigation. But a more elegant solution is to boot directly into game through a bypass.

6. **Didn't measure before choosing a compute target.** Defaulted to CPU
   and never benchmarked CPU vs. CUDA. The choice may be right, the
   pipeline is environment-bound, and a small MLP on vector observations
   often runs faster on CPU. But we can't tell whether that reasoning was
   applied or whether CPU was just the default. The failure isn't the
   choice, it's the absence of a measurement behind it.

7. **Reward design is reasonable on paper; penalties are unverified in
   practice.** The profile is simple and sensible. Prior experience says
   damage/step penalties at initialization can lock in a suboptimal
   survival policy before the agent discovers progress. Whether the
   current penalties dominate early training hasn't been measured.
   This needs a magnitude check, not removal.

8. **Poor sense of training time.** Evaluates far too early relative to
   the game's horizon. At 20k steps it's asking whether the boss has been
   reached, which is a question that can't be answered until orders of magnitude
   later. Either it's calibrated on short autoresearch loops
   (Karpathy-style) or it's a general agent time-horizon issue.

9. **Iterates before there's signal.** Already on architecture v3 at
   ~20k steps, when the policy is still effectively random. Every
   subsequent change is being evaluated against noise, not against
   evidence. A continuation note of #8.
## Human interventions log

**001 — Parallelization** (~40k steps)
Trigger: Throughput was low. Agent was running a single environment.
Intervention: Told it to spin up concurrent Isaac instances with a shared
policy. Result: 7 workers, later narrowed to 6 training + 1 eval.
Note: Agent did not propose this on its own. Human-directed.


**002 — Manual goal overwrite** (~85k steps)

New Goal: 
```
Build a RL agent that learns to play The Binding of Isaac from
scratch. Bridge game state to Python and actions back. Design the RL
architecture: environment, observation space, action space, reward
function, algorithm, training loop. Add a live terminal for episode count,
reward, and loss.
Training runs detached and must never be stopped by you. I pause and resume
this goal manually; you cannot pause yourself.
Workflow: set up, launch training detached, then output PAUSE on its own
line and stop. I'll pause the goal. When I resume — typically 12–24 hours
later — review progress, report status and any decisions, then either
continue or set up the next run and signal PAUSE again. Repeat.
The goal is complete when the agent can consistently clear the first floor,
including the boss.
```

Trigger: Agent's self-written goal produced constant monitoring with no
handoff point. No signal for "I'm done, pause me," so the workflow had no
way to stop token burn while training ran.

Intervention: Human rewrote the goal. ~35 lines → 9 → ~15, ending with
only mechanism lines: training detached, human controls pause/resume, a
handoff signal fires when the agent has nothing left to do. Research-method
constraints were deliberately left out, those are the judgments the
project exists to test.

Note: First real test of whether goal-level constraints change agent
behavior. Pending.

Meta-finding: The agent wrote a goal as if it would run continuously, with
no representation of being paused. Some of this is harness, see *Note —
Async process handling*. But the agent didn't propose a workaround for the
missing primitive.


## Note — Async process handling (feature gap)

**Context:** When a long training job is running, the intuitive behavior is
to wait, not to keep polling, checking, and burning tokens while nothing
has changed. The harness doesn't support this. Current primitives:

- **Sleep within a running turn.** Codex can wait between checks. The
 sleep itself doesn't generate tokens, but each subsequent model-driven
 check does. Not a durable checkpoint-triggered pause.
- **Pause the goal.** Stops automatic continuation. Requires the trainer
 to run as an independent, detached process so its lifetime doesn't
 depend on the agent. Resume with `/goal resume`.
- **External watcher.** A Python or PowerShell process watches for
 completion, checkpoint, or failure with zero LLM tokens spent waiting.
 It can notify you or launch a bounded Codex review through the CLI. This
 requires custom wiring, and launching a review isn't necessarily the
 same as resuming the paused goal.

**The gap:** No native way to wake the agent when an external condition is
met. Every option is either polling or user-initiated.

**Suggestion for OpenAI:** Add a native hook or watcher primitive so the
agent can wait on an external condition without polling:
- **Event-driven wake-ups.** Let the goal register a condition: file
 changed, process exited, command returned, log line matched, and
 suspend until it fires. No model turns while waiting.
- **True sleep with zero token cost.** Current sleep reduces burn but
 doesn't eliminate it.
- **Signal channel for background processes.** A first-class way for a
 detached process to notify the agent, so the user isn't relaying
 "training finished" into the chat.

**Why it matters for RL:** The canonical loop: train for hours
unattended, wake on a milestone, review, sleep, isn't supported. Every
option is either token-costly polling or user-initiated.
