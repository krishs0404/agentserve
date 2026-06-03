# AgentServe — Video Script

**Format**: Screen recording, ~8 minutes. Research talk style.
**Visuals**: Terminal showing code structure + 4 plots. No live demo.
**Tone**: Lead with the problem, explain the insight, show the numbers, be honest about tradeoffs.

---

## [0:00–0:45] Hook — The Problem Nobody Talks About

*Show: GitHub README or the agentserve/ directory in terminal*

> "When people talk about making LLMs faster, they talk about FlashAttention, quantization,
> better kernels. But there's a layer nobody talks about: scheduling. Who runs next? In what
> order? When does a request get its batch slot?
>
> For a chatbot, this doesn't matter much. Every request looks the same — one user, one
> response, done. But for an AI agent — a system that fires dozens of LLM calls to complete
> a task — it matters a lot.
>
> Here's why. When an agent calls tools, it fires them in bursts. Five classification calls,
> two extraction calls, one code generation call — all at once. The classifiers take 20 tokens.
> The code generation takes 800. In a FIFO queue, all five classifiers sit blocked behind the
> code generator. But those classifiers are blocking dependencies — the agent cannot take its
> next step until all five return. So the code generator occupies the batch for seconds while
> the agent stalls waiting for answers it could have had immediately.
>
> That's the problem. AgentServe fixes it."

---

## [0:45–2:15] What AgentServe Is

*Show: `ls agentserve/engine/` in terminal*

> "AgentServe is a hand-written Llama 3.2 inference engine with scheduling policies that
> understand agent workload structure. It implements Flash Attention 2 via PyTorch SDPA,
> a paged KV cache, continuous batching, and prefix caching — and layers agent-aware
> scheduling on top of all of it.
>
> Three scheduling layers work together.
>
> First: a difficulty classifier. Every incoming prompt gets tagged easy, medium, or hard
> in under a millisecond. The classifier looks at the tail of the prompt — the current
> instruction — not the full accumulated conversation history. It detects multi-turn format
> automatically: a 20,000-token SWE-bench context doesn't count as 'hard' just because it's
> long; what matters is what the model is being asked to do right now.
>
> Second: the scheduler. Instead of one FIFO queue, three O(1) deques — one per difficulty
> level. Easy always drains first. Two additional policies layer on: soft overflow admits
> extra easy requests past the normal batch cap since they exit fast, and preemption evicts
> the youngest hard request when an easy one has been waiting too long.
>
> Third: trajectory-aware policies. These schedule at the session level, not the request
> level. For multi-step agent workflows — ReAct loops, plan-execute chains — the scheduler
> needs to understand that steps within a trajectory have sequential dependencies. I'll show
> why this matters when I get to the results.
>
> Six scheduling modes total, from plain FIFO to a relative batching approach that groups
> requests by predicted output length instead of by difficulty bin. Let me show you what
> these actually do."

---

## [2:15–5:00] Results

*Open `notes/plots/` — have all four plots ready to switch between*

### Plot 1: `ablation_latency.png` (~50 seconds)

> "Here's the ablation. Six modes, same 100-request heterogeneous agent workload — 64%
> classification and extraction calls, 27% summaries, 9% code generation — running on
> Llama 3.2-1B on an NVIDIA A10G with Flash Attention 2.
>
> Mode (a) is FIFO: 11.9 seconds mean latency for easy requests. Mode (d), all three
> scheduling policies together, cuts that to 8.1 seconds — a 32% reduction with zero
> model changes, zero kernel changes, just a different order of execution.
>
> Hard request latency goes up, and that's the explicit tradeoff. The agent doesn't care
> if code generation takes a bit longer. It cares that the five classifiers blocking its
> next action return fast.
>
> One thing I want to highlight: this 32% improvement is at 60% easy requests. I also
> swept the workload mix. When easy requests are only 3% of traffic — which is closer to
> what real SWE-bench sessions look like — the improvement jumps to 78%, because each
> easy request is stuck behind far more hard ones. At 75% easy, it drops to 9%. The
> scheduling benefit scales with how much blockage exists.
>
> Mode (e) makes a different tradeoff — I'll come back to it."

### Plot 2: `trajectory_speedup.png` (~60 seconds)

> "The more important result is this one. I'm now measuring trajectory completion time —
> wall-clock time from when an agent starts a multi-step task to when it finishes. Each
> trajectory is 3 or 4 sequential LLM calls with real step dependencies.
>
> Look at the priority scheduling bar — mode (b). Zero improvement on ReAct. Slightly
> negative on Plan-Execute. That's the key finding of this project: per-request latency
> improvement does not translate to trajectory completion time when you don't understand
> the step dependencies. Making individual requests faster doesn't help if you're not
> coordinating across the steps of the same session.
>
> The trajectory-aware policies do. TrajectoryProgress, which prioritizes sessions past
> their midpoint, cuts ReAct completion time 6x — from 100 seconds down to 17 seconds.
> TrajectoryDeadline, which scores by urgency using remaining work divided by time
> remaining, cuts Plan-Execute 2.6x — a 2.5-minute pipeline down to 60 seconds.
>
> Which policy wins depends on the session structure: TrajectoryProgress is better for
> short uniform-step chains, TrajectoryDeadline is better for longer chains where early
> steps are expensive and urgency builds. The average speedup across all templates is
> 2.7x for TrajectoryDeadline, 2.5x for TrajectoryProgress. Priority alone: 0.99x."

### Plot 3: `sensitivity_sweep.png` (~30 seconds)

> "One natural concern: this system relies on a classifier. What happens when the
> classifier is wrong?
>
> I ran a sweep injecting random label noise — flipping requests to the wrong bucket
> at increasing rates. At 0% noise, the scheduling benefit is 66%. At 50% noise — half
> of all labels randomly wrong — the benefit is still 65%. It barely moves.
>
> The reason: scheduling works through statistical separation, not perfect classification.
> A majority of requests in the right bucket is sufficient. The scheduler doesn't need
> to know exactly which requests are easy; it needs enough of them to flow through first."

### Plot 4: `latency_cdf.png` (~25 seconds)

> "The CDF shows it's not just the mean that improves — the whole distribution shifts.
> FIFO has a long tail of easy requests taking 15 to 19 seconds. With all three policies,
> the P95 drops from 19 seconds to 14 seconds. Easy requests escape faster across the
> entire distribution, not just on average."

---

## [5:00–6:15] The Novel Finding — Relative Batching

*Point to mode (e) on the ablation plot*

> "Let me come back to mode (e). This was the result I didn't expect.
>
> The motivation is GPU efficiency. Our decode step pads all KV caches in the batch to
> the length of the longest sequence. If you pair a 15-token easy request with a 400-token
> hard request, you're running the full attention computation for 400 positions for both —
> 97% wasted compute for the short one.
>
> So I built a sliding window scheduler. Instead of three fixed difficulty bins, an online
> linear model predicts the expected output length for each request and updates its weights
> after every completion. The scheduler then picks whichever window of pending requests has
> the smallest variance in predicted length — requests that will finish at roughly the same
> time go into the batch together.
>
> The result: easy latency barely moves, only 3% better than FIFO. But hard latency
> actually improves 9%. Relative batching makes a balanced tradeoff — no winner, no loser,
> it reduces padding waste uniformly across difficulty classes.
>
> Mode (f) tried to combine priority ordering with relative batching within each tier.
> Easy latency matched the priority modes, but hard latency stayed penalized. The reason:
> the benefit of relative batching comes specifically from grouping across the easy-hard
> boundary. Priority ordering prevents that by design. They're complementary mechanisms,
> not additive ones.
>
> The practical implication: priority scheduling is right when you have heterogeneous
> traffic and easy latency is the bottleneck. Relative batching is right for specialized
> workloads — like code debugging agents — where nearly all requests are medium or hard
> and there are no easy requests to promote."

---

## [6:15–7:15] Real Data Validation

> "Everything so far is on a synthetic workload I designed. I wanted to check whether
> real production traffic looked similar, so I integrated the lmcache-agentic-traces
> dataset — 787 real multi-turn agent sessions from SWE-bench, GAIA, and WildClaw, with
> 24,000 LLM iterations and ground-truth output lengths.
>
> Two findings. First, real workloads look nothing like my synthetic benchmark. My
> synthetic is 60% easy, 25% medium, 15% hard. SWE-bench is 0% easy, 60% medium, 40%
> hard — almost no short classification calls, just code-debugging tool calls at every
> turn. The scheduler that's optimal for a general-purpose assistant is not optimal for
> a code agent. For SWE-bench, priority scheduling provides zero benefit because there
> are no easy requests to promote. The right lever there is trajectory-aware scheduling
> and prefix caching.
>
> Second: 669 of those SWE-bench sessions all share the same 14,000-token system prompt.
> I replayed them through the engine. The prefix KV cache achieved a 90% hit rate —
> nearly every session skipped recomputing attention over that system prompt entirely. On
> synthetic workloads with unique prompts, the hit rate is 0%. Real agent deployments have
> massive prefix-sharing opportunities that synthetic benchmarks completely miss."

---

## [7:15–8:00] Future Directions + Close

> "Three directions I'd pursue next.
>
> First: a session-state classifier. The current classifier looks at the tail of the
> current prompt — the most recent instruction. It can't predict when a tool result will
> trigger a long code patch, because that requires knowing where the agent is in its task:
> has it found the root cause, is it ready to synthesize a fix? Features like turn number,
> recent error signals, and output length history from the same session would close this
> gap. This is what I found when I trained the classifier on 24,000 real SWE-bench
> iterations — it underperformed keywords because it lacks that cross-turn context.
>
> Second: workload-adaptive policy selection. Right now you pick a scheduling mode before
> the workload starts. A production system serving mixed traffic — code agents, research
> agents, general assistants — should detect the request distribution in real time and
> switch policies automatically. The difficulty distribution of arriving requests is a
> live signal you can use.
>
> Third: full variable-length paged attention. The current implementation stores KV tensors
> in a pre-allocated pool with block tables, eliminating per-step Python allocation. But
> the attention step still pads to the longest sequence. A custom Triton kernel using the
> block tables directly would eliminate that padding entirely — the remaining gap between
> this engine and vLLM's throughput.
>
> The core argument: LLM inference for agents is a different problem from LLM inference
> for chatbots. Heterogeneous request bursts, DAG dependencies between calls, multi-step
> session structure — these properties don't exist in chatbot traffic, and standard
> schedulers don't account for them. A 6x trajectory speedup with zero model changes is
> the evidence that scheduling is an underexplored lever."

---

## Recording Checklist

Before recording:
- [ ] `uv run pytest tests/ -q` passes (90 tests)
- [ ] Four plots open and clean: `ablation_latency.png`, `trajectory_speedup.png`, `sensitivity_sweep.png`, `latency_cdf.png`
- [ ] Terminal showing `ls agentserve/engine/` ready for the architecture section
- [ ] Font size 18–20pt, notifications off, clean desktop
- [ ] Have `notes/findings.md` open as a reference for exact numbers

Key numbers to have memorised:
- **−32%** easy latency (modes b–d vs FIFO, max_tokens=64)
- **−41%** easy latency at max_tokens=256 (grows with sequence length)
- **6×** ReAct trajectory speedup (traj_progress)
- **2.6×** Plan-Execute speedup (traj_deadline)
- **0.99×** priority scheduling alone on trajectories (no improvement)
- **66% → 65%** sensitivity sweep (0% to 50% noise)
- **78%** scheduling benefit at 3% easy workload, **9%** at 75% easy
- **90%** prefix cache hit rate on real SWE-bench sessions
- **0%** easy requests in real SWE-bench traces (vs 60% synthetic)
