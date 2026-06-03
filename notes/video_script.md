# AgentServe — Video Script

**Format**: Screen recording, ~8 minutes. Research talk style.  
**Visuals**: Terminal + 4 plots from `notes/plots/`. Run `demo.py` live for ~30 seconds.  
**Tone**: Explain the *why* before the *how*. No product hype.

---

## [0:00–0:45] Hook — The Problem Nobody Talks About

*Show: blank terminal or GitHub README in browser*

> "When people talk about making LLMs faster, they talk about FlashAttention, CUDA
> graphs, quantization. But there's a layer nobody talks about: scheduling. Who runs
> next? In what order? When does a request get its batch slot?
>
> For a chatbot, this doesn't matter much. Requests look the same — one user, one
> response, done. But for an agent — a system running dozens of LLM calls to complete
> a task — it matters enormously. Here's why.
>
> When an agent fires tool calls, it fires them in bursts. Five classification calls,
> two extraction calls, one code generation call — all at once. The classify calls take
> 20 tokens each. The code generation takes 800 tokens. In a standard FIFO queue, the
> five classifiers sit blocked behind the code generator. But those classifiers are
> blocking dependencies — the agent literally cannot take its next step until all five
> come back. So your code generator occupies the batch for 12 seconds while the agent
> stalls."

---

## [0:45–2:00] What AgentServe Is

*Show: `ls agentserve/engine/` in terminal, then talk through the layers*

> "AgentServe is a custom inference engine built around this insight. It's a
> hand-written Llama 3.2 implementation — attention, paged KV cache, continuous
> batching — layered with scheduling policies that understand agent workload structure.
>
> Three layers work together. First: a difficulty classifier. Every prompt gets
> classified in under a millisecond — easy, medium, or hard — based on keyword
> analysis and output-length prediction. The classifier detects multi-turn
> conversations and focuses on the tail of the prompt where the current instruction
> lives, not the accumulated history.
>
> Second: the scheduler. Instead of one FIFO queue, there are three O(1) deques —
> one per difficulty level. Easy requests always drain first. Two additional policies
> layer on top: soft overflow admits extra easy requests beyond the normal batch cap,
> and preemption evicts the youngest hard request if an easy one has been waiting too
> long.
>
> Third: trajectory-aware policies. For multi-step agent workflows — ReAct loops,
> plan-execute chains — two plug-in policies schedule at the session level, not just
> the request level. I'll come back to those when I show the results."

---

## [2:00–2:30] Architecture Walkthrough

*Show: `ls agentserve/engine/` in terminal or a simple ASCII diagram*

> "Here's the engine structure. The entry point is `engine.py`, which ties three
> layers together.
>
> Layer one: the difficulty classifier in `difficulty.py`. Inspects the tail of
> the prompt for keywords. Tags each request easy, medium, or hard in under a
> millisecond. Detects multi-turn conversation format and focuses on the most recent
> instruction, not the full accumulated history.
>
> Layer two: the scheduler in `scheduler.py`. Six modes. The simplest is three O(1)
> FIFO deques — one per difficulty level. Easy drains first. Soft overflow admits extra
> easy requests past the batch cap. Preemption evicts the youngest hard request when
> an easy one has waited too long. For specialized workloads, a sliding-window
> relative batching mode groups requests by predicted output length instead of by
> difficulty bin.
>
> Layer three: pluggable trajectory policies in `policies.py`. Two methods — priority
> key and on-complete. Drop in any policy object; the scheduling loop doesn't change.
> I'll show why this design matters when I get to the trajectory results."

---

## [2:30–5:00] Results

*Open `notes/plots/`. Walk through in order.*

### Plot 1: `ablation_latency.png` (~45 seconds)

> "Here's the ablation. Six scheduling modes, same 100-request agent workload on a
> real Llama 3.2-1B on an NVIDIA A10G GPU with Flash Attention 2.
>
> Mode (a) is FIFO — 11.9 seconds mean latency for easy requests. Mode (d), all three
> policies, cuts that to 8.1 seconds — a 32% reduction, just from scheduling. No model
> changes, no kernel changes, just a different order of execution.
>
> Hard request latency goes up, which is the explicit tradeoff. The agent doesn't care
> if code generation takes a bit longer — it cares that the five classifiers blocking
> its next step return fast.
>
> One thing worth noting: this 32% improvement is measured at 60% easy requests. I also
> ran a sweep varying the workload mix. When easy requests are rare — 3% of the workload
> — the improvement jumps to 78%, because each easy request is blocked behind more hard
> ones. At 75% easy, it drops to 9%. The scheduling benefit is proportional to how much
> blockage exists.
>
> Mode (e) is something different — I'll come back to it in a minute."

### Plot 2: `trajectory_speedup.png` (~60 seconds)

> "The more interesting result is this one. Here I'm measuring trajectory completion
> time — the wall-clock time from when an agent starts a multi-step task to when it
> finishes. Each trajectory is 3-4 sequential LLM calls with real step dependencies.
>
> Plain priority scheduling — mode (b) — does almost nothing. Zero improvement on
> ReAct, slightly negative on Plan-Execute. That's the key finding: per-request
> latency improvement doesn't translate to trajectory completion time if you don't
> understand the step dependencies. Scheduling individual requests faster doesn't help
> if you're not coordinating across the steps of the same task.
>
> The trajectory-aware policies do. TrajectoryProgress — which prioritizes sessions
> past their midpoint — cuts ReAct completion time 6x. 100 seconds down to 17 seconds.
> TrajectoryDeadline — which scores by urgency — cuts Plan-Execute 2.6x. A 4-step
> pipeline that used to take 2.5 minutes now finishes in 60 seconds."

### Plot 3: `sensitivity_sweep.png` (~30 seconds)

> "One concern with a classifier-based system is: what happens when the classifier
> is wrong? I injected random noise — flipping labels to the wrong bucket at various
> rates.
>
> At 0% noise, the scheduling benefit is 66%. At 50% noise — when half of all requests
> are randomly misclassified — the benefit is still 66%. The scheduling policy is
> robust to classifier errors because the benefit comes from statistical separation,
> not perfect classification. A majority of requests in the right bucket is sufficient."

### Plot 4: `latency_cdf.png` (~30 seconds)

> "And the CDF shows it's not just the mean that shifts — the whole distribution
> moves. FIFO has a long tail of easy requests taking 15-19 seconds. With all three
> policies, the 95th percentile drops to 14 seconds. Easy requests escape faster
> across the entire distribution."

---

## [5:00–6:15] The Novel Finding — Relative Batching

*Show ablation_latency.png, point to mode (e) and (f) columns*

> "Now let me come back to mode (e). This was the most interesting result.
>
> The insight is about GPU efficiency. Our decode step pads all KV caches in the batch
> to the length of the longest sequence. If you pair a 15-token easy request with a
> 400-token hard request, you're running a matrix multiply of size 400 for the short
> request — 97% waste. The short request finishes in 15 steps; the remaining 385 steps
> just waste its slot.
>
> So I built a sliding window scheduler. Instead of three fixed bins, you predict a
> continuous output length for each request using an online linear model that updates
> after every completion. The scheduler picks whichever group of pending requests has
> the smallest variance in predicted length. Requests that will finish at the same
> time go in together.
>
> The result was a surprise. Easy latency barely moves — only 3% better than FIFO. But
> hard latency actually improves 9% versus FIFO. Relative batching makes a completely
> different tradeoff than priority scheduling: instead of picking a winner and loser,
> it reduces padding waste uniformly.
>
> Mode (f) tried to combine both — priority ordering between tiers, relative batching
> within each tier. Easy latency matched the priority modes, but hard latency didn't
> recover. The reason: the benefit of relative batching comes from grouping across the
> easy/hard boundary. Priority ordering prevents that by design.
>
> These aren't substitutes. Priority scheduling is right for heterogeneous workloads
> where easy latency is the bottleneck. Relative batching is right for specialized
> workloads — like SWE-bench — where nearly all requests are medium or hard, so there
> are no easy requests to promote."

---

## [6:15–7:15] Real Data — Not Just Synthetic

> "Everything so far was on a synthetic workload I designed with the right distribution
> to make scheduling differences visible. I wanted to check whether real production
> traffic looked similar.
>
> I integrated the lmcache-agentic-traces dataset — 787 multi-turn agent sessions from
> real SWE-bench, GAIA, and WildClaw agent runs, with 24,000 LLM iterations and
> ground-truth output lengths.
>
> Two findings. First: real workloads look nothing like my synthetic benchmark. My
> synthetic was 60% easy, 25% medium, 15% hard. SWE-bench is 0% easy, 60% medium,
> 40% hard. Almost no short classification calls — just code-debugging tool calls at
> every turn. The scheduler that's right for a general-purpose agent isn't the same
> one that's right for a code agent.
>
> Second: 669 SWE-bench sessions all share the same 14,000-token system prompt. I
> replayed those sessions through the engine. The prefix KV cache achieved a 90% hit
> rate — almost every session skipped recomputing that system prompt entirely. On
> synthetic workloads where all prompts are unique, hit rate is 0%. Real deployments
> have massive prefix-sharing opportunities that synthetic benchmarks miss entirely."

---

## [7:15–8:00] Future Directions + Close

> "Three directions I'd pursue next.
>
> First: a session-state classifier. The current classifier looks at the tail of the
> prompt — the most recent instruction. But it can't predict when a tool result will
> trigger a long code patch, because that depends on where the agent is in its task:
> has it found the root cause? Is it ready to write? A classifier with features like
> turn number, recent error signals, and intermediate output lengths would close this
> gap.
>
> Second: workload-adaptive scheduling. Right now you pick a scheduling mode and run
> with it. A production system serving mixed traffic — some code agents, some research
> agents, some general assistants — should detect the workload distribution and switch
> policies automatically. The difficulty distribution of arriving requests is a real-
> time signal you can use.
>
> Third: FlashAttention integration. The scheduling improvements — 32% easy-latency
> reduction, 6x trajectory speedup — are orthogonal to kernel throughput. This engine
> runs hand-written PyTorch attention, which is about 16x slower than vLLM on raw
> throughput. Stack these scheduling policies on top of production kernels and the
> relative gains are the same.
>
> The core argument of this project is simple: LLM inference for agents is not the
> same problem as LLM inference for chatbots. The workload structure is different —
> heterogeneous bursts, DAG dependencies, multi-step sessions — and the scheduler
> should reflect that. A 6x trajectory speedup with zero model changes makes the case
> that scheduling is an underexplored lever."

---

## Recording Checklist

Before recording:
- [ ] `uv run pytest tests/ -q` passes (84 tests)
- [ ] `uv run python scripts/demo.py` renders the side-by-side display
- [ ] `notes/plots/ablation_latency.png`, `trajectory_speedup.png`, `sensitivity_sweep.png`, `latency_cdf.png` open and look clean
- [ ] Terminal font size increased to 18-20pt for readability
- [ ] Close Slack, email notifications, etc.

Suggested recording order:
1. Record the live `demo.py` section first — takes the most takes
2. Record the hook and system overview
3. Record the results walkthrough with plots pre-opened
4. Edit: cut 30 seconds from each plot section if running long
