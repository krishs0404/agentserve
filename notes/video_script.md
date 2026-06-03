# AgentServe — Video Script

**Format**: Screen recording, ~8–9 minutes. Research track.
**Required questions covered**: Q1 (Why), Q2 (How), Q3 (Use cases), Q4 (What more)
**Visuals**: Terminal + 4 plots from `notes/plots/`. No live demo.

---

## [0:00–0:50] Q1: Why Did You Build This?

*Show: GitHub repo homepage or blank terminal*

> "AI agents are becoming the dominant way people use large language models — not single
> questions, but systems that make dozens of LLM calls to complete a task. Code debugging
> agents that read files, run tests, and write patches. Research assistants that fetch
> papers, extract citations, and synthesize answers. Customer service pipelines that
> classify intent, retrieve context, and generate responses — all in one workflow.
>
> The bottleneck I kept running into was this: when an agent fires multiple LLM calls
> at once — five classification calls and one code generation call simultaneously — a
> standard inference engine runs them in arrival order. FIFO. So the five classifiers
> sit blocked behind the expensive code generation for seconds. But those classifiers
> are blocking dependencies. The agent can't take its next step until all five return.
>
> Every LLM inference system I looked at — vLLM, SGLang — treats a stream of agent
> requests the same as a stream of chatbot requests. They don't know that some requests
> are DAG dependencies and some are background work. That's the problem I built
> AgentServe to fix."

---

## [0:50–2:20] Q2: How Does It Work?

*Show: `ls agentserve/engine/` in terminal*

> "AgentServe is a research project implementing a custom LLM inference engine —
> I hand-wrote the Llama 3.2 architecture from scratch, including the attention
> mechanism, paged KV cache, prefix caching, and continuous batching loop — and then
> layered agent-aware scheduling on top of all of it.
>
> The engine has three scheduling layers that work together.
>
> Layer one is the difficulty classifier. Every incoming prompt is classified in under
> a millisecond as easy, medium, or hard. It scans the tail of the prompt — the current
> instruction, not the full conversation history — for signal words. 'Classify this,'
> 'yes or no,' 'true or false' → easy. 'Write a function,' 'implement,' 'refactor' →
> hard. Everything else → medium. The classifier also detects multi-turn conversations
> so that a 20,000-token SWE-bench context doesn't get labeled hard just because it's
> long.
>
> Layer two is the scheduler. Instead of one FIFO queue, three O(1) deques — one per
> difficulty level — so easy requests always drain first. Two additional policies stack
> on: soft overflow admits extra easy requests past the normal batch cap since they exit
> fast, and preemption evicts the youngest hard request when an easy one has been waiting
> too long.
>
> Layer three is trajectory-aware scheduling. For multi-step agent sessions — ReAct
> think-act loops, plan-execute pipelines — the scheduler needs to understand step
> dependencies. Two plug-in policies: TrajectoryProgress prioritizes sessions past their
> midpoint, and TrajectoryDeadline scores by urgency, defined as remaining work divided
> by time remaining.
>
> I also built a sixth mode — relative batching — where instead of three fixed bins,
> an online linear model predicts expected output length for each request and the
> scheduler groups requests with similar predicted lengths together. This reduces
> KV-cache padding waste during the GPU decode step."

---

## [2:20–4:50] Results — Evaluation & Evidence

*Open `notes/plots/`. Walk through in order.*

### Plot 1: `ablation_latency.png` (~50 seconds)

> "Here's the ablation study. Six scheduling modes, same 100-request heterogeneous
> agent workload — 64% easy calls, 27% medium, 9% hard — on Llama 3.2-1B on an NVIDIA
> A10G GPU with Flash Attention 2.
>
> Baseline FIFO: 11.9 seconds mean latency for easy requests. All three policies
> together — mode (d) — cuts that to 8.1 seconds. 32% reduction with zero model changes.
> Just a different order of execution.
>
> Hard request latency goes up. That's the explicit tradeoff — the right tradeoff, because
> hard requests are background work and easy requests are blocking dependencies.
>
> The scheduling benefit isn't fixed at 32%. I ran a sweep varying the workload mix.
> When easy requests are only 3% of traffic — closer to real SWE-bench sessions — the
> benefit jumps to 78%. At 75% easy, it drops to 9%. The benefit scales with how much
> blockage exists."

### Plot 2: `trajectory_speedup.png` (~60 seconds)

> "The more important result. Now I'm measuring trajectory completion time — wall-clock
> time from when an agent starts a multi-step task to when it finishes.
>
> Look at the priority scheduling bar. Zero improvement on ReAct. Slightly negative on
> Plan-Execute. That's the central finding: per-request latency improvement does not
> translate to task completion time when you don't understand step dependencies.
>
> The trajectory-aware policies do. TrajectoryProgress cuts ReAct completion time 6x —
> 100 seconds down to 17. TrajectoryDeadline cuts Plan-Execute 2.6x — a 2.5-minute
> pipeline down to 60 seconds. Average speedup across all templates: 2.7x."

### Plot 3: `sensitivity_sweep.png` (~25 seconds)

> "One concern: the system relies on a classifier. What if it's wrong? I swept noise
> from 0% to 50% — half of all labels randomly flipped. The scheduling benefit went
> from 66% to 65%. It barely moved. The system works through statistical separation,
> not perfect classification."

### Plot 4: `latency_cdf.png` (~25 seconds)

> "And it's not just the mean. The CDF shows the full distribution shifting. FIFO has
> easy requests taking up to 19 seconds at the 95th percentile. With all three policies,
> P95 drops to 14 seconds."

---

## [4:50–5:40] Novel Finding — Relative Batching

*Point to mode (e) on ablation plot*

> "The most unexpected result was mode (e) — relative batching. The idea is GPU
> efficiency: our decode step pads all KV caches to the longest sequence in the batch.
> Pair a 15-token easy request with a 400-token hard request and you're running 400
> positions of attention for both — 97% wasted compute for the short one.
>
> So I built a sliding window scheduler. Instead of three fixed bins, an online linear
> model predicts output length per request and the scheduler minimizes intra-batch
> variance. The result: easy latency barely moves — 3% better than FIFO. But hard
> latency improves 9%. It's a balanced tradeoff rather than a winner-loser tradeoff.
>
> Trying to combine priority ordering with relative batching — mode (f) — produced the
> same result as pure priority. The benefit of relative batching comes from grouping
> across the easy-hard boundary; priority ordering blocks that."

---

## [5:40–6:30] Real Data Validation

> "Everything so far is synthetic. I validated on the lmcache-agentic-traces dataset —
> 787 real multi-turn agent sessions from SWE-bench, GAIA, and WildClaw, with 24,000
> LLM iterations and ground-truth output lengths.
>
> Two findings. First: real workloads look nothing like my synthetic benchmark. My
> synthetic is 60% easy. SWE-bench is 0% easy — almost no classification calls, just
> code-debugging tool calls. For specialized workloads, priority scheduling provides
> no benefit because there are no easy requests to promote.
>
> Second: 669 SWE-bench sessions all share the same 14,000-token system prompt. When I
> replayed them through the engine, the prefix KV cache hit 90% of requests — almost
> every session skipped recomputing that system prompt. On synthetic workloads with
> unique prompts, the hit rate is 0%."

---

## [6:30–7:30] Q3: Who Uses This and Why Does It Matter?

> "There are three concrete deployment scenarios where this matters.
>
> The first is multi-agent orchestration frameworks — systems like LangGraph, CrewAI,
> or AutoGen where a coordinator agent fires dozens of sub-agent LLM calls in parallel.
> Right now every inference server those frameworks hit treats each call as independent.
> AgentServe knows that some of those calls are on the critical path and some are
> background work, and it schedules accordingly.
>
> The second is coding assistants. A code review pipeline might simultaneously run:
> a fast 'is this a syntax error' classifier, a medium 'summarize this function' call,
> and a slow 'suggest a refactor' generation. With FIFO scheduling, the classifier waits
> behind the refactor. With AgentServe, it returns in milliseconds and the UI can
> already highlight the issue while the deeper analysis runs.
>
> The third is research and analysis pipelines — systems that fetch multiple sources,
> extract information from each, and synthesize a final answer. The extraction calls are
> cheap and the synthesis is expensive. AgentServe lets the cheap calls run first so the
> synthesis step has all the context it needs without artificial delay.
>
> The broader impact is on inference efficiency. As agents become the dominant interface
> to LLMs, inference systems optimized for single-turn chat will increasingly be the
> bottleneck. The 6x trajectory speedup this project demonstrates requires zero model
> changes — it's pure infrastructure. That makes it deployable on top of any existing
> serving stack."

---

## [7:30–8:15] Q4: What Would You Add Next?

> "Three directions.
>
> First: a session-state classifier. The current classifier looks at the current
> instruction. It can't predict when a tool result will trigger a long code patch
> because that requires knowing where the agent is in its task — has it found the root
> cause, is it ready to synthesize a fix? Features like turn number, recent error
> signals, and output history from the same session would close this gap. I confirmed
> this limitation empirically: training the classifier on 24,000 real SWE-bench
> iterations achieved only 46% accuracy versus 64% for the keyword heuristic, because
> structural features don't capture task state.
>
> Second: workload-adaptive policy selection. Right now you pick a mode before the
> workload starts. A production system serving mixed traffic should detect the incoming
> request distribution in real time and switch policies automatically. The difficulty
> distribution of arriving requests is a live signal.
>
> Third: full variable-length paged attention. The paged KV pool I implemented
> eliminates per-step Python allocation, but the attention step still pads to the
> longest sequence. A custom Triton kernel using block tables directly would eliminate
> that padding — closing the remaining gap with production systems like vLLM.
>
> The core argument: LLM inference for agents is a different problem from LLM inference
> for chatbots. The workload has structure — heterogeneous bursts, DAG dependencies,
> multi-step sessions — and the scheduler should reflect that. A 6x task completion
> speedup with zero model changes is the evidence that scheduling is an underexplored
> lever in the agent stack."

---

## Recording Checklist

Before recording:
- [ ] `uv run pytest tests/ -q` passes (90 tests)
- [ ] Four plots open: `ablation_latency.png`, `trajectory_speedup.png`, `sensitivity_sweep.png`, `latency_cdf.png`
- [ ] Terminal showing `ls agentserve/engine/` for the architecture section
- [ ] Font 18–20pt, notifications off
- [ ] Have `notes/findings.md` open for exact numbers

**Video question coverage:**
- Q1 Why: [0:00–0:50] — bottleneck identified, inspiration explained
- Q2 How: [0:50–2:20] — full architecture, research track framing
- Q3 Use cases: [6:30–7:30] — three concrete scenarios + societal impact
- Q4 What more: [7:30–8:15] — three concrete future directions

**Key numbers memorised:**
- −32% easy latency (modes b–d vs FIFO)
- −41% at max_tokens=256 (grows with sequence length)
- 6× ReAct TCT speedup, 2.6× Plan-Execute
- 0.99× priority scheduling alone on trajectories (zero benefit)
- 66% → 65% sensitivity sweep (robust to 50% noise)
- 78% benefit at 3% easy workload
- 90% prefix cache hit rate on SWE-bench
- 0% easy requests in SWE-bench (vs 60% synthetic)
- 555 tok/s with torch.compile + PyTorch 2.5 (vs 314 baseline)
