# AgentServe: research walkthrough and discussion guide

## A defensible introduction

“AgentServe is my experimental PyTorch inference engine for investigating how
scheduling affects multi-step agent workloads. I built request and trajectory
policies on top of continuous batching and KV caching, then audited numerical
correctness and challenged the policies' prediction assumptions with recorded
agent traces. In a session-held-out study, keyword length estimates failed to beat
a constant median and gave almost no useful ordering. Previous-turn length had
better ordering despite worse absolute error. The next step is to test that signal
against whole-session completion time and fairness using causal replay.”

That describes implemented systems work and an empirical finding. It does not
claim a state-of-the-art serving engine or validated GPU speedup.

## The reasoning chain

An agent alternates LLM calls with tools and sometimes human waits. Its user sees
the time to finish a task, not the latency of one isolated request. Two schedulers
can improve different metrics: one may complete many short calls quickly while
another reduces the time until entire sessions finish. Neither dominates every
workload or fairness objective.

A simplified sequential session has completion time

`C_session = sum(queue_wait_i + LLM_service_i) + sum(tool_wait_i)`.

The next LLM call becomes ready at `completion(previous_call) + tool_wait`.
Policy changes therefore alter future arrivals. A replay that fixes all turn
arrival timestamps can violate dependencies and answer the wrong question. For
branching workflows, the corresponding object is a critical path through a DAG;
the current trajectory templates represent sequential chains, not general DAGs.

Short-job-first scheduling minimizes average completion time for a restricted
single-server setting with known service times and all jobs ready together.
Those assumptions do not establish optimality for continuous LLM batching,
uncertain output lengths, KV limits, or agent dependencies. Output length is one
component of cost: long prefills and reused prefixes can change service time.

A strong evaluation therefore needs three separate steps:

1. **Numerical validity:** does the engine compute the intended model outputs?
2. **Signal validity:** can available features estimate or order relevant work?
3. **Decision validity:** does using that signal improve the chosen end-to-end
   objective under realistic load, without unacceptable tail/fairness costs?

The current project has bounded numerical checks and a recorded-data signal
study. The end-to-end policy claim remains open.

## What the code implements

A request moves from pending through prefill and decoding to completion. Prefill
processes prompt tokens and produces KV state. Autoregressive decode appends a
new token and its K/V entries. Continuous batching admits new requests while
others decode, rather than waiting for an entire fixed batch to finish.

The Llama implementation includes RMSNorm, grouped-query attention, SwiGLU, and
rotary positions. GQA uses fewer KV heads than query heads, reducing KV storage.
Ordinary KV storage keeps per-request tensors; paged storage allocates fixed-size
blocks. Prefix caching can reuse matching prompt KV, but a matching hash alone
does not establish a measured GPU speedup, and truncated prompt text cannot
establish a production cache hit rate.

Scheduler variants include FIFO, keyword priority, batch overflow, preemption,
and length-relative batching. The pluggable interface also includes trajectory
progress and deadline heuristics. Progress assumes total step count is known;
real agents often do not know it. Deadline urgency relies on approximate remaining
work. The implementations are mechanisms for experiments, not established optimal
algorithms. Deadline keys are now refreshed before admission because insertion-time
ordering becomes stale while requests wait. This adds sorting overhead that
needs to be measured under load.

Overflow uses more batch capacity, so a comparison must report actual batch size
and memory alongside latency. Preemption has a re-prefill/cache cost. Relative
batching can reduce padding but also delay an otherwise ready request. Strict
short-request priority can penalize long requests; a finite workload completing
successfully does not prove starvation resistance under sustained arrivals.

## Correctness work and its boundaries

The earlier audit fixed attention-head flattening and mixed-length KV updates.
Both defects can yield plausible outputs while corrupting numerical results.
Tests now compare logits and KV contents across repeated steps, unequal lengths,
and ordinary/paged storage, rather than checking output shapes alone.

Independent tiny Hugging Face comparisons use identical random weights. The new
scaled-RoPE tests compare rotations across frequency bands and long absolute
positions, then compare repeated greedy decode around position 8192. These use
short sequences at large position offsets: they exercise positional scaling,
not the performance or memory behavior of a genuinely 8K-token context.

Llama 3.2 uses frequency-dependent RoPE scaling. The earlier theta-only
implementation omitted it. Scaling is now implemented; checkpoint configuration
mismatches are rejected, and a missing output projection is filled from embeddings
only for checkpoints declaring tied embeddings. The local checkpoint validator
compares float32 logits and greedy tokens against Hugging Face, using identical
token IDs and mixed request lengths.

Access to the intended pretrained checkpoint is gated. The validator passed a
synthetic saved-checkpoint smoke test, which is not pretrained validation. CPU
parity does not establish CUDA parity; successful finite tests do not prove
correctness for every sequence or dtype. No new GPU performance result is claimed.

## What the trace study establishes

The local extract has 24,880 recorded turns from 767 sessions and four models.
Its prompts are truncated last user/tool messages, not full model contexts. Five
80/20 splits group by session to avoid putting adjacent turns in both training
and test. The exact split IDs, dataset hash, and raw predictions are committed.

The constant median uses training labels only. The keyword heuristic uses fixed
20/100/256-token estimates. Previous-turn length uses only an earlier output in
the same session, with the training median for a session's first turn. This is
causal online evaluation, assuming within-session file order is correct.

Mean request MAE: median 144.49, keywords 147.66, previous turn 164.43 tokens.
Session macro MAE weights every session equally. A session-level paired bootstrap
accounts for dependence between turns within a session, conditional on each split.
It does not correct source selection bias, related tasks across sessions, or
correlation across the five overlapping splits.

For sampled cross-session pairs, previous-turn length correctly orders about
63.36%, versus 49.48% for keywords with half credit for prediction ties. Keyword
estimates tie on 88.71% of evaluated pairs. Previous-turn predictions can be noisy
in magnitude yet retain useful relative order. Neither result proves these pairs
would overlap in a live queue or that scheduling by them would improve sessions.

This is one narrow, reproducible finding. It suggests testing a cheap temporal
feature and separating calibration metrics from ranking metrics. It does not
justify deploying previous-turn priority as a proven improvement.

## Questions to prepare for

**What is novel compared with vLLM or SGLang?**
The project is a small experimental substrate, not a competing production engine.
The research question is whether session information changes admission decisions
and task completion under realistic agent dependencies. A compelling contribution
would be a controlled finding or policy evaluated against strong baselines; that
has not yet been established for end-to-end serving.

**Why implement an engine instead of integrating with an existing one?**
It makes admission, KV lifetime, and numerical state inspectable and easy to
modify. The tradeoff is substantial validation burden and limited performance
representativeness. A later integration would test whether the result survives a
production runtime; the current prototype is useful for controlled exploration.

**Why not use shortest remaining processing time?**
Remaining output is unknown, and token count alone omits prefill and KV costs.
Use an oracle only as a diagnostic bound within a specified simulator/service
model, then compare realistic predictors without leaking future labels.

**Does lower MAE mean a better scheduler?**
No. The median is strong for absolute error but cannot order requests. The current
previous-turn result illustrates the distinction. Measure ordering on actual
contending sets, then session latency and fairness after making decisions.

**Why split by session? Could leakage remain?**
Adjacent turns share state and outputs; random turn splitting exaggerates
independence. Grouping by session prevents that direct leakage. Related tasks,
repositories, model families, and collection time can still overlap. Task-family,
chronological, and model-held-out splits are useful robustness checks.

**Why not claim tool-inclusive completion time already?**
The local pairs lack gaps, and old converted arrivals assume two seconds of
generation. Neither supports causal session timing. New conversion retains source
order, predecessor indices, and original gaps, but old committed traces are not
regenerated and no causal runner has produced a result yet.

**What if tool time dominates?**
Reducing queue time may have little relative effect on total completion. Compare
with and without tool waits, report absolute seconds, and sweep load. At light
load a work-conserving admission policy may make little difference.

**Who loses when short requests win?**
Long requests can have worse waiting time and slowdown. Report long-job p95/p99,
maximum wait, per-session completion, and slowdown versus isolated service time.
Test sustained bursts; consider explicit aging or a bounded-wait policy and
measure its efficiency cost. Do not claim existing variants provide a universal
fairness guarantee.

**How would you design the next experiment?**
Validate a supported pretrained checkpoint first. Preserve per-session causal
order and tool gaps. Pair identical initial arrivals and work across FIFO,
previous-turn priority, keywords, and a carefully labeled oracle. Sweep arrival
burstiness, concurrency, context growth, prefix reuse, and length heterogeneity.
Use warmup and repeated seeds, preserve raw per-request/session events, and report
uncertainty plus tail penalties. A simulator needs an explicit uncalibrated service
model label until hardware measurements support it.

**What could falsify your hypothesis?**
If useful ranking fails to improve whole-session completion at realistic load, if
prefill/batching effects erase benefits, or if gains require unacceptable long-job
penalties, the policy is not justified. Publishing that outcome is preferable to
claiming improvement from synthetic easy-request latency alone.

**Which work is yours and which used AI?**
State the documented AI assistance openly. Be ready to derive the latency model,
explain the KV bug and RoPE correction, reproduce a split, and defend each metric.
Ownership is demonstrated by understanding and verification; do not imply sole
manual implementation or unaided analysis.
