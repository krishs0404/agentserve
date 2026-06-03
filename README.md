# AgentServe

**A custom LLM inference engine built for agents, not chatbots — with scheduling policies that understand the structure of agentic workloads.**

---

## The Problem

When an AI agent runs, it fires **heterogeneous bursts** of LLM calls simultaneously: five `classify()` calls, two `extract()` calls, and one `write_code()` call, all at once. A standard FIFO scheduler runs these in arrival order — so the five classifiers sit blocked behind the expensive code-generation call for hundreds of decode steps.

The classifiers are **blocking dependencies** in the agent's DAG. Every extra millisecond they wait is a millisecond the agent can't proceed. The code-generation call is background work — it doesn't unblock anything until it finishes.

Standard inference engines (vLLM, SGLang) don't know this. They see a stream of tokens, not a stream of agent tasks. AgentServe does.

---

## What AgentServe Builds

A complete inference engine — PyTorch attention, paged KV-cache, continuous batching, prefix cache — layered with scheduling policies that exploit agent workload structure.

### Difficulty Classifier

Every incoming prompt is classified in under 1ms before it reaches the scheduler. Two implementations:

**Keyword heuristic** (`difficulty.py`) — scans the tail of the prompt (last 800 chars) for signal words. Detects multi-turn conversation format and skips the length-threshold check for accumulated context, since in multi-turn agents later turns have longer prompts but often shorter responses.

**Learned predictor** (`length_predictor.py`) — an 8-feature linear model that predicts expected output token count as a continuous value (not just easy/medium/hard buckets). Initial weights are hand-coded from prior findings; online SGD updates them after every completed request. The predictor adapts to the workload being served — a cluster of SWE-bench sessions will shift its distribution toward medium/hard after a few hundred requests.

### Scheduling Policies

Six modes, each additive on top of the previous:

| Mode | Mechanism |
|---|---|
| **(a) Baseline FIFO** | Strict arrival order. The baseline everything is compared against. |
| **(b) Priority only** | Three O(1) FIFO deques — easy, medium, hard. Scheduler always drains easy first, then medium, then hard. Within a tier, arrival order is preserved. |
| **(c) Priority + Overflow** | When the decode batch is at capacity and an easy request arrives, admit it anyway up to 1.25× the batch cap. An easy request exits in ~20 tokens; the slot cost is negligible. |
| **(d) All 3 Policies** | Adds preemption: if all batch slots are occupied by hard requests and an easy request has waited, evict the youngest hard request (fewest output tokens, lowest re-prefill cost) back to pending. |
| **(e) Relative Batching** | Replaces the three-bin approach entirely. The `OutputLengthPredictor` assigns each request a continuous predicted token count ŷ. A sliding window over the pending queue picks whichever group of requests has the smallest intra-batch variance in ŷ — requests that will finish at roughly the same time go in together, reducing KV-padding waste during decode. An age penalty prevents starvation. |
| **(f) Priority + Relative** | **The combined approach.** Keeps the three-deque priority ordering (easy before medium before hard) but applies relative batching *within* each tier. Easy requests still unblock agent DAGs first, but within the easy tier the scheduler picks the requests whose predicted lengths cluster most tightly, minimising per-step KV-padding waste without sacrificing the priority bias. |

### Trajectory-Aware Scheduling

For multi-step agent workflows (ReAct loops, plan-execute pipelines, reflection chains), two pluggable policies operate at the trajectory level rather than the request level:

**`TrajectoryProgressPolicy`** — prioritizes trajectories past their midpoint. A trajectory on step 3 of 3 is scheduled before one on step 1 of 3, even if the latter arrived earlier. Finishing near-complete trajectories frees batch slots faster.

**`TrajectoryDeadlinePolicy`** — schedules by urgency = remaining output tokens / time remaining until deadline. Deadlines are set at submission proportional to estimated serial completion time. As time passes, urgency increases; near-deadline trajectories jump to the front.

Both implement the `SchedulerPolicy` ABC — adding a new trajectory policy is one class with two methods.

### Paged KV-Cache + Prefix Cache

The block allocator manages KV memory in fixed-size blocks (default: 16 tokens). The prefix cache stores KV tensors for completed prefills keyed by a chained block hash, with LFU eviction. On a cache hit, the engine seeds `req.kv_cache` with the stored tensors — the model's forward pass skips recomputing attention over the matched prefix entirely.

On real multi-turn agent traces (669 SWE-bench sessions sharing a ~14K-token system prompt), the prefix cache achieves a **90% hit rate**, eliminating redundant prefill computation for the shared context.

---

## Results

All GPU benchmarks run on **Llama 3.2-1B** on a single **NVIDIA A10G** (24 GB).

### Ablation: per-request latency

Workload: 100 synthetic agent requests — 64 easy (classification/extraction), 27 medium (summary/explanation), 9 hard (code generation). All requests compete for the same batch slots simultaneously.

| Mode | Easy lat | Hard lat | Throughput | TTFT |
|---|---|---|---|---|
| (a) Baseline FIFO | 11.89 s | 10.91 s | 314 tok/s | 9.11 s |
| (b) Priority only | **8.01 s** | 19.30 s | 321 tok/s | 8.59 s |
| (c) Priority + Overflow | **8.11 s** | 19.07 s | **332 tok/s** | 8.13 s |
| (d) All 3 Policies | **8.11 s** | 19.13 s | 331 tok/s | 8.15 s |
| (e) Relative Batching | 11.57 s | **9.91 s** | 317 tok/s | 8.77 s |
| (f) Priority + Relative | **8.09 s** | 19.51 s | 315 tok/s | 8.66 s |

**Key findings:**

Modes (b)–(d) cut easy-request latency **~32%** (11.9 s → 8.0 s) at the cost of making hard requests wait ~82% longer. This is the right tradeoff when easy requests are blocking agent DAG dependencies.

Mode (e) makes a fundamentally different tradeoff: easy latency barely changes (−3%) but hard latency improves 6% vs FIFO. Relative batching reduces KV-padding waste uniformly across difficulty classes — it doesn't pick winners and losers.

Mode (f) matches priority mode on easy latency (8.09 s, same as modes b–d) but doesn't recover hard latency. The within-tier relative batching provides no additional benefit because the predictor's estimates already cluster within each tier — easy requests all predict 20–40 tokens, so the sliding window selects essentially the same candidates FIFO would. The latency benefit of mode (e) comes specifically from grouping *across* the easy/hard boundary (packing easy+easy+easy batches), which mode (f)'s priority ordering prevents by design.

**The practical takeaway:** these are complementary mechanisms, not substitutes. For heterogeneous agent workloads where easy latency is the bottleneck, use mode (d). For workloads where the difficulty distribution is relatively uniform (like SWE-bench: 0% easy, 60% medium, 40% hard), use mode (e).

### Trajectory completion time

20 trajectories per template, all competing simultaneously. TCT = wall time from first step submission to last step completion.

| Policy | ReAct (3-step) | Plan-Execute (4-step) | Reflect (3-step) | Chat (4-turn) |
|---|---|---|---|---|
| FIFO | 99.6 s | 156.3 s | 128.4 s | 169.6 s |
| Priority | 100.0 s (+0%) | 158.8 s | 129.5 s | 173.2 s |
| traj_progress | **16.7 s (6.0×)** | 127.5 s (1.2×) | **75.3 s (1.7×)** | 162.4 s |
| traj_deadline | **16.7 s (6.0×)** | **60.1 s (2.6×)** | 112.3 s | 156.7 s |

The critical finding: **per-request priority scheduling gives zero benefit for trajectory completion time.** It improves individual request latency but doesn't understand step dependencies. Trajectory-aware policies cut ReAct TCT 6× and Plan-Execute TCT 2.6×.

Which trajectory policy wins depends on the template: `traj_progress` is best for short 3-step chains (front-load near-complete trajectories), `traj_deadline` is best for longer 4-step chains with uneven token budgets (urgency scoring).

### Classifier robustness

The scheduling benefit is robust to classifier errors. Using the mock model with artificially injected noise (random label flips at various rates):

| Classifier noise | Easy-latency improvement vs FIFO |
|---|---|
| 0% (perfect) | +66% |
| 10% | +67% |
| 20% | +63% |
| 30% | +64% |
| 50% (random) | +66% |

The benefit stays above 60% even when half of all labels are randomly assigned. This is because the scheduling advantage comes from *statistical* separation of easy and hard — a majority of requests landing in the right bucket is sufficient.

### Real agent traces: prefix cache

Evaluated on 50 sessions from the [lmcache-agentic-traces](https://huggingface.co/datasets/sammshen/lmcache-agentic-traces) dataset (669 SWE-bench sessions, 85 GAIA sessions, 10 WildClaw sessions):

| Workload | Prefix cache hit rate |
|---|---|
| Synthetic (unique prompts) | 0% |
| SWE-bench real traces (shared system prompt) | **90%** |

The SWE-bench sessions all share the same ~14K-token system prompt. After the first session caches the prefix KV tensors, every subsequent session hits them and skips recomputing attention over that prefix.

The real traces also reveal that production agent workloads have a very different difficulty distribution from synthetic benchmarks:
- SWE-bench: 0.1% easy / 60% medium / 40% hard
- GAIA: 1% easy / 69% medium / 30% hard
- Synthetic benchmark: 60% easy / 25% medium / 15% hard

For specialized workloads (all code debugging or all research), the primary scheduling win is from prefix caching and trajectory-aware policies — not per-request priority ordering (there are almost no easy requests to promote).

---

## Architecture

```
Incoming request
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│                    AgentServe Engine                     │
│                                                          │
│  ┌──────────────────┐      ┌──────────────────────────┐ │
│  │ DifficultyClassi-│      │    OutputLengthPredictor  │ │
│  │ fier (keywords + │      │  (8-feature linear model, │ │
│  │ multi-turn detect│      │   online SGD updates)     │ │
│  └────────┬─────────┘      └────────────┬─────────────┘ │
│           │ priority (0/1/2)            │ ŷ (continuous) │
│           │                             │                 │
│           ▼                             ▼                 │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Agent-Aware Scheduler                   │ │
│  │                                                      │ │
│  │  Mode (a): Single FIFO deque                        │ │
│  │  Mode (b): Three priority deques (O(1) enqueue)     │ │
│  │  Mode (c): + Soft overflow for easy requests        │ │
│  │  Mode (d): + Preempt youngest hard request          │ │
│  │  Mode (e): Sliding-window variance minimization     │ │
│  │            (flat queue, groups by predicted length)  │ │
│  │  Mode (f): Priority deques + within-tier relative   │ │
│  │            batching (best of both)                  │ │
│  │                                                      │ │
│  │  Trajectory path (pluggable SchedulerPolicy):       │ │
│  │    TrajectoryProgressPolicy — prefer past-midpoint  │ │
│  │    TrajectoryDeadlinePolicy — urgency scheduling    │ │
│  └──────────────────────┬──────────────────────────────┘ │
│                          │ prefill + decode batches        │
│                          ▼                                 │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Prefix Cache (LFU eviction)             │ │
│  │   Block-granular hash → stored KV tensors           │ │
│  │   On hit: seeds req.kv_cache, skips recompute       │ │
│  └──────────────────────┬──────────────────────────────┘ │
│                          │                                 │
│                          ▼                                 │
│  ┌─────────────────────────────────────────────────────┐ │
│  │   Llama 3.2 (PyTorch, fp16)                         │ │
│  │   RMSNorm · RoPE · GQA · SwiGLU                     │ │
│  │   Batched prefill + batched decode                   │ │
│  │   Paged KV-cache block allocator                     │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## Running It

```bash
# Install dependencies
uv sync

# Run all tests on CPU (no GPU required)
uv run pytest

# Mock model benchmark — all 6 scheduling modes, instant results
uv run python scripts/bench_ablation.py --use-mock --num-requests 40

# Full ablation on real model (GPU required)
uv run python scripts/bench_ablation.py \
    --model-dir /path/to/Llama-3.2-1B-Instruct \
    --num-requests 100 \
    --output-json notes/results.json

# Trajectory benchmark (mock, fast)
uv run python scripts/bench_trajectories.py --n-traj 20 --out notes/plots/

# Trajectory benchmark on real model
uv run python scripts/bench_trajectories.py \
    --model-dir /path/to/Llama-3.2-1B-Instruct \
    --n-traj 20 --max-batch 4 --out notes/plots/

# Generate the 4 benchmark plots
uv run python scripts/plot_all.py

# Run everything on Modal A10G (~$0.60, results land in notes/)
modal run scripts/run_modal.py
modal run --detach scripts/run_modal.py   # detached mode

# Convert lmcache-agentic-traces dataset for trace replay
uv run python scripts/convert_lmcache_traces.py --n-sessions 100

# Replay real agent traces through the engine
uv run python scripts/bench_agent_trace.py \
    --trace traces/lmcache_100.jsonl --compare

# Train the learned output-length classifier
uv run python scripts/train_classifier.py \
    --real-pairs notes/lmcache_training_pairs.jsonl
```

---

## Project Structure

```
agentserve/
  model/
    config.py            ModelConfig dataclass; TinyConfig (tests), Llama32_1B/3B/8B
    llama.py             Llama 3.2: RMSNorm, RoPE, GQA attention (fp16), SwiGLU FFN,
                         batched decode (forward_decode_batch)
    loader.py            Load HuggingFace safetensors; handles Llama 3.2 weight tying
  engine/
    request.py           Request dataclass + lifecycle (PENDING→PREFILL→DECODE→DONE)
    difficulty.py        Keyword heuristic classifier → easy/medium/hard + priority int;
                         multi-turn detection so long conversation contexts don't
                         misclassify as hard
    length_predictor.py  Online output-length predictor: 8-feature linear model with
                         SGD updates after every completed request
    learned_difficulty.py  Batch-trained linear classifier with online calibration and
                           noise injection for sensitivity analysis
    cache.py             Paged KV-cache block allocator (logical bookkeeping)
    prefix_cache.py      Prefix cache with LFU eviction; stores actual KV tensors
    scheduler.py         Continuous-batching scheduler: 3-deque path, relative-batching
                         path, combined path, and pluggable SchedulerPolicy path
    policies.py          SchedulerPolicy ABC; Fifo, Priority, TrajectoryProgress, Deadline
    trajectory.py        TrajectorySpec generator: react / plan_execute / reflect / chat
    sampling.py          Temperature / top-k / top-p sampling; fp16-safe
    engine.py            Main step loop tying all components; wires predictor updates
  server/
    app.py               FastAPI OpenAI-compatible server
scripts/
  bench_ablation.py      6-mode ablation (a: FIFO through f: Priority+Relative)
  bench_trajectories.py  Trajectory TCT benchmark: 4 policies × 4 templates
  bench_agent_trace.py   Replay a real or synthetic agent trace
  plot_all.py            Generate the 4 focused benchmark plots
  run_modal.py           Modal A10G benchmark runner (ablation + trajectory)
  train_classifier.py    Train + evaluate learned classifier; run sensitivity sweep
  convert_lmcache_traces.py  Convert lmcache-agentic-traces for replay and training
  generate_synthetic.py  Synthetic 50-request agent trace
traces/
  synthetic_50.jsonl        Pre-generated synthetic trace
  lmcache_50.jsonl          50 real SWE-bench sessions (from lmcache dataset)
  lmcache_85_gaia.jsonl     85 GAIA research sessions
tests/
  test_scheduler.py      Priority ordering, overflow, preemption, relative batching
  test_cache.py          Block allocator alloc/free/fragmentation
  test_prefix_cache.py   Prefix matching, LFU eviction, KV tensor storage
  test_difficulty.py     Classifier on known prompts, multi-turn detection
  test_engine_mock.py    End-to-end with MockModel (CPU)
notes/
  plots/                 Generated benchmark plots
    ablation_latency.png     Easy vs hard latency across all 6 modes
    trajectory_speedup.png   TCT speedup over FIFO per policy × template
    sensitivity_sweep.png    Scheduling benefit vs classifier noise rate
    latency_cdf.png          Easy-request latency CDF (full distribution shift)
  results_*.json         Raw benchmark output (per-mode stats + latency arrays)
```

---

## Design Decisions

**Attention implementation: `F.scaled_dot_product_attention` (Flash Attention 2)**
Both attention paths use PyTorch's SDPA, which automatically dispatches to Flash Attention 2 on CUDA with PyTorch 2.4+. This gives production-grade efficiency with no custom kernel code. The causal mask and GQA expansion are handled in PyTorch; SDPA fuses the softmax and matmuls into a single GPU kernel. The scheduling policies sit above the attention layer and are independent of it — any future kernel upgrade (Triton, cuDNN, etc.) does not change the scheduling architecture.

**Why a heuristic classifier rather than a purely learned one?**
The classifier runs synchronously before scheduling and must add zero latency. The keyword heuristic takes microseconds. The learned predictor (`length_predictor.py`) is complementary: it provides continuous estimates for the relative-batching mode, and improves with every completion via online SGD. The heuristic remains the primary scheduler signal.

**Why does relative batching improve hard-request latency but not easy-request latency?**
Relative batching groups requests by predicted output length — it reduces KV-padding waste uniformly across all difficulty classes. It doesn't give any request priority over others. Priority scheduling explicitly advantages easy requests at the cost of hard ones. The two mechanisms are complementary: mode (f) applies priority ordering first, then relative batching within each tier.

**Why pluggable trajectory policies?**
The `SchedulerPolicy` ABC decouples policy logic from the scheduling loop. `priority_key()` defines ordering, `on_request_complete()` updates state. Adding a new policy is one class. The three-deque logic is unchanged when `policy=None`.

**Why does per-request priority give zero benefit for trajectory completion time?**
A trajectory's TCT is determined by the slowest step in its chain, and each step can only start after the previous one returns. Priority scheduling improves individual request latency within a step but doesn't coordinate across steps. Trajectory-aware policies schedule entire sessions as units, which is what actually reduces TCT.

**Real workloads vs synthetic benchmarks**
The synthetic workload (60% easy, 25% medium, 15% hard) models general-purpose tool-calling agents. Real production traces (lmcache-agentic-traces) are very different: SWE-bench is 0.1% easy, 60% medium, 40% hard — almost no short classification calls, dominated by code-debugging tool calls. For specialized workloads, the primary wins come from prefix caching (90% hit rate) and trajectory-aware scheduling, not per-request priority ordering.

---

## Known Limitations

- **No FlashAttention**: PyTorch attention with manual masking. ~16× lower throughput than vLLM on the same hardware. Orthogonal to scheduling.
- **Fake tokenizer in trace replay**: the `bench_agent_trace.py` replay uses `ord(c) % 256` per character rather than real BPE. Token counts are approximate; latency ratios between modes are valid.
- **Difficulty classifier misses code patches in multi-turn context**: later turns in a SWE-bench session have long accumulated prompts but often short outputs (tool calls). The classifier correctly skips the length threshold, but has no way to predict when a tool result will trigger a long code patch response without session-state features.
- **Prefix cache hit rate is 0% on synthetic workloads**: synthetic prompts are unique, so no prefix sharing occurs. The 90% hit rate is observed only on real multi-session traces with shared system prompts.
