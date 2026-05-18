# AgentServe

**A custom LLM inference engine built for agents — with scheduling policies that understand the structure of agentic workloads.**

Benchmarked on Llama 3.2-1B on an NVIDIA H100.

---

## The Problem with Standard Schedulers

When a language model serves a chatbot, every request looks roughly the same: one user turn, one model response, done. Standard inference engines (including vLLM) run a FIFO queue and batch whatever's waiting. That's optimal for chatbots.

Agents are different. A single agent task fires **heterogeneous bursts**: maybe five `classify()` calls, two `extract()` calls, and one `write_code()` call, all simultaneously. The classify calls take 20 tokens each. The write_code call takes 800 tokens. In FIFO order, the five classifiers get stuck behind the code generation — and since classify calls are *blocking dependencies* in the agent's DAG, the whole task stalls waiting for them.

The core insight: **finishing fast requests faster is more valuable than throughput fairness**, because fast requests are disproportionately likely to be blocking the agent's next step.

---

## What AgentServe Does

AgentServe is a hand-written inference engine — PyTorch attention, paged KV-cache, continuous batching, prefix cache — layered with scheduling policies that exploit the structure of agent workloads.

Three scheduling policies work together:

**Policy 1 — Priority ordering**
A lightweight difficulty classifier (heuristic on prompt length and keywords) assigns each request to an `easy / medium / hard` bucket. The scheduler maintains three FIFO deques and always drains easy before medium before hard. Within a tier, arrival order is preserved. The data structure is three `deque`s — `O(1)` enqueue and dequeue.

**Policy 2 — Soft overflow for easy requests**
When the decode batch is at `max_batch_size` and an easy request arrives, it's admitted anyway up to `1.25×` capacity. An extra easy request generates ~20 tokens and exits in a few steps, freeing its slot before the batch would have rotated anyway. The marginal cost is tiny; the latency benefit is real.

**Policy 3 — Preemption of young hard requests**
If all batch slots are occupied by hard requests and an easy request has been waiting, the youngest hard request (fewest output tokens generated so far) is returned to pending and re-prefilled. Eligibility threshold: fewer than 10 output tokens — they've invested almost no compute, so the re-prefill cost is a rounding error.

**Trajectory-aware policies** extend this to multi-step agent workflows (ReAct loops, plan-execute, reflection chains). Two trajectory policies implement `SchedulerPolicy` as a plug-in:
- `TrajectoryProgressPolicy` — prioritizes trajectories past their midpoint (closer to completion = finish first, free the slot)
- `TrajectoryDeadlinePolicy` — schedules by urgency = remaining tokens / time remaining, with fallback to progress ordering when no deadline pressure exists

---

## Results

All benchmarks run on Llama 3.2-1B on a single H100 80GB. Workload: 100 synthetic agent requests (64% easy, 27% medium, 9% hard by the classifier).

### Per-request latency ablation

| Configuration | Throughput | Easy lat (mean) | Hard lat (mean) | TTFT (mean) |
|---|---|---|---|---|
| (a) Baseline FIFO | 862 tok/s | 8.33 s | 7.61 s | 6.19 s |
| (b) Priority only | 880 tok/s | **5.66 s** | 13.94 s | 5.99 s |
| (c) Priority + Overflow | 876 tok/s | 5.85 s | 14.42 s | 5.77 s |
| (d) All 3 policies | 826 tok/s | 6.32 s | 15.31 s | 6.21 s |

Priority scheduling cuts easy-request latency by **32%** (8.33 s → 5.66 s) with a slight throughput gain. Hard requests wait longer — that's the trade-off, and it's the right one: in an agent DAG, easy requests are usually blocking dependencies; hard requests are usually background work.

The "all 3 policies" configuration shows diminishing returns here because the synthetic workload doesn't have a high enough rate of concurrent easy+hard requests to trigger preemption often. It would shine more under sustained burst traffic.

### Trajectory completion time (real model, H100)

30 trajectories per template, 120 total competing for batch slots simultaneously. TCT = wall time from first step submission to last step completion.

**P50 TCT:**

| Policy | react (3-step) | plan_execute (4-step) | reflect (3-step) | chat (4-step) |
|---|---|---|---|---|
| FIFO | 43.9 s | 68.1 s | 55.8 s | 74.3 s |
| Priority | 43.4 s | 68.2 s | 55.7 s | 74.6 s |
| traj_progress | **6.6 s** | 54.8 s | 32.0 s | 70.4 s |
| traj_deadline | **6.8 s** | **25.6 s** | 50.1 s | 69.9 s |

`traj_progress` cuts ReAct P50 TCT **6.6×** vs FIFO (43.9 s → 6.6 s) by front-loading trajectories past their midpoint — short 3-step trajectories aren't blocked by long ones in mid-flight.

`traj_deadline` cuts plan_execute P50 TCT **2.7×** vs FIFO (68.1 s → 25.6 s) by urgency scoring — 4-step trajectories accumulate increasing remaining_tokens/time_remaining pressure as steps pile up, and the scheduler promotes them before they miss their window.

Priority alone does almost nothing here: per-request difficulty ordering doesn't understand that steps of the same trajectory should be kept together. The trajectory-aware policies do.

### vLLM comparison

| System | Throughput |
|---|---|
| AgentServe (all policies) | 826 tok/s |
| vLLM 0.21 (FIFO) | 13,915 tok/s |

vLLM is 16× faster in raw throughput — FlashAttention3, CUDA graphs, and production-grade kernels. This is expected and orthogonal to the scheduling story: AgentServe's scheduling improvements (32% latency reduction) apply *on top of* whatever kernel throughput the engine has. The value here is in the scheduling architecture, not the GEMM performance.

---

## Architecture

```
                     ┌────────────────────────────────────────────┐
                     │              AgentServe Engine              │
                     │                                             │
 Submitted request ► │  ┌─────────────┐    ┌──────────────────┐  │
                     │  │  Difficulty  │    │   Prefix Cache   │  │
                     │  │  Classifier  │    │   (LFU evict)    │  │
                     │  └──────┬──────┘    └────────┬─────────┘  │
                     │         │ priority             │ kv slices  │
                     │         ▼                      ▼            │
                     │  ┌──────────────────────────────────────┐  │
                     │  │        Agent-Aware Scheduler          │  │
                     │  │                                        │  │
                     │  │  Baseline path (policy=None):          │  │
                     │  │  · 3 priority deques (easy/med/hard)   │  │
                     │  │  · Soft overflow for easy requests      │  │
                     │  │  · Preempt young hard requests          │  │
                     │  │                                        │  │
                     │  │  Trajectory path (policy=<object>):    │  │
                     │  │  · Sorted list keyed by policy.key()   │  │
                     │  │  · Pluggable: progress / deadline       │  │
                     │  └──────────────┬───────────────────────┘  │
                     │                 │ prefill + decode batches   │
                     │                 ▼                            │
                     │  ┌──────────────────────────────────────┐  │
                     │  │   Llama 3.2 Model (PyTorch, fp16)    │  │
                     │  │   RMSNorm · RoPE · GQA · SwiGLU      │  │
                     │  │   Paged KV-cache block allocator      │  │
                     │  └──────────────────────────────────────┘  │
                     └────────────────────────────────────────────┘
```

---

## Running It

```bash
# Install
uv sync

# All tests on CPU — no GPU needed
uv run pytest

# Mock model benchmark (scheduling policies, CPU, instant)
uv run python scripts/bench_ablation.py --use-mock --output-json notes/mock_results.json

# Full ablation on real model (H100 recommended)
uv run python scripts/bench_ablation.py \
    --model-dir /path/to/llama-3.2-1b \
    --output-json notes/results.json

# Plot the ablation results
uv run python scripts/plot_results.py \
    --results notes/results.json \
    --out notes/plots/

# Trajectory scheduling benchmark (mock, fast)
uv run python scripts/bench_trajectories.py --n-traj 30 --out notes/plots/

# Trajectory benchmark on real model
uv run python scripts/bench_trajectories.py \
    --model-dir /path/to/llama-3.2-1b \
    --estimated-tps 500 \
    --n-traj 30 \
    --out notes/plots/
```

---

## Project Structure

```
agentserve/
  model/
    config.py          ModelConfig dataclass; TinyConfig (tests), Llama32_1B/3B/8B
    llama.py           Llama: RMSNorm, RoPE, GQA attention (fp16), SwiGLU FFN
    loader.py          Load HuggingFace safetensors; handles Llama 3.2 weight tying
  engine/
    request.py         Request dataclass + lifecycle (PENDING→PREFILL→DECODE→DONE)
    difficulty.py      Heuristic classifier → easy / medium / hard + priority int
    cache.py           Paged KV-cache block allocator
    prefix_cache.py    Prefix cache with LFU eviction
    scheduler.py       Continuous-batching scheduler: 3-deque path + plug-in policy path
    policies.py        SchedulerPolicy ABC; Fifo, Priority, TrajectoryProgress, Deadline
    trajectory.py      TrajectorySpec generator: react / plan_execute / reflect / chat
    sampling.py        Temperature / top-k / top-p sampling; fp16-safe
    engine.py          Main step loop tying all components together
  server/
    app.py             FastAPI OpenAI-compatible server
scripts/
  bench_ablation.py    4-mode ablation (baseline / priority / overflow / preemption)
  bench_trajectories.py  Trajectory TCT benchmark across 4 policies × 4 templates
  plot_results.py      Generate plots from ablation JSON output
  bench_agent_trace.py   Replay a recorded agent trace
  generate_synthetic.py  Synthetic 50-request agent trace
  record_trace.py      Proxy to capture live agent API calls
traces/
  synthetic_50.jsonl   Pre-generated trace
tests/
  test_scheduler.py    Priority ordering, overflow, preemption
  test_cache.py        Block allocator alloc/free/fragmentation
  test_prefix_cache.py Prefix matching, LFU eviction
  test_difficulty.py   Classifier on known prompts
  test_engine_mock.py  End-to-end with MockModel (CPU)
notes/
  plots/               Generated benchmark plots (PNG)
  results_with_vllm.json  Ablation + vLLM baseline results
  trajectory_results.json Trajectory TCT per policy × template
```

---

## Design Decisions

**Why a hand-written attention instead of FlashAttention?**
The goal was to build the full stack from scratch — understanding every component. The attention kernel is standard PyTorch `scaled_dot_product` with a causal mask in fp16. Production throughput would come from dropping in a FlashAttention or Triton kernel; the scheduling layer doesn't change.

**Why a heuristic classifier instead of a learned one?**
Latency. A classifier that runs before scheduling must add zero overhead to the critical path. The heuristic (prompt length + keyword matching) runs in microseconds. A small learned classifier could improve accuracy but would need separate benchmarking to justify the overhead.

**Why pluggable policies instead of baking them in?**
The `SchedulerPolicy` ABC decouples policy logic from the scheduling loop. Adding a new policy is one class with two methods — `priority_key()` defines ordering, `on_request_complete()` updates state. The existing three-deque logic remains unchanged when `policy=None`.

**Why does vLLM beat AgentServe 16× in throughput?**
FlashAttention3, CUDA graphs, and C++ kernels vs Python loops. This is a kernel throughput gap, not a scheduling gap. The scheduling improvements — 32% easy-request latency reduction, 6.6× ReAct TCT reduction — are orthogonal and would stack on top of any backend.
