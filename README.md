# AgentServe

**An inference engine that knows it's serving an agent, not a chatbot.**

---

## Motivation

Agents produce heterogeneous request streams: a single task fires classification calls, extraction calls, and code-generation calls simultaneously.  Generic schedulers treat every request identically — a `classify` call and a `write-a-program` call wait in the same FIFO queue.  But in an agent DAG, the trivial calls are often *blocking*: the agent can't proceed to the next step until every leaf-node tool call resolves.  Prioritizing fast-finishing requests unblocks the agent's DAG sooner, reducing total task completion time even if hard requests wait a little longer.

---

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │             AgentServe Engine            │
                        │                                          │
  HTTP Request  ──────► │  ┌──────────────┐   ┌───────────────┐  │
  (OpenAI API)          │  │  Difficulty   │   │ Prefix Cache  │  │
                        │  │  Classifier  │   │  (LFU evict)  │  │
                        │  └──────┬───────┘   └───────┬───────┘  │
                        │         │ priority           │ kv blocks │
                        │         ▼                    ▼           │
                        │  ┌──────────────────────────────────┐   │
                        │  │     Agent-Aware Scheduler         │   │
                        │  │  · Policy 1: Priority ordering    │   │
                        │  │  · Policy 2: Easy overflow        │   │
                        │  │  · Policy 3: Preempt young hard   │   │
                        │  └──────────────┬───────────────────┘   │
                        │                 │ batch                   │
                        │                 ▼                         │
                        │  ┌──────────────────────────────────┐   │
                        │  │          Llama Model              │   │
                        │  │  (prefill + batched decode)       │   │
                        │  └──────────────────────────────────┘   │
                        │                 │ logits                  │
                        │                 ▼                         │
                        │  ┌──────────────────────────────────┐   │
                        │  │    Sampling (top-k / top-p)       │   │
                        │  └──────────────────────────────────┘   │
                        │                 │ token                   │
                        └─────────────────┼───────────────────────┘
                                          │
                                          ▼
                                   Response (text)
```

---

## The Three Policies

**Policy 1 — Priority scheduling**
Requests are sorted by difficulty (easy → medium → hard).  Within the same difficulty level, arrival order is preserved (FIFO).  This ensures trivial tool calls unblock downstream agent steps as fast as possible.

**Policy 2 — Soft overflow for easy requests**
When the decode batch is at capacity, easy requests are admitted anyway up to `max_batch * 1.25`.  The marginal cost of one extra easy request is tiny — it generates ~20 tokens and exits quickly, freeing its slot.

**Policy 3 — Preemption of young hard requests**
If all batch slots are occupied by hard requests and an easy request is waiting, the *youngest* hard request (fewest output tokens generated) is preempted back to the pending queue.  Only hard requests with fewer than 10 output tokens are eligible — they haven't invested much compute yet, so the re-prefill cost is low.

All three policies are toggled by `agent_aware=True/False`, enabling direct A/B comparison on identical workloads.

---

## Quick Start

```bash
# Install dependencies
uv sync

# Run all tests on CPU (no GPU needed)
uv run pytest

# Generate the synthetic agent trace
uv run python scripts/generate_synthetic.py

# Benchmark with mock model (CPU, instant)
uv run python scripts/bench_throughput.py --use-mock --num-requests 20 --compare

# Benchmark agent trace replay
uv run python scripts/bench_agent_trace.py --trace traces/synthetic_50.jsonl --compare
```

---

## Benchmark (real model, GPU required)

```bash
# Agent-aware mode
uv run python scripts/bench_throughput.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --num-requests 100 \
    --agent-aware

# Baseline FIFO mode
uv run python scripts/bench_throughput.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --num-requests 100 \
    --baseline

# Side-by-side comparison
uv run python scripts/bench_throughput.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --num-requests 100 \
    --compare
```

---

## Results (placeholder — fill after running on GPU)

| Metric                       | Agent-Aware | Baseline FIFO | Δ |
|------------------------------|-------------|---------------|---|
| Total throughput (tok/s)     |             |               |   |
| Easy request mean latency    |             |               |   |
| Hard request mean latency    |             |               |   |
| Mean TTFT                    |             |               |   |
| Prefix cache hit rate        |             |               |   |
| Agent task completion time   |             |               |   |

---

## Project Structure

```
agentserve/
  model/
    config.py        Model configs: TinyConfig (tests), Llama32 1B/3B/8B
    llama.py         Llama model: RMSNorm, RoPE, GQA attention, SwiGLU
    loader.py        Load HuggingFace safetensor weights
  engine/
    request.py       Request dataclass + lifecycle state machine
    difficulty.py    Heuristic difficulty classifier (easy/medium/hard)
    cache.py         Paged KV-cache block allocator
    prefix_cache.py  Prefix cache with LFU eviction (agent-aware)
    scheduler.py     Continuous-batching scheduler with 3 agent-aware policies
    sampling.py      Temperature / top-k / top-p sampling
    engine.py        Main engine loop
  server/
    app.py           FastAPI OpenAI-compatible API server
scripts/
  generate_synthetic.py   Generate 50-request agent-like trace
  bench_throughput.py     Throughput / latency benchmark
  bench_agent_trace.py    Replay a trace, measure task completion time
  compare_vllm.py         AgentServe vs vLLM comparison + plots
  record_trace.py         Proxy to record live agent API calls
traces/
  synthetic_50.jsonl      Pre-generated 50-request trace
tests/
  test_scheduler.py       Scheduler policies (priority, overflow, preemption)
  test_cache.py           Block allocator (alloc, free, fragmentation)
  test_prefix_cache.py    Prefix matching, LFU eviction, stats
  test_difficulty.py      Classifier on known easy/medium/hard prompts
  test_engine_mock.py     End-to-end with MockModel (CPU only)
notes/
  understanding.md        Component explanations (fill in yourself)
  experiments.md          Experiment log with prediction/measurement columns
  daily.md                Daily learning journal
```

---

## Roadmap

**v0 (current)**
- Llama 3.2 model implementation (1B, 3B, 8B configs)
- Agent-aware scheduler: priority ordering, soft overflow, preemption
- Prefix cache with LFU eviction
- MockModel for CPU-only testing
- OpenAI-compatible FastAPI server
- Benchmark harness and synthetic trace generator

**v1 (next)**
- Speculative decoding (draft model for easy requests)
- Chunked prefill (process long prompts in stages to reduce TTFT)
- Replay recorded OpenHands agent traces
- Flash attention integration for GPU mode

**v2 (future)**
- Custom attention kernels (Triton)
- Tensor parallelism
- Continuous prefix cache eviction with block-level sharing
- Online difficulty learning (update classifier from actual output lengths)
