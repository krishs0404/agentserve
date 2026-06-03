# AgentServe — Key Findings Reference

All numbers from GPU runs on A10G, Llama 3.2-1B, unless noted.

---

## 1. Priority Scheduling (per-request)

**Workload**: 100 requests, 64% easy / 27% medium / 9% hard, max_tokens=64

| Metric | FIFO (a) | All 3 Policies (d) | Change |
|---|---|---|---|
| Easy mean latency | 11.84 s | 8.07 s | **−32%** |
| Hard mean latency | 10.85 s | 19.02 s | **+75%** |
| Throughput | 314 tok/s | 333 tok/s | +6% |
| TTFT | 9.07 s | 8.10 s | −11% |

**The tradeoff is explicit**: easy requests unblock agent DAG dependencies fast (+32% better), hard requests wait longer (+75%). This is the right tradeoff for heterogeneous agents where easy calls are blocking dependencies.

---

## 2. Trajectory-Aware Scheduling

**Workload**: 20 trajectories × 4 templates, 80 total, all competing simultaneously.

| Policy | ReAct (3-step) | Plan-Execute (4-step) | Reflect (3-step) | Chat (4-turn) | Avg speedup |
|---|---|---|---|---|---|
| FIFO | 99.6 s | 156.3 s | 128.4 s | 169.6 s | 1.00× |
| Priority | 100.0 s | 158.8 s | 129.5 s | 173.2 s | **0.99×** |
| traj_progress | **16.7 s** | 127.5 s | **75.3 s** | 162.4 s | 2.48× |
| traj_deadline | **16.7 s** | **60.1 s** | 112.3 s | 156.7 s | **2.70×** |

**The critical finding**: per-request priority gives zero trajectory benefit (0.99× ≈ 1.00×). Trajectory-aware policies are qualitatively different — 6× for ReAct, 2.6× for Plan-Execute.

**Policy selection**:
- traj_progress wins for short/uniform sessions (ReAct: 6×, Reflect: 1.7×)
- traj_deadline wins for sessions with uneven step costs (Plan-Execute: 2.6×)
- Neither helps Chat — sequential turns have no parallelism to exploit

---

## 3. Scheduling Benefit vs Workload Heterogeneity (mock model)

Priority scheduling benefit scales with workload composition:

| Easy % | Hard % | Easy-latency improvement |
|---|---|---|
| 3% | 72% | +78% |
| 15% | 60% | +63% |
| 30% | 45% | +47% |
| 60% | 15% | +24% (our benchmark) |
| 75% | 5% | +9% |

**Insight**: benefit is highest when easy requests are rare and thus most blocked. SWE-bench (~0% easy) gets zero priority benefit — but maximum trajectory-aware benefit.

---

## 4. Relative Batching (mode e) vs Priority (mode d)

Both reduce latency relative to FIFO, but in fundamentally different ways:

| Mode | Easy latency | Hard latency | Mechanism |
|---|---|---|---|
| (d) All 3 Policies | −32% | +75% | Promotes easy, penalises hard |
| (e) Relative Batching | −3% | **−9%** | Reduces KV-padding waste uniformly |

**Relative batching** groups requests by predicted output length, minimising intra-batch variance. Benefits all difficulty classes equally (no winner/loser). Best for homogeneous workloads (SWE-bench).

**Combined (f)** = same as Priority. Within-tier relative batching adds no benefit because predicted lengths already cluster within each tier — the benefit of mode (e) comes from grouping across tiers.

---

## 5. Classifier Robustness (mock model with noise injection)

| Noise rate | Easy-latency improvement |
|---|---|
| 0% (perfect) | +66% |
| 10% | +67% |
| 20% | +64% |
| 30% | +64% |
| 50% (random) | +65% |

**Finding**: scheduling benefit is robust to classifier errors. At 50% random labelling, still +65% improvement. The benefit comes from statistical separation, not perfect classification.

---

## 6. Classifier Accuracy: Learned vs Keywords

On synthetic benchmark prompts:
- Keyword heuristic: 75.5% bucket accuracy
- Learned linear classifier (8 features + online SGD): 78.0% (+2.5%)

On 24,880 real lmcache-agentic-traces (SWE-bench/GAIA):
- Keyword heuristic: 63.8%
- Learned linear classifier: 46.0% (−17.8%)

**Finding**: the learned classifier is better on synthetic data but worse on real SWE-bench data. Real agent traces are dominated by medium tool calls regardless of prompt content — structural features can't distinguish them from hard code patches. Session-state features (turn count, recent errors) would be needed.

---

## 7. Prefix Cache — Real vs Synthetic Traces

| Workload | Hit rate |
|---|---|
| Synthetic (unique prompts) | 0% |
| SWE-bench real (50 sessions, shared system prompt) | **90%** |

SWE-bench has 669 sessions all sharing the same ~14K-token system prompt. After the first session, every subsequent session skips recomputing attention over that prefix.

---

## 8. Real Workload Distribution vs Synthetic Assumption

| Workload | Easy | Medium | Hard |
|---|---|---|---|
| Synthetic (our benchmark) | 60% | 25% | 15% |
| SWE-bench real traces | 0.1% | 60% | 40% |
| GAIA real traces | 1% | 69% | 30% |

Real production code agents have almost no easy requests — they're dominated by medium tool calls and hard code generation. Generic synthesis agents (customer service, research) would look more like the synthetic distribution.

---

## 9. Scheduling Benefit Grows with Sequence Length

Comparison at max_tokens=64 (batch=16) vs max_tokens=256 (batch=4):

| Mode | Easy improvement (64) | Easy improvement (256) |
|---|---|---|
| (b) Priority only | −33% | −35% |
| (c) Priority + Overflow | −32% | **−41%** |
| (d) All 3 Policies | −32% | −39% |
| (e) Relative Batching | −3% | +2% (worse) |

At max_tokens=256, priority scheduling cuts easy latency **41%** vs 32% at max_tokens=64. The reason: hard requests occupy batch slots for 4× longer at longer sequences, so easy requests accumulate more blockage in FIFO. Priority scheduling relieves proportionally more blockage.

Relative batching weakens at batch=4 — the sliding window has fewer candidates to optimize, and there are fewer opportunities to group similar-length requests.

**Practical implication**: priority scheduling is most valuable for long-context agent workloads (tool results with large context windows, multi-turn code review). The benefit compounds with sequence length.

## 10. SDPA / Flash Attention 2 Impact

Throughput comparison before/after SDPA at max_tokens=64:

| Mode | Pre-SDPA | Post-SDPA | Delta |
|---|---|---|---|
| (a) FIFO | 313 tok/s | 314 tok/s | +0.4% |
| (c) Priority+Overflow | 332 tok/s | 334 tok/s | +0.6% |
| (e) Relative | 317 tok/s | 320 tok/s | +0.9% |

At max_tokens=64, within noise. The bottleneck is the Python scheduler loop, not the CUDA kernel. At max_tokens=256 (longer sequences), throughput improves more (+8-10%) as SDPA's memory efficiency advantage grows.

---

## Summary Numbers for Video/Paper

- Priority scheduling: **−32% easy latency** on heterogeneous agent workloads
- ReAct trajectory: **6× faster** completion time with traj_progress
- Plan-Execute: **2.6× faster** with traj_deadline  
- Prefix cache: **90% hit rate** on real SWE-bench sessions
- Classifier robustness: **+65% benefit at 50% noise** (scheduling works even with imperfect labels)
- Heterogeneity: **+78% benefit** when only 3% easy, **+24%** at 60% easy
