# AgentServe

A personal ML systems prototype for studying scheduling in multi-step agent workloads.
Implemented in PyTorch: grouped-query Llama attention, continuous batching, ordinary
and paged KV storage, prefix reuse, and request/trajectory scheduling policies.

**Current finding:** keyword output-length estimates fail to beat a constant
training-set median on held-out agent sessions. That challenges an assumption
behind short-request priority and motivates better evaluation before performance claims.

| Evidence | Status |
|---|---|
| Numerical correctness | CPU regression coverage for repeated, mixed-length batched decoding and tiny Hugging Face Llama parity |
| Recorded trace study | 24,880 turns, 767 sessions, five session-held-out splits; raw predictions and split manifests included |
| Pretrained Llama 3.2 parity | Unvalidated; scaled RoPE tested against Hugging Face on random weights; intended checkpoint access is gated |
| GPU scheduling speedups | Historical measurements require reruns after correctness fixes |
| Whole-session latency and fairness | Open questions; no validated result from dependency-preserving replay |

Read the [experiment report](docs/length-prediction.md) for methods, uncertainty,
negative results, and limitations. [Historical notes](docs/historical-results.md)
retain earlier exploratory results for provenance only.

## Reproduce

Python 3.11 or newer. From the repository root:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/study_lengths.py
```

The trace study itself uses only the Python standard library and the repository
classifier. It runs on CPU, without a model download. The default input is already
tracked. Results go to `experiments/length_prediction/`: dataset SHA-256, exact
train/test session IDs, all held-out predictions, and metrics for every seed.

## Architecture

- `agentserve/model/`: Llama implementation and safetensors loader.
- `agentserve/engine/`: scheduler, length estimates, KV allocation, prefix cache,
  request lifecycle, and trajectory policies.
- `agentserve/server/`: HTTP serving interface.
- `scripts/`: benchmark drivers, trace conversion, and the reproducible trace study.
- `tests/`: numerical parity, cache, scheduling, tokenization, and leakage checks.

Priority, overflow, preemption, and relative batching are experimental mechanisms.
Their implementation does not establish better end-to-end agent performance.
The HTTP interface and engine are research scaffolding, with no production SLA.

## Next experiment

Scaled RoPE and strict checkpoint-configuration checks are implemented and tested.
Pretrained validation still requires an accessible local checkpoint:

```bash
python scripts/validate_checkpoint.py /path/to/checkpoint --device cuda --out /tmp/parity.json
```

This compares float32 prefill logits and eight mixed-length batched decode steps
against Hugging Face, checking greedy tokens and reporting numerical errors. It
is a bounded correctness check, not a throughput benchmark or exhaustive validation.
The command was smoke-tested with a synthetic saved checkpoint; access to the
intended pretrained Llama checkpoint is gated on this workstation.
Then replay sessions causally: next arrival = previous completion + recorded tool
wait. Compare identical arrivals and workloads across policies after warmup, with
multiple repetitions. Report whole-session completion time, request p50/p95/p99,
long-request slowdown, and starvation alongside aggregate latency.

The existing converter assigns a fixed two-second generation duration to create
arrival times. Those arrivals cannot establish causal whole-session latency.
The local prompt/output pairs omit full contexts and tool gaps; they cannot measure
prefix reuse, growing-context GPU cost, or tool-inclusive session completion time.

## References and provenance

- [LMCache dataset documentation](https://huggingface.co/datasets/DiscoPosse/lmcache-agentic-traces), CC BY 4.0. Local pairs were produced by the existing conversion pipeline; see the report for provenance limits.
- [TraceLab](https://arxiv.org/abs/2606.30560), coding-agent workload characterization. Motivation for studying session structure; its findings are not measurements of AgentServe.
- [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), intended pretrained reference.

Development used substantial AI assistance (Claude Code and Codex) for implementation,
debugging, tests, analysis, and documentation. Results should be assessed through
the committed code, raw artifacts, and stated validation scope.
