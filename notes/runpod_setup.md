# RunPod Benchmark Setup Guide

Goal: run the 4-mode ablation on a real Llama-3.2-1B model on a GPU instance,
then compare against vLLM on the same workload.

---

## 1. Provision a RunPod instance

- Pod type: **RTX A4000** (16 GB VRAM) or **A100 40 GB** if available
- Template: **RunPod PyTorch 2.x** (comes with CUDA 11.8+, Python 3.10+)
- Disk: 30 GB minimum (model is ~2.5 GB, plus venv)

---

## 2. SSH in and clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/agentserve.git
cd agentserve
```

---

## 3. Install dependencies

```bash
pip install uv
uv sync
```

If `uv` is not in PATH after install:
```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

---

## 4. Download Llama-3.2-1B-Instruct

You need a HuggingFace token with access to Meta's Llama models
(request access at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).

```bash
# Set your HF token
export HF_TOKEN=hf_...

# Download model weights (safetensors, ~2.5 GB)
uv run huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
    --include "*.safetensors" "*.json" \
    --local-dir /data/llama-3.2-1b \
    --token $HF_TOKEN
```

The model will be at `/data/llama-3.2-1b/`.

---

## 5. Run the CPU sanity check first

```bash
uv run python scripts/bench_ablation.py --use-mock --num-requests 40
```

Expected: all 4 modes complete in under 5 seconds, easy requests have lower
latency in modes (b), (c), (d) vs baseline (a).

---

## 6. Run the full ablation on GPU

```bash
uv run python scripts/bench_ablation.py \
    --model-dir /data/llama-3.2-1b \
    --model-size 1b \
    --num-requests 100 \
    --max-tokens 128 \
    --max-batch 16 \
    --output-json notes/results_ablation.json
```

This runs 4 × 100 requests. Each mode takes ~2–5 min on A4000.
Results are written to `notes/results_ablation.json`.

---

## 7. Install and benchmark vLLM

```bash
pip install vllm

uv run python scripts/bench_ablation.py \
    --model-dir /data/llama-3.2-1b \
    --model-size 1b \
    --num-requests 100 \
    --max-tokens 128 \
    --max-batch 16 \
    --compare-vllm \
    --output-json notes/results_with_vllm.json
```

**What to expect vs vLLM:**
- vLLM will likely win on **raw throughput** (tok/s) for homogeneous workloads
  because it has FlashAttention, Triton kernels, and a mature CUDA runtime.
- AgentServe wins on **easy-request latency** in agent workloads (heterogeneous
  mix). The gap in easy-request mean latency between our mode (d) and vLLM
  baseline is the headline result.
- vLLM has its own priority scheduling (`sampling_params.priority`) but it does
  not tie priority to output-length estimation or DAG structure.

---

## 8. Generate plots

```bash
uv run python scripts/compare_vllm.py \
    --results notes/results_with_vllm.json \
    --output notes/plots/
```

This generates:
- `latency_cdf.png` — latency CDFs by difficulty class, all modes
- `throughput_bar.png` — throughput comparison bar chart
- `ttft_box.png` — TTFT boxplot per mode

---

## 9. Key numbers to record in README

| Metric                            | (a) Baseline | (b) Priority | (c) +Overflow | (d) All 3 | vLLM |
|-----------------------------------|--------------|--------------|---------------|-----------|------|
| Throughput (tok/s)                |              |              |               |           |      |
| Easy request mean latency (s)     |              |              |               |           |      |
| Hard request mean latency (s)     |              |              |               |           |      |
| Mean TTFT (s)                     |              |              |               |           |      |
| Prefix cache hit rate             |              |              |               |           | N/A  |

Fill this into the README's Results section after the run.

---

## Troubleshooting

**`RuntimeError: Failed to load weights for parameters: ...`**
- The safetensors files weren't downloaded fully. Re-run the huggingface-cli
  download command.

**CUDA OOM during benchmark**
- Reduce `--max-batch` (try 8) and `--num-requests` (try 50).
- Check GPU memory: `nvidia-smi`.

**`ModuleNotFoundError: No module named 'safetensors'`**
- `uv add safetensors` then `uv sync`.

**Slow first run**
- The first run compiles CUDA kernels. Subsequent runs are faster.
- For A4000, expect ~40–60 tok/s on 1B model at batch=16.
