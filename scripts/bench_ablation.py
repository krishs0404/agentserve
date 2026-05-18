"""
Ablation study: measures the contribution of each scheduling policy independently.

Four modes:
  (a) baseline   — plain FIFO, no agent-aware policies
  (b) priority   — Policy 1 only (easy requests scheduled first)
  (c) overflow   — Policy 1 + Policy 2 (+ soft batch overflow for easy requests)
  (d) all        — Policy 1 + Policy 2 + Policy 3 (+ preempt young hard requests)

For each mode the script measures:
  - Total wall time
  - Throughput (tok/s)
  - Easy / medium / hard request mean latency
  - Mean TTFT
  - Prefix cache hit rate
  - Agent task completion time (time from first to last completion in each DAG wave)

Usage (mock model, CPU — instant results):
  python scripts/bench_ablation.py --use-mock --num-requests 40

Usage (real model, GPU required):
  python scripts/bench_ablation.py \\
    --model-dir /path/to/Llama-3.2-1B-Instruct \\
    --num-requests 100 --max-tokens 128

Usage (against vLLM for external comparison):
  python scripts/bench_ablation.py --model-dir /path/to/Llama-3.2-1B-Instruct \\
    --compare-vllm --num-requests 100 --max-tokens 128
"""

import argparse
import json
import os
import statistics
import sys
import time
from typing import Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentserve.engine.engine import Engine
from agentserve.engine.request import Request
from agentserve.model.config import TinyConfig, Llama32_1B, Llama32_3B, Llama32_8B


def build_tokenizer(model_dir: str | None, use_mock: bool) -> Callable[[str], list[int]]:
    """Return a tokenize(text) -> list[int] function.

    Uses the real Llama tokenizer when model_dir is provided; falls back to
    ord(c) % 256 for mock/CPU runs.  The fake tokenizer is fast and keeps tests
    dependency-free, but it produces ~3-5× more tokens than BPE for the same
    prompt — fine for intra-mode comparison, misleading for vLLM throughput.
    """
    if model_dir and not use_mock:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model_dir)
            print(f"  Loaded tokenizer from {model_dir} (vocab size {tok.vocab_size})")
            return lambda text: tok.encode(text, add_special_tokens=False)
        except Exception as e:
            print(f"  WARNING: could not load tokenizer from {model_dir}: {e}")
            print("  Falling back to ord(c) % vocab_size fake tokenizer.")
    return lambda text: [ord(c) % 256 for c in text]

# ── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM = (
    "You are a helpful AI assistant embedded in an automated agent pipeline. "
    "Answer concisely. Do not add conversational filler."
)

EASY = [
    f"{SYSTEM}\n\nClassify as POSITIVE or NEGATIVE. One word only.\nReview: 'Excellent build quality, highly recommended.'",
    f"{SYSTEM}\n\nTrue or false: Python lists are mutable. Answer with one word.",
    f"{SYSTEM}\n\nExtract the error code from: 'Error 404: Not Found'. Reply with the number only.",
    f"{SYSTEM}\n\nYes or no: Is 97 a prime number?",
    f"{SYSTEM}\n\nLabel as BUG, FEATURE, or DOCS: 'README is missing installation steps.' One word.",
    f"{SYSTEM}\n\nSentiment: 'The API is fast but the docs are confusing.' Positive/Negative/Mixed. One word.",
    f"{SYSTEM}\n\nIs JSON a binary or text format? One word.",
    f"{SYSTEM}\n\nExtract the HTTP method: 'POST /api/v1/users HTTP/1.1'. Reply with method only.",
]

MEDIUM = [
    f"{SYSTEM}\n\nSummarize in 2 sentences: REST APIs use stateless HTTP to expose resources via standard methods.",
    f"{SYSTEM}\n\nExplain the difference between authentication and authorisation in 3 sentences.",
    f"{SYSTEM}\n\nWhat is the time complexity of quicksort? Explain in 2 sentences.",
    f"{SYSTEM}\n\nConvert to async Python (brief): def fetch(url): return requests.get(url).json()",
    f"{SYSTEM}\n\nList 3 common causes of N+1 query problems in ORMs. One sentence each.",
]

HARD = [
    f"{SYSTEM}\n\nWrite a Python function implementing binary search with type hints, docstring, and error handling.",
    f"{SYSTEM}\n\nImplement a thread-safe LRU cache class in Python with get() and put() methods.",
    f"{SYSTEM}\n\nWrite a FastAPI route for user registration: validate input, hash password, return JWT.",
    f"{SYSTEM}\n\nDesign a rate limiter class in Python. Include token bucket algorithm. Production-quality.",
]


def make_workload(
    n: int,
    max_tokens: int,
    tokenize: Callable[[str], list[int]],
    seed: int = 42,
) -> list[Request]:
    """Generate a realistic agent workload: 60% easy, 25% medium, 15% hard."""
    import random
    rng = random.Random(seed)

    n_easy   = max(1, int(n * 0.60))
    n_medium = max(1, int(n * 0.25))
    n_hard   = max(1, n - n_easy - n_medium)

    prompts = (
        [rng.choice(EASY)   for _ in range(n_easy)]   +
        [rng.choice(MEDIUM) for _ in range(n_medium)] +
        [rng.choice(HARD)   for _ in range(n_hard)]
    )
    rng.shuffle(prompts)

    return [
        Request(
            prompt=p,
            token_ids=tokenize(p),
            max_tokens=max_tokens,
        )
        for p in prompts
    ]


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_mode(
    label: str,
    model_dir: str | None,
    use_mock: bool,
    config,
    requests: list[Request],
    max_batch_size: int,
    enable_priority: bool,
    enable_overflow: bool,
    enable_preemption: bool,
) -> dict:
    engine = Engine(
        config=config,
        use_mock=use_mock,
        agent_aware=enable_priority or enable_overflow or enable_preemption,
        max_batch_size=max_batch_size,
        num_cache_blocks=1024,
        model_dir=model_dir,
        enable_priority=enable_priority,
        enable_overflow=enable_overflow,
        enable_preemption=enable_preemption,
    )

    # Re-create requests for this run (fresh state)
    fresh = [
        Request(prompt=r.prompt, token_ids=list(r.token_ids), max_tokens=r.max_tokens)
        for r in requests
    ]

    t_start = time.monotonic()
    completed = engine.generate(fresh)
    wall = time.monotonic() - t_start

    by_diff: dict[str, list[float]] = {"easy": [], "medium": [], "hard": []}
    ttfts = []
    for r in completed:
        if r.latency > 0:
            by_diff.get(r.difficulty, []).append(r.latency)
        if r.ttft > 0:
            ttfts.append(r.ttft)

    output_tokens = sum(r.num_output_tokens for r in completed)

    def mean(xs): return statistics.mean(xs) if xs else 0.0
    def p95(xs):  return sorted(xs)[int(len(xs) * 0.95)] if len(xs) >= 2 else (xs[0] if xs else 0.0)

    return {
        "label":              label,
        "wall_s":             wall,
        "throughput_tps":     output_tokens / wall if wall > 0 else 0,
        "mean_ttft_s":        mean(ttfts),
        "easy_mean_lat_s":    mean(by_diff["easy"]),
        "med_mean_lat_s":     mean(by_diff["medium"]),
        "hard_mean_lat_s":    mean(by_diff["hard"]),
        "easy_p95_lat_s":     p95(by_diff["easy"]),
        "hard_p95_lat_s":     p95(by_diff["hard"]),
        "prefix_hit_rate":    engine.metrics.prefix_hit_rate,
        "n_easy":             engine.metrics.difficulty_counts.get("easy",   0),
        "n_medium":           engine.metrics.difficulty_counts.get("medium", 0),
        "n_hard":             engine.metrics.difficulty_counts.get("hard",   0),
        "steps":              engine.metrics.steps,
        # Raw latency arrays — consumed by plot_results.py to draw CDFs
        "easy_latencies":     by_diff["easy"],
        "medium_latencies":   by_diff["medium"],
        "hard_latencies":     by_diff["hard"],
        "all_ttfts":          ttfts,
    }


# ── vLLM comparison ───────────────────────────────────────────────────────────

def run_vllm(model_dir: str, requests: list[Request], max_tokens: int) -> dict | None:
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("  vLLM not installed — skipping. Install with: pip install vllm")
        return None

    import os
    # vLLM 0.21.0 tries to warm up DeepGEMM FP8 kernels unconditionally on H100,
    # even for float16 models. Disable it so the benchmark doesn't require deep_gemm.
    os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

    llm = LLM(model=model_dir, dtype="float16", gpu_memory_utilization=0.80)
    params = SamplingParams(max_tokens=max_tokens, temperature=1.0)
    prompts = [r.prompt for r in requests]

    t_start = time.monotonic()
    outputs = llm.generate(prompts, params)
    wall = time.monotonic() - t_start

    output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    return {
        "label":          "vLLM (FIFO)",
        "wall_s":         wall,
        "throughput_tps": output_tokens / wall if wall > 0 else 0,
        "note": "vLLM runs all requests FIFO with PagedAttention; no per-request priority.",
    }


# ── Printing ──────────────────────────────────────────────────────────────────

def print_table(results: list[dict]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        _print_rich(results, Console(), box)
    except ImportError:
        _print_plain(results)


def _print_rich(results, console, box_module):
    from rich.table import Table
    t = Table(title="AgentServe Ablation Results", box=box_module.ROUNDED)
    t.add_column("Metric", style="bold cyan")
    for r in results:
        t.add_column(r["label"], justify="right")

    rows = [
        ("Wall time (s)",          lambda r: f"{r.get('wall_s', '—'):.2f}"         if "wall_s"          in r else "—"),
        ("Throughput (tok/s)",     lambda r: f"{r.get('throughput_tps', '—'):.1f}" if "throughput_tps"  in r else "—"),
        ("Mean TTFT (s)",          lambda r: f"{r.get('mean_ttft_s', '—'):.3f}"    if "mean_ttft_s"     in r else "—"),
        ("Easy mean lat (s)",      lambda r: f"{r.get('easy_mean_lat_s', '—'):.3f}" if "easy_mean_lat_s" in r else "—"),
        ("Medium mean lat (s)",    lambda r: f"{r.get('med_mean_lat_s', '—'):.3f}" if "med_mean_lat_s"  in r else "—"),
        ("Hard mean lat (s)",      lambda r: f"{r.get('hard_mean_lat_s', '—'):.3f}" if "hard_mean_lat_s" in r else "—"),
        ("Easy P95 lat (s)",       lambda r: f"{r.get('easy_p95_lat_s', '—'):.3f}" if "easy_p95_lat_s"  in r else "—"),
        ("Hard P95 lat (s)",       lambda r: f"{r.get('hard_p95_lat_s', '—'):.3f}" if "hard_p95_lat_s"  in r else "—"),
        ("Prefix hit rate",        lambda r: f"{r.get('prefix_hit_rate', 0):.1%}"  if "prefix_hit_rate" in r else "—"),
        ("Easy / Med / Hard",      lambda r: f"{r.get('n_easy','?')}/{r.get('n_medium','?')}/{r.get('n_hard','?')}" if "n_easy" in r else "—"),
        ("Engine steps",           lambda r: str(r.get("steps", "—"))),
    ]
    for label, fn in rows:
        t.add_row(label, *[fn(r) for r in results])
    console.print(t)


def _print_plain(results: list[dict]) -> None:
    labels = "  |  ".join(r["label"] for r in results)
    print(f"\n{'='*70}")
    print(f" AgentServe Ablation: {labels}")
    print(f"{'='*70}")
    fields = [
        ("Wall time (s)",          "wall_s"),
        ("Throughput (tok/s)",     "throughput_tps"),
        ("Mean TTFT (s)",          "mean_ttft_s"),
        ("Easy mean lat (s)",      "easy_mean_lat_s"),
        ("Medium mean lat (s)",    "med_mean_lat_s"),
        ("Hard mean lat (s)",      "hard_mean_lat_s"),
        ("Prefix hit rate",        "prefix_hit_rate"),
        ("Engine steps",           "steps"),
    ]
    for label, key in fields:
        vals = []
        for r in results:
            v = r.get(key, "—")
            vals.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        print(f"  {label:<28} " + "  |  ".join(f"{v:>12}" for v in vals))
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="AgentServe ablation benchmark")
    p.add_argument("--use-mock",     action="store_true", default=False,
                   help="Use mock model (CPU, instant — for sanity checks)")
    p.add_argument("--model-dir",    default=None,
                   help="Path to HuggingFace model directory (e.g. /data/Llama-3.2-1B-Instruct)")
    p.add_argument("--model-size",   default="1b", choices=["1b", "3b", "8b"])
    p.add_argument("--num-requests", type=int,  default=60)
    p.add_argument("--max-tokens",   type=int,  default=64)
    p.add_argument("--max-batch",    type=int,  default=16)
    p.add_argument("--compare-vllm", action="store_true", default=False,
                   help="Also benchmark vLLM on the same workload (requires vLLM installed)")
    p.add_argument("--output-json",  default=None,
                   help="Write raw results to a JSON file")
    args = p.parse_args()

    if not args.use_mock and args.model_dir is None:
        print("ERROR: Provide --model-dir or --use-mock")
        sys.exit(1)

    # Config
    if args.use_mock:
        config = TinyConfig
    elif args.model_size == "1b":
        config = Llama32_1B
    elif args.model_size == "3b":
        config = Llama32_3B
    else:
        config = Llama32_8B

    tokenize = build_tokenizer(args.model_dir, args.use_mock)
    workload = make_workload(args.num_requests, args.max_tokens, tokenize=tokenize)
    print(f"Workload: {args.num_requests} requests  "
          f"({sum(1 for r in workload if 'classify' in r.prompt.lower() or 'true or false' in r.prompt.lower() or 'yes or no' in r.prompt.lower())} easy-ish), "
          f"max_tokens={args.max_tokens}, batch={args.max_batch}")
    print()

    common = dict(
        model_dir=args.model_dir,
        use_mock=args.use_mock,
        config=config,
        requests=workload,
        max_batch_size=args.max_batch,
    )

    modes = [
        ("(a) Baseline FIFO",      False, False, False),
        ("(b) Priority only",       True,  False, False),
        ("(c) Priority + Overflow", True,  True,  False),
        ("(d) All 3 Policies",      True,  True,  True),
    ]

    results = []
    for label, pri, ovf, pre in modes:
        print(f"  Running {label}...")
        r = run_mode(label, enable_priority=pri, enable_overflow=ovf,
                     enable_preemption=pre, **common)
        results.append(r)
        print(f"    wall={r['wall_s']:.2f}s  tps={r['throughput_tps']:.1f}  "
              f"easy_lat={r['easy_mean_lat_s']:.3f}s  hard_lat={r['hard_mean_lat_s']:.3f}s")

    if args.compare_vllm and args.model_dir:
        print("  Running vLLM baseline...")
        vr = run_vllm(args.model_dir, workload, args.max_tokens)
        if vr:
            results.append(vr)

    print()
    print_table(results)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.output_json}")


if __name__ == "__main__":
    main()
