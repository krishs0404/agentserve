"""
Throughput benchmark: fire N concurrent requests through the engine.

Measures:
  - Total throughput (tokens/s across all requests)
  - Mean / P50 / P95 per-request latency
  - Mean TTFT (time-to-first-token)
  - Prefix cache hit rate
  - Difficulty distribution

Run with both modes and see a side-by-side comparison:
  uv run python scripts/bench_throughput.py --use-mock --num-requests 20 --agent-aware
  uv run python scripts/bench_throughput.py --use-mock --num-requests 20 --baseline
"""

import argparse
import sys
import os
import time
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentserve.model.config import TinyConfig, Llama32_1B, Llama32_3B, Llama32_8B
from agentserve.engine.engine import Engine
from agentserve.engine.request import Request

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# Shared system prompt for prefix-cache testing
SYSTEM_PROMPT = (
    "You are a helpful AI assistant embedded in an automated agent pipeline. "
    "Answer concisely and directly. Do not include conversational filler. "
    "This is an agent pipeline — clarity and brevity are critical."
)

EASY_PROMPTS = [
    f"{SYSTEM_PROMPT}\n\nClassify this review as POSITIVE or NEGATIVE. One word only.\nReview: 'Great product, very satisfied!'",
    f"{SYSTEM_PROMPT}\n\nTrue or false: HTTP is a stateless protocol. Answer with one word only.",
    f"{SYSTEM_PROMPT}\n\nExtract the version number from: 'Using requests==2.28.0'. Reply with the version only.",
    f"{SYSTEM_PROMPT}\n\nYes or no: Is 17 a prime number?",
    f"{SYSTEM_PROMPT}\n\nLabel as BUG, FEATURE, or QUESTION: 'The login button doesn't work.' One word only.",
]

MEDIUM_PROMPTS = [
    f"{SYSTEM_PROMPT}\n\nSummarize in 2 sentences: REST APIs use HTTP methods to perform CRUD operations on resources.",
    f"{SYSTEM_PROMPT}\n\nExplain the difference between a list and a tuple in Python in 3 sentences.",
    f"{SYSTEM_PROMPT}\n\nWrite a brief docstring for: def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    f"{SYSTEM_PROMPT}\n\nWhat are 2 common causes of memory leaks in JavaScript? One line each.",
    f"{SYSTEM_PROMPT}\n\nConvert to JavaScript (brief): def greet(name): return f'Hello, {{name}}!'",
]

HARD_PROMPTS = [
    f"{SYSTEM_PROMPT}\n\nWrite a Python function that implements binary search with proper type hints and docstring.",
    f"{SYSTEM_PROMPT}\n\nImplement a simple LRU cache in Python with get and put operations.",
    f"{SYSTEM_PROMPT}\n\nWrite a complete FastAPI endpoint for user authentication via JWT. Include request/response models.",
]


def make_requests(n: int, max_tokens: int) -> list[Request]:
    """Generate a mix of easy/medium/hard requests sharing a system prompt."""
    all_prompts = []
    # 60% easy, 25% medium, 15% hard
    n_easy   = max(1, int(n * 0.60))
    n_medium = max(1, int(n * 0.25))
    n_hard   = max(1, n - n_easy - n_medium)

    import random
    random.seed(0)
    all_prompts = (
        [random.choice(EASY_PROMPTS)   for _ in range(n_easy)]   +
        [random.choice(MEDIUM_PROMPTS) for _ in range(n_medium)] +
        [random.choice(HARD_PROMPTS)   for _ in range(n_hard)]
    )
    random.shuffle(all_prompts)

    requests = []
    for i, prompt in enumerate(all_prompts):
        # Use character-level tokenisation as a stand-in for real tokenisation
        token_ids = [ord(c) % 256 for c in prompt]
        requests.append(Request(
            prompt=prompt,
            token_ids=token_ids,
            max_tokens=max_tokens,
            temperature=1.0,
        ))
    return requests


def run_bench(
    agent_aware: bool,
    use_mock: bool,
    model_name: str,
    num_requests: int,
    max_tokens: int,
    max_batch_size: int,
) -> dict:
    if use_mock:
        config = TinyConfig
    elif "1b" in model_name.lower():
        config = Llama32_1B
    elif "3b" in model_name.lower():
        config = Llama32_3B
    elif "8b" in model_name.lower():
        config = Llama32_8B
    else:
        config = TinyConfig

    engine = Engine(
        config=config,
        use_mock=use_mock,
        agent_aware=agent_aware,
        max_batch_size=max_batch_size,
        num_cache_blocks=512,
    )

    requests = make_requests(num_requests, max_tokens)
    t_start = time.monotonic()
    completed = engine.generate(requests)
    wall_time = time.monotonic() - t_start

    latencies = [r.latency for r in completed if r.latency > 0]
    ttfts     = [r.ttft    for r in completed if r.ttft    > 0]
    total_tokens = sum(r.num_output_tokens for r in completed)

    result = {
        "mode":           "agent-aware" if agent_aware else "baseline",
        "requests":       len(completed),
        "wall_time_s":    wall_time,
        "throughput_tps": total_tokens / wall_time if wall_time > 0 else 0,
        "mean_latency_s": statistics.mean(latencies) if latencies else 0,
        "p50_latency_s":  statistics.median(latencies) if latencies else 0,
        "p95_latency_s":  sorted(latencies)[int(len(latencies)*0.95)] if len(latencies) >= 2 else 0,
        "mean_ttft_s":    statistics.mean(ttfts) if ttfts else 0,
        "prefix_hit_rate": engine.metrics.prefix_hit_rate,
        "prefix_hits":    engine.metrics.prefix_cache_hits,
        "diff_easy":      engine.metrics.difficulty_counts.get("easy", 0),
        "diff_medium":    engine.metrics.difficulty_counts.get("medium", 0),
        "diff_hard":      engine.metrics.difficulty_counts.get("hard", 0),
        "steps":          engine.metrics.steps,
    }
    return result


def print_results(results: list[dict]) -> None:
    if HAS_RICH:
        _print_rich(results)
    else:
        _print_plain(results)


def _print_rich(results: list[dict]) -> None:
    console = Console()
    table = Table(title="AgentServe Throughput Benchmark", box=box.ROUNDED)
    table.add_column("Metric", style="bold cyan")
    for r in results:
        table.add_column(r["mode"].upper(), justify="right")

    rows = [
        ("Requests completed",    lambda r: str(r["requests"])),
        ("Wall time (s)",         lambda r: f"{r['wall_time_s']:.2f}"),
        ("Throughput (tok/s)",    lambda r: f"{r['throughput_tps']:.1f}"),
        ("Mean latency (s)",      lambda r: f"{r['mean_latency_s']:.3f}"),
        ("P50 latency (s)",       lambda r: f"{r['p50_latency_s']:.3f}"),
        ("P95 latency (s)",       lambda r: f"{r['p95_latency_s']:.3f}"),
        ("Mean TTFT (s)",         lambda r: f"{r['mean_ttft_s']:.3f}"),
        ("Prefix hit rate",       lambda r: f"{r['prefix_hit_rate']:.1%}"),
        ("Easy requests",         lambda r: str(r["diff_easy"])),
        ("Medium requests",       lambda r: str(r["diff_medium"])),
        ("Hard requests",         lambda r: str(r["diff_hard"])),
        ("Engine steps",          lambda r: str(r["steps"])),
    ]
    for label, fn in rows:
        table.add_row(label, *[fn(r) for r in results])

    console.print(table)


def _print_plain(results: list[dict]) -> None:
    modes = " | ".join(r["mode"].upper() for r in results)
    print(f"\n{'='*60}")
    print(f"AgentServe Throughput Benchmark: {modes}")
    print(f"{'='*60}")
    fields = [
        ("Requests completed", "requests"),
        ("Wall time (s)",      "wall_time_s"),
        ("Throughput (tok/s)", "throughput_tps"),
        ("Mean latency (s)",   "mean_latency_s"),
        ("P50 latency (s)",    "p50_latency_s"),
        ("P95 latency (s)",    "p95_latency_s"),
        ("Mean TTFT (s)",      "mean_ttft_s"),
        ("Prefix hit rate",    "prefix_hit_rate"),
        ("Easy requests",      "diff_easy"),
        ("Medium requests",    "diff_medium"),
        ("Hard requests",      "diff_hard"),
        ("Engine steps",       "steps"),
    ]
    for label, key in fields:
        vals = []
        for r in results:
            v = r[key]
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        print(f"  {label:<25} {' | '.join(f'{v:>12}' for v in vals)}")
    print()


def main():
    parser = argparse.ArgumentParser(description="AgentServe throughput benchmark")
    parser.add_argument("--use-mock", action="store_true", default=False,
                        help="Use mock model (CPU, no GPU needed)")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct",
                        help="HuggingFace model name (ignored if --use-mock)")
    parser.add_argument("--num-requests", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=32,
                        help="Max tokens per request (keep low for benchmarking)")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--agent-aware", action="store_true", default=False,
                        help="Run agent-aware mode only")
    parser.add_argument("--baseline", action="store_true", default=False,
                        help="Run baseline mode only")
    parser.add_argument("--compare", action="store_true", default=False,
                        help="Run both modes and compare (default if neither flag set)")
    args = parser.parse_args()

    run_both = args.compare or (not args.agent_aware and not args.baseline)

    results = []
    kwargs = dict(
        use_mock=args.use_mock,
        model_name=args.model,
        num_requests=args.num_requests,
        max_tokens=args.max_tokens,
        max_batch_size=args.max_batch_size,
    )

    if args.agent_aware or run_both:
        print("Running agent-aware mode...")
        results.append(run_bench(agent_aware=True, **kwargs))

    if args.baseline or run_both:
        print("Running baseline (FIFO) mode...")
        results.append(run_bench(agent_aware=False, **kwargs))

    print_results(results)


if __name__ == "__main__":
    main()
