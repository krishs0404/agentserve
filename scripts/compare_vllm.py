"""
Compare AgentServe vs vLLM on the same workload.

Runs the same set of prompts through both engines (AgentServe mock + vLLM if
available) and produces a comparison table and optional matplotlib plots.

vLLM comparison requires vLLM installed and a GPU:
  pip install vllm
  uv run python scripts/compare_vllm.py --model meta-llama/Llama-3.2-1B-Instruct

For CPU-only comparison (both use mock / random logits):
  uv run python scripts/compare_vllm.py --use-mock
"""

import argparse
import sys
import os
import time
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentserve.model.config import TinyConfig
from agentserve.engine.engine import Engine
from agentserve.engine.request import Request


PROMPTS = [
    "Classify as POSITIVE or NEGATIVE. One word. 'Great product!'",
    "Yes or no: is 7 a prime number?",
    "Extract the version: 'Using torch==2.1.0'. Reply with version only.",
    "Write a Python function that implements binary search with type hints.",
    "Summarize in 2 sentences: REST is an architectural style for distributed systems.",
    "Implement a stack in Python with push, pop, and peek.",
    "True or false: the sun rises in the west.",
    "Write unit tests for: def add(a, b): return a + b",
    "Label as BUG or FEATURE: 'Login fails on mobile'. One word.",
    "Design a rate-limiting system for an API with 10k requests/sec.",
]


def run_agentserve(agent_aware: bool, use_mock: bool, model_name: str, max_tokens: int) -> dict:
    config = TinyConfig  # always use tiny for mock
    engine = Engine(config=config, use_mock=use_mock, agent_aware=agent_aware)

    requests = []
    for prompt in PROMPTS:
        token_ids = [ord(c) % 256 for c in prompt]
        requests.append(Request(prompt=prompt, token_ids=token_ids, max_tokens=max_tokens))

    t0 = time.monotonic()
    completed = engine.generate(requests)
    wall = time.monotonic() - t0

    latencies = [r.latency for r in completed if r.latency > 0]
    return {
        "engine": f"AgentServe ({'agent-aware' if agent_aware else 'baseline'})",
        "completed": len(completed),
        "wall_time_s": wall,
        "throughput_tps": sum(r.num_output_tokens for r in completed) / wall if wall > 0 else 0,
        "mean_latency_s": statistics.mean(latencies) if latencies else 0,
        "p95_latency_s": sorted(latencies)[int(len(latencies)*0.95)] if len(latencies) >= 2 else 0,
        "prefix_hit_rate": engine.metrics.prefix_hit_rate,
    }


def run_vllm(model_name: str, max_tokens: int) -> dict | None:
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("vLLM not installed — skipping vLLM comparison.")
        return None

    llm = LLM(model=model_name)
    sp = SamplingParams(max_tokens=max_tokens, temperature=1.0)

    t0 = time.monotonic()
    outputs = llm.generate(PROMPTS, sp)
    wall = time.monotonic() - t0

    tokens_out = sum(len(o.outputs[0].token_ids) for o in outputs)
    return {
        "engine": "vLLM (baseline)",
        "completed": len(outputs),
        "wall_time_s": wall,
        "throughput_tps": tokens_out / wall if wall > 0 else 0,
        "mean_latency_s": wall / len(outputs),
        "p95_latency_s": 0,  # vLLM doesn't expose per-request timing easily
        "prefix_hit_rate": 0,
    }


def print_comparison(results: list[dict]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        console = Console()
        table = Table(title="AgentServe vs vLLM", box=box.ROUNDED)
        table.add_column("Metric", style="bold cyan")
        for r in results:
            table.add_column(r["engine"], justify="right")
        fields = [
            ("Completed",        "completed"),
            ("Wall time (s)",    "wall_time_s"),
            ("Throughput (tok/s)", "throughput_tps"),
            ("Mean latency (s)", "mean_latency_s"),
            ("P95 latency (s)",  "p95_latency_s"),
            ("Prefix hit rate",  "prefix_hit_rate"),
        ]
        for label, key in fields:
            vals = []
            for r in results:
                v = r[key]
                vals.append(f"{v:.3f}" if isinstance(v, float) else str(v))
            table.add_row(label, *vals)
        console.print(table)
    except ImportError:
        for r in results:
            print(f"\n{r['engine']}:")
            for k, v in r.items():
                if k != "engine":
                    print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")


def maybe_plot(results: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    labels = [r["engine"] for r in results]
    throughputs = [r["throughput_tps"] for r in results]
    latencies   = [r["mean_latency_s"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(labels))
    axes[0].bar(x, throughputs, color=["#2ecc71", "#3498db", "#e74c3c"][:len(labels)])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_title("Throughput (tokens/s)")
    axes[0].set_ylabel("tok/s")

    axes[1].bar(x, latencies, color=["#2ecc71", "#3498db", "#e74c3c"][:len(labels)])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_title("Mean Latency (s)")
    axes[1].set_ylabel("seconds")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "results", "comparison.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Plot saved to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--use-mock", action="store_true", default=False)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--skip-vllm", action="store_true", default=False)
    parser.add_argument("--plot", action="store_true", default=False)
    args = parser.parse_args()

    results = []
    print("Running AgentServe (agent-aware)...")
    results.append(run_agentserve(True, args.use_mock, args.model, args.max_tokens))
    print("Running AgentServe (baseline FIFO)...")
    results.append(run_agentserve(False, args.use_mock, args.model, args.max_tokens))

    if not args.skip_vllm and not args.use_mock:
        print("Running vLLM...")
        r = run_vllm(args.model, args.max_tokens)
        if r:
            results.append(r)

    print_comparison(results)
    if args.plot:
        maybe_plot(results)


if __name__ == "__main__":
    main()
