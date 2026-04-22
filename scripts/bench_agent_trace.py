"""
Replay a recorded agent trace through the engine.

Headline metric: total wall-clock time from first request to last completion.
This is what the end user actually experiences — total task completion time.

Usage:
  uv run python scripts/bench_agent_trace.py --trace traces/synthetic_50.jsonl --agent-aware
  uv run python scripts/bench_agent_trace.py --trace traces/synthetic_50.jsonl --baseline
"""

import argparse
import json
import sys
import os
import time
import statistics
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentserve.engine.engine import Engine
from agentserve.engine.request import Request
from agentserve.model.config import TinyConfig


def load_trace(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r["arrival_delay_ms"])
    return records


def replay_trace(trace: list[dict], engine: Engine, use_mock: bool) -> dict:
    """
    Replay a trace through the engine respecting arrival_delay_ms.
    Runs engine steps in a tight loop; requests are injected at their
    scheduled arrival time relative to trace start.
    """
    if not trace:
        return {}

    t0 = time.monotonic()
    submitted = [False] * len(trace)
    all_requests: list[Request] = []

    # Pre-build Request objects
    for record in trace:
        prompt = record["prompt"]
        token_ids = [ord(c) % 256 for c in prompt]  # simple stand-in tokeniser
        req = Request(
            prompt=prompt,
            token_ids=token_ids,
            max_tokens=64,
        )
        all_requests.append(req)

    # Run loop: inject requests at scheduled delays, step engine
    max_wait = sum(r["arrival_delay_ms"] for r in trace) / 1000 + 60  # safety timeout

    while True:
        now = time.monotonic() - t0
        # Submit any requests whose arrival time has passed
        for i, record in enumerate(trace):
            if not submitted[i] and now * 1000 >= record["arrival_delay_ms"]:
                engine.submit(all_requests[i])
                submitted[i] = True

        engine.step()

        if engine._is_idle() and all(submitted):
            break
        if now > max_wait:
            print(f"WARNING: timeout after {max_wait:.0f}s")
            break

    total_wall = time.monotonic() - t0
    completed = [r for r in all_requests if r.is_done]
    latencies = [r.latency for r in completed if r.latency > 0]
    ttfts     = [r.ttft    for r in completed if r.ttft    > 0]
    output_tokens = sum(r.num_output_tokens for r in completed)

    by_diff = {}
    for r in completed:
        by_diff[r.difficulty] = by_diff.get(r.difficulty, 0) + 1

    return {
        "total_wall_time_s": total_wall,
        "requests_total":    len(trace),
        "requests_completed": len(completed),
        "output_tokens":     output_tokens,
        "throughput_tps":    output_tokens / total_wall if total_wall > 0 else 0,
        "mean_latency_s":    statistics.mean(latencies) if latencies else 0,
        "p95_latency_s":     sorted(latencies)[int(len(latencies)*0.95)] if len(latencies) >= 2 else 0,
        "mean_ttft_s":       statistics.mean(ttfts) if ttfts else 0,
        "prefix_hit_rate":   engine.metrics.prefix_hit_rate,
        "diff_easy":         by_diff.get("easy",   0),
        "diff_medium":       by_diff.get("medium", 0),
        "diff_hard":         by_diff.get("hard",   0),
    }


def print_result(label: str, result: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  Mode: {label}")
    print(f"{'='*60}")
    for key, val in result.items():
        if isinstance(val, float):
            print(f"  {key:<30} {val:.3f}")
        else:
            print(f"  {key:<30} {val}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Replay an agent trace through AgentServe")
    parser.add_argument("--trace", default="traces/synthetic_50.jsonl")
    parser.add_argument("--use-mock", action="store_true", default=True)
    parser.add_argument("--agent-aware", action="store_true", default=False)
    parser.add_argument("--baseline", action="store_true", default=False)
    parser.add_argument("--compare", action="store_true", default=False)
    args = parser.parse_args()

    trace_path = os.path.join(os.path.dirname(__file__), "..", args.trace)
    if not os.path.exists(trace_path):
        print(f"Trace file not found: {trace_path}")
        print("Run: uv run python scripts/generate_synthetic.py")
        sys.exit(1)

    trace = load_trace(trace_path)
    print(f"Loaded {len(trace)} requests from {trace_path}")

    run_both = args.compare or (not args.agent_aware and not args.baseline)

    if args.agent_aware or run_both:
        engine = Engine(config=TinyConfig, use_mock=args.use_mock, agent_aware=True)
        result = replay_trace(trace, engine, args.use_mock)
        print_result("AGENT-AWARE", result)

    if args.baseline or run_both:
        engine = Engine(config=TinyConfig, use_mock=args.use_mock, agent_aware=False)
        result = replay_trace(trace, engine, args.use_mock)
        print_result("BASELINE (FIFO)", result)


if __name__ == "__main__":
    main()
