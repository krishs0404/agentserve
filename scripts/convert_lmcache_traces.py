#!/usr/bin/env python3
"""
Convert the lmcache-agentic-traces HuggingFace dataset into two artifacts:

  1. traces/lmcache_N.jsonl  — trace file for bench_agent_trace.py
     Each line: {"prompt": "...", "arrival_delay_ms": ..., "max_tokens": ...}
     Multiple concurrent sessions are interleaved by absolute arrival time.

  2. notes/lmcache_training_pairs.jsonl  — (prompt_text, output_length) pairs
     for training the learned difficulty classifier on real agent data.

Usage:
    # Quick local run (100 sessions, no GPU):
    uv run python scripts/convert_lmcache_traces.py --n-sessions 100

    # Full dataset:
    uv run python scripts/convert_lmcache_traces.py --n-sessions 787

    # Filter to one workload type:
    uv run python scripts/convert_lmcache_traces.py --source swebench --n-sessions 200

Dataset: https://huggingface.co/datasets/sammshen/lmcache-agentic-traces
  787 sessions, 24,881 LLM iterations
  Sources: swebench (669), gaia (85), wildclaw (10)
  output_length: median 104 tokens, heavy tail to 11K+
  pre_gap: median 0.71s between tool calls (real inter-request timing)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _format_prompt(messages: list[dict]) -> str:
    """
    Serialize an OpenAI messages array to a flat prompt string.

    This preserves the full conversation prefix so that successive turns in a
    session share a common text prefix — exactly what the prefix cache needs.
    We truncate very long messages at 4000 chars to keep prompts tractable for
    the fake tokenizer (which maps every character to one token).
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content") or ""

        # tool_calls: format as a brief tool-call description
        if msg.get("tool_calls"):
            calls = msg["tool_calls"]
            if isinstance(calls, list):
                call_strs = []
                for tc in calls[:3]:   # at most 3 tool calls
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    name = fn.get("name", "tool")
                    call_strs.append(f"[call:{name}]")
                content = content + " ".join(call_strs)

        # truncate very long content
        if len(content) > 4000:
            content = content[:3900] + "...[truncated]"

        parts.append(f"<{role}> {content}")

    return "\n".join(parts)


def _session_arrival_times(iterations: list[dict], session_start_ms: float) -> list[float]:
    """
    Compute absolute arrival time (ms from benchmark start) for each iteration
    in a session. The first iteration starts at session_start_ms. Each subsequent
    iteration arrives after the previous one's pre_gap (tool execution time).

    We don't know the previous iteration's generation latency, so we use only
    pre_gap as the inter-request spacing (conservative: ignores generation time,
    which means concurrent sessions will overlap heavily — realistic for a busy
    serving system).
    """
    times = []
    t = session_start_ms
    for i, it in enumerate(iterations):
        times.append(t)
        # pre_gap is the tool execution time BEFORE this iteration.
        # After this iteration completes, the agent waits pre_gap[i+1] seconds
        # before the next iteration. We approximate total turn time as 2s
        # (generation) + next pre_gap.
        next_gap_s = iterations[i + 1]["pre_gap"] if i + 1 < len(iterations) else 0.0
        t += (2.0 + next_gap_s) * 1000  # ms
    return times


def load_dataset(n_sessions: int, source_filter: str | None) -> dict[str, list[dict]]:
    """
    Download and load the lmcache-agentic-traces dataset from HuggingFace.
    Returns a dict: session_id -> list of iterations (sorted by turn order).
    """
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        print("ERROR: 'datasets' package not installed.")
        print("Run: uv add datasets  or  pip install datasets")
        sys.exit(1)

    print("Loading sammshen/lmcache-agentic-traces from HuggingFace...")
    print("(First run downloads ~2.4 GB; cached on subsequent runs)")

    ds = hf_load("sammshen/lmcache-agentic-traces", split="train")
    print(f"Dataset loaded: {len(ds)} rows")

    # Group by session
    sessions: dict[str, list[dict]] = defaultdict(list)
    for row in ds:
        sid = row["session_id"]
        if source_filter and not sid.startswith(source_filter):
            continue
        sessions[sid].append(dict(row))

    # Sort iterations within each session (they're already ordered but enforce it)
    for sid in sessions:
        sessions[sid].sort(key=lambda r: len(r.get("input", [])))

    # Limit to n_sessions
    limited = dict(list(sessions.items())[:n_sessions])
    total_iters = sum(len(v) for v in limited.values())
    print(f"Selected {len(limited)} sessions, {total_iters} iterations")

    return limited


def build_trace(
    sessions: dict[str, list[dict]],
    max_iters_per_session: int = 20,
    session_stagger_ms: float = 500.0,
) -> list[dict]:
    """
    Convert sessions to a flat list of trace records sorted by arrival time.

    Sessions are staggered by session_stagger_ms so the serving system sees
    a realistic mix of new and in-flight requests simultaneously.
    """
    records = []
    for i, (sid, iters) in enumerate(sessions.items()):
        session_start = i * session_stagger_ms
        iters = iters[:max_iters_per_session]
        arrivals = _session_arrival_times(iters, session_start)

        for it, arrival_ms in zip(iters, arrivals):
            prompt = _format_prompt(it.get("input", []))
            output_len = it.get("output_length", 64)
            records.append({
                "session_id":       sid,
                "arrival_delay_ms": round(arrival_ms, 1),
                "prompt":           prompt,
                "max_tokens":       min(output_len, 256),  # cap for engine throughput
                "output_length":    output_len,
            })

    records.sort(key=lambda r: r["arrival_delay_ms"])
    return records


def build_training_pairs(sessions: dict[str, list[dict]]) -> list[dict]:
    """
    Extract (prompt, output_length) pairs for training the learned classifier.

    We use the LAST user message as the prompt (what the agent is currently
    responding to), paired with the real completion token count. This gives the
    classifier a realistic signal: short tool-call requests → short outputs,
    complex synthesis steps → long outputs.
    """
    pairs = []
    for sid, iters in sessions.items():
        for it in iters:
            messages = it.get("input", [])
            output_len = it.get("output_length", 0)
            if output_len == 0:
                continue

            # Find the last user/tool message as the "prompt" for classification
            last_user = ""
            for msg in reversed(messages):
                if msg.get("role") in ("user", "tool"):
                    content = msg.get("content") or ""
                    if content:
                        last_user = content[:2000]  # truncate
                        break

            if not last_user:
                continue

            pairs.append({
                "prompt":        last_user,
                "output_length": output_len,
                "session_id":    sid,
                "model":         it.get("model", ""),
            })

    return pairs


def main():
    p = argparse.ArgumentParser(description="Convert lmcache-agentic-traces to AgentServe formats")
    p.add_argument("--n-sessions", type=int, default=100,
                   help="Number of sessions to convert (max 787)")
    p.add_argument("--source", default=None, choices=["swebench", "gaia", "wildclaw"],
                   help="Filter to one source workload")
    p.add_argument("--max-iters", type=int, default=20,
                   help="Max iterations per session in the replay trace")
    p.add_argument("--stagger-ms", type=float, default=500.0,
                   help="Ms between session start times (controls concurrency)")
    p.add_argument("--trace-out", default=None,
                   help="Output JSONL trace path (default: traces/lmcache_N.jsonl)")
    p.add_argument("--pairs-out", default=None,
                   help="Output training pairs JSONL path (default: notes/lmcache_<N>_pairs.jsonl)")
    args = p.parse_args()

    # Resolve output paths
    root = Path(__file__).parent.parent
    trace_name = f"lmcache_{args.n_sessions}"
    if args.source:
        trace_name += f"_{args.source}"
    trace_out = Path(args.trace_out) if args.trace_out else root / "traces" / f"{trace_name}.jsonl"
    pairs_out = Path(args.pairs_out) if args.pairs_out else root / "notes" / f"lmcache_{trace_name}_pairs.jsonl"

    trace_out.parent.mkdir(parents=True, exist_ok=True)
    pairs_out.parent.mkdir(parents=True, exist_ok=True)

    # Load
    sessions = load_dataset(args.n_sessions, args.source)

    # Build and write trace
    print(f"\nBuilding replay trace (max {args.max_iters} iters/session)...")
    records = build_trace(sessions, args.max_iters, args.stagger_ms)
    with open(trace_out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Trace written: {trace_out}  ({len(records)} requests)")

    # Build and write training pairs
    print("Building classifier training pairs...")
    pairs = build_training_pairs(sessions)
    with open(pairs_out, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"Training pairs written: {pairs_out}  ({len(pairs)} pairs)")

    # Print output distribution stats
    output_lens = [p["output_length"] for p in pairs]
    if output_lens:
        output_lens.sort()
        n = len(output_lens)
        print(f"\nOutput length distribution ({n} pairs):")
        print(f"  Median : {output_lens[n // 2]} tokens")
        print(f"  Mean   : {sum(output_lens) / n:.0f} tokens")
        print(f"  P95    : {output_lens[int(n * 0.95)]} tokens")
        print(f"  Max    : {output_lens[-1]} tokens")
        easy  = sum(1 for x in output_lens if x <= 40)
        medium= sum(1 for x in output_lens if 40 < x < 150)
        hard  = sum(1 for x in output_lens if x >= 150)
        print(f"  Buckets: easy={easy} ({easy/n:.0%})  "
              f"medium={medium} ({medium/n:.0%})  "
              f"hard={hard} ({hard/n:.0%})")

    print("\nNext steps:")
    print("  # Run trace replay benchmark (mock model):")
    print(f"  uv run python scripts/bench_agent_trace.py --trace {trace_out.relative_to(root)} --compare")
    print("")
    print("  # Retrain classifier on real data:")
    print(f"  uv run python scripts/train_classifier.py --real-pairs {pairs_out.relative_to(root)}")


if __name__ == "__main__":
    main()
