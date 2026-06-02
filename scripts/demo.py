#!/usr/bin/env python3
"""
AgentServe Live Scheduling Demo

Two engines process the same agent workload side-by-side:
  Left  — FIFO baseline (no scheduling intelligence)
  Right — Agent-aware  (priority + overflow + preemption)

Watch easy requests (●) escape the queue faster on the right while
hard requests (■) keep the left-side queue backed up.

Usage:
    uv run python scripts/demo.py
    uv run python scripts/demo.py --mode priority     # compare specific mode
    uv run python scripts/demo.py --requests-per-sec 3

Press Ctrl+C to stop.
"""

from __future__ import annotations
import argparse
import os
import random
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from agentserve.engine.engine import Engine
from agentserve.engine.request import Request
from agentserve.model.config import TinyConfig

# ── Constants ──────────────────────────────────────────────────────────────────

# Each engine step sleeps this long — makes generation visible at human speed.
# easy  (20 tok): ~20 steps × 50ms = ~1.0s wall time
# hard (256 tok): ~256 steps × 50ms = ~12.8s wall time
STEP_DELAY_S = 0.05

DIFF_COLOR = {"easy": "green", "medium": "yellow", "hard": "red1"}
DIFF_ICON  = {"easy": "●",     "medium": "◆",      "hard": "■"}
MAX_TOKENS = {"easy": 20,      "medium": 100,       "hard": 256}

# Agent workload mix: 60% easy (classify/extract), 25% medium, 15% hard
WORKLOAD_MIX = ["easy"] * 12 + ["medium"] * 5 + ["hard"] * 3

PROMPTS = {
    "easy": [
        "Classify as POSITIVE or NEGATIVE. One word.\nReview: 'Fast shipping, great product!'",
        "True or false: Python lists are mutable. One word only.",
        "Extract the error code from: 'Error 404: Not Found'. Number only.",
        "Yes or no: Is the number 97 prime?",
        "Label as BUG, FEATURE, or DOCS: 'README missing install steps.' One word.",
        "Sentiment: 'API is fast but docs are confusing.' Positive/Negative/Mixed.",
    ],
    "medium": [
        "Summarize in 2 sentences: REST APIs use stateless HTTP to expose resources.",
        "Explain authentication vs authorisation in 3 sentences.",
        "What is the time complexity of quicksort? Explain in 2 sentences.",
        "List 3 common causes of N+1 query problems in ORMs. One sentence each.",
    ],
    "hard": [
        "Write a Python function implementing binary search with type hints and docstring.",
        "Implement a thread-safe LRU cache class in Python with get() and put() methods.",
        "Write a FastAPI route for user registration: validate input, hash password, return JWT.",
        "Design a rate limiter in Python using the token bucket algorithm. Production-quality.",
    ],
}


def _make_request(difficulty: str) -> Request:
    prompt = random.choice(PROMPTS[difficulty])
    return Request(
        prompt=prompt,
        token_ids=[ord(c) % 256 for c in prompt],
        max_tokens=MAX_TOKENS[difficulty],
    )


# ── Demo engine wrapper ────────────────────────────────────────────────────────

class DemoEngine:
    """
    Engine wrapper that runs in a background thread and exposes
    thread-safe state snapshots for the live display.
    """

    def __init__(
        self,
        label: str,
        agent_aware: bool,
        use_relative: bool = False,
        use_combined: bool = False,
    ):
        self.label = label
        self._engine = Engine(
            config=TinyConfig,
            use_mock=True,
            agent_aware=agent_aware or use_relative or use_combined,
            max_batch_size=8,
            max_prefill_per_step=4,
            enable_priority=agent_aware,
            enable_overflow=agent_aware,
            enable_preemption=agent_aware,
            use_relative_batching=use_relative,
            use_combined_batching=use_combined,
        )
        self._lock = threading.Lock()
        self._completed: list[Request] = []
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def submit(self, req: Request) -> None:
        with self._lock:
            self._engine.submit(req)

    def _run(self) -> None:
        while self._running:
            with self._lock:
                done = self._engine.step()
                self._completed.extend(done)
            time.sleep(STEP_DELAY_S)

    # ── Snapshot for rendering (no lock held for long) ─────────────────────

    def snapshot(self) -> dict:
        with self._lock:
            pending  = list(self._engine.scheduler.pending)
            decoding = list(self._engine.scheduler.decoding)
            done     = list(self._completed)
            metrics  = self._engine.metrics

        by_diff: dict[str, list[float]] = {"easy": [], "medium": [], "hard": []}
        for r in done:
            if r.latency > 0:
                by_diff.get(r.difficulty, []).append(r.latency)

        def mean(xs: list) -> float:
            return statistics.mean(xs) if xs else 0.0

        return {
            "pending":     pending,
            "decoding":    decoding,
            "n_done":      len(done),
            "easy_lat":    mean(by_diff["easy"]),
            "med_lat":     mean(by_diff["medium"]),
            "hard_lat":    mean(by_diff["hard"]),
            "throughput":  metrics.throughput_tokens_per_sec,
            "easy_count":  metrics.difficulty_counts.get("easy",   0),
            "hard_count":  metrics.difficulty_counts.get("hard",   0),
        }

    # ── Rich panel renderer ────────────────────────────────────────────────

    def render(self, width: int = 52) -> Panel:
        s = self.snapshot()

        body = Text()

        # ── Pending queue ──────────────────────────────────────────────────
        body.append(f"  Pending  ({len(s['pending']):>3})\n", style="bold")
        shown = 0
        for r in s["pending"]:
            if shown >= 40:
                body.append(f"  … +{len(s['pending']) - shown} more\n", style="dim")
                break
            icon  = DIFF_ICON.get(r.difficulty, "?")
            color = DIFF_COLOR.get(r.difficulty, "white")
            body.append(icon + " ", style=color)
            shown += 1
        if shown > 0 and shown < 40:
            body.append("\n")

        # ── Decode batch ───────────────────────────────────────────────────
        body.append(f"\n  Decode   ({len(s['decoding']):>3}/8)\n", style="bold")
        if not s["decoding"]:
            body.append("  —\n", style="dim")
        for r in s["decoding"][:8]:
            color  = DIFF_COLOR.get(r.difficulty, "white")
            done   = r.num_output_tokens
            total  = max(r.max_tokens, 1)
            filled = int(16 * done / total)
            bar    = "█" * filled + "░" * (16 - filled)
            body.append(f"  {bar} ", style=color)
            body.append(f"{r.difficulty[:4]} {done:3d}/{total}\n", style="dim")

        # ── Metrics ────────────────────────────────────────────────────────
        body.append("\n  ─── Metrics ──────────────────────\n", style="dim")

        def _lat_line(label: str, val: float, style: str) -> None:
            if val > 0:
                body.append(f"  {label:<18}", style="dim")
                body.append(f"{val:6.2f}s\n", style=style)
            else:
                body.append(f"  {label:<18}  —\n", style="dim")

        _lat_line("Easy latency",  s["easy_lat"], "green")
        _lat_line("Medium latency",s["med_lat"],  "yellow")
        _lat_line("Hard latency",  s["hard_lat"], "red1")

        body.append(f"  {'Completed':<18}{s['n_done']:>5}\n",    style="dim")

        # Panel border colour: blue for agent-aware, red for baseline
        is_baseline = "FIFO" in self.label or "Baseline" in self.label
        border = "red1" if is_baseline else "bright_blue"
        title  = f"[bold {'red1' if is_baseline else 'bright_blue'}]{self.label}[/]"

        return Panel(body, title=title, border_style=border,
                     width=width, height=28)


# ── Arrival generator ──────────────────────────────────────────────────────────

def _arrival_loop(
    engines: list[DemoEngine],
    requests_per_sec: float,
    seed: int = 42,
) -> None:
    rng = random.Random(seed)
    # Pre-burst: fill queues so the difference is immediately visible
    for _ in range(20):
        diff = rng.choice(WORKLOAD_MIX)
        base = _make_request(diff)
        for eng in engines:
            clone = Request(
                prompt=base.prompt,
                token_ids=list(base.token_ids),
                max_tokens=base.max_tokens,
            )
            eng.submit(clone)

    # Trickle requests at the configured rate
    while True:
        diff = rng.choice(WORKLOAD_MIX)
        base = _make_request(diff)
        for eng in engines:
            clone = Request(
                prompt=base.prompt,
                token_ids=list(base.token_ids),
                max_tokens=base.max_tokens,
            )
            eng.submit(clone)
        time.sleep(rng.expovariate(requests_per_sec))


# ── Main ───────────────────────────────────────────────────────────────────────

MODES = {
    "priority":  ("(d) All 3 Policies",    True,  False, False),
    "relative":  ("(e) Relative Batching", False, True,  False),
    "combined":  ("(f) Priority+Relative", True,  False, True),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="AgentServe live scheduling demo")
    ap.add_argument(
        "--mode", default="priority",
        choices=list(MODES.keys()),
        help="Which agent-aware mode to compare against FIFO",
    )
    ap.add_argument(
        "--requests-per-sec", type=float, default=1.5,
        help="Average new requests per second (default 1.5)",
    )
    args = ap.parse_args()

    label, agent_aware, use_rel, use_comb = MODES[args.mode]

    baseline = DemoEngine("(a) Baseline FIFO", agent_aware=False)
    aware    = DemoEngine(label, agent_aware=agent_aware,
                          use_relative=use_rel, use_combined=use_comb)

    baseline.start()
    aware.start()

    console = Console()
    console.print()
    console.print(
        "[bold]AgentServe — Live Scheduling Demo[/]  "
        f"[dim]comparing FIFO vs {label}[/]",
        justify="center",
    )
    console.print(
        "[dim]● easy  ◆ medium  ■ hard   |   Ctrl+C to exit[/]",
        justify="center",
    )
    console.print()

    arrival_thread = threading.Thread(
        target=_arrival_loop,
        args=([baseline, aware], args.requests_per_sec),
        daemon=True,
    )
    arrival_thread.start()

    try:
        with Live(console=console, refresh_per_second=10, screen=False) as live:
            while True:
                cols = Columns(
                    [baseline.render(width=55), aware.render(width=55)],
                    equal=True,
                )
                live.update(cols)
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        baseline.stop()
        aware.stop()

    # Final summary
    bl = baseline.snapshot()
    aw = aware.snapshot()
    console.print()
    console.print("[bold]Final Summary[/]")
    console.print(f"  {'Mode':<30}  {'Easy lat':>9}  {'Hard lat':>9}  {'Done':>6}")
    console.print(f"  {'-'*30}  {'-'*9}  {'-'*9}  {'-'*6}")
    for name, s in [("(a) Baseline FIFO", bl), (label, aw)]:
        easy = f"{s['easy_lat']:.2f}s" if s["easy_lat"] > 0 else "—"
        hard = f"{s['hard_lat']:.2f}s" if s["hard_lat"] > 0 else "—"
        console.print(f"  {name:<30}  {easy:>9}  {hard:>9}  {s['n_done']:>6}")
    console.print()


if __name__ == "__main__":
    main()
