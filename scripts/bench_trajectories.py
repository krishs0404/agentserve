#!/usr/bin/env python3
"""
Trajectory scheduling benchmark.

Generates multi-step agent trajectories (react / plan_execute / reflect / chat)
and measures trajectory completion time (TCT) under four scheduling policies:
  - fifo              : strict arrival-order FIFO
  - priority          : per-request difficulty priority
  - traj_progress     : prefer trajectories past their midpoint
  - traj_deadline     : urgency = remaining_work / time_remaining

Outputs:
  - Console table of P50 / P99 TCT and deadline-miss rate per policy × template
  - JSON with raw TCT arrays:  <out>/trajectory_results.json
  - Grouped bar plot of P99 TCT: <out>/trajectory_p99_tct.png

Usage:
  python scripts/bench_trajectories.py [--n-traj 30] [--out notes/plots/]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from time import monotonic

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentserve.engine.engine import Engine
from agentserve.engine.request import Request
from agentserve.engine.trajectory import TrajectoryGenerator, TrajectorySpec
from agentserve.engine.policies import (
    FifoPolicy,
    PriorityPolicy,
    TrajectoryProgressPolicy,
    TrajectoryDeadlinePolicy,
)
from agentserve.model.config import TinyConfig, Llama32_1B


# ── Constants ────────────────────────────────────────────────────────────────

TEMPLATES = ["react", "plan_execute", "reflect", "chat"]

POLICIES = {
    "fifo":          FifoPolicy(),
    "priority":      PriorityPolicy(),
    "traj_progress": TrajectoryProgressPolicy(),
    "traj_deadline": TrajectoryDeadlinePolicy(estimated_tps=500.0),
}

# Colour per policy (consistent across plots)
POLICY_COLOURS = {
    "fifo":          "#8c8c8c",
    "priority":      "#4c72b0",
    "traj_progress": "#dd8452",
    "traj_deadline": "#55a868",
}

DEADLINE_SLACK = 3.0     # deadline = slack × serial_completion_estimate


# ── Engine factory ───────────────────────────────────────────────────────────

def make_engine(policy=None, model_dir=None, estimated_tps=50.0, max_batch_size=4) -> Engine:
    use_mock = model_dir is None
    config = Llama32_1B if model_dir is not None else TinyConfig
    return Engine(
        config=config,
        use_mock=use_mock,
        agent_aware=True,
        max_batch_size=max_batch_size,
        max_prefill_per_step=min(4, max_batch_size),
        scheduler_policy=policy,
        model_dir=model_dir,
    )


# ── Trajectory runner ────────────────────────────────────────────────────────

def run_trajectory_workload(
    specs: list[TrajectorySpec],
    policy=None,
    model_dir=None,
    estimated_tps=50.0,
    max_batch_size=4,
) -> dict[str, dict]:
    """Run all trajectories with sequential step dependencies.

    Returns:
        dict trajectory_id -> {"tct": float, "template": str, "missed": bool}
    """
    engine = make_engine(policy, model_dir=model_dir, estimated_tps=estimated_tps,
                         max_batch_size=max_batch_size)

    # Build a lookup: trajectory_id -> TrajectorySpec (for step continuation)
    spec_map = {s.trajectory_id: s for s in specs}

    # pending_steps[tid] = list of (step_index, StepSpec) remaining after step 0
    pending_steps: dict[str, list] = {}
    tct_data: dict[str, dict] = {}

    # Submit step 0 of every trajectory
    t_start_wall = monotonic()
    for spec in specs:
        tid = spec.trajectory_id
        total_toks = spec.total_output_tokens
        deadline = t_start_wall + DEADLINE_SLACK * (total_toks / estimated_tps)

        step0 = spec.steps[0]
        req = _make_request(step0.prompt, step0.max_tokens, tid, 0, spec.num_steps, deadline)
        tct_data[tid] = {
            "start": req.arrival_time,
            "end": 0.0,
            "template": spec.template,
            "deadline": deadline,
        }
        pending_steps[tid] = list(enumerate(spec.steps[1:], start=1))
        engine.submit(req)

    # Step loop: run engine and submit continuations when steps complete
    for _ in range(1_000_000):
        if engine._is_idle():
            break
        completed = engine.step()
        for req in completed:
            tid = req.trajectory_id
            if tid is None:
                continue
            remaining = pending_steps.get(tid, [])
            if remaining:
                step_idx, next_spec = remaining.pop(0)
                spec = spec_map[tid]
                deadline = tct_data[tid]["deadline"]
                next_req = _make_request(
                    next_spec.prompt, next_spec.max_tokens,
                    tid, step_idx, spec.num_steps, deadline,
                )
                engine.submit(next_req)
            else:
                # Final step done — record trajectory completion time
                tct_data[tid]["end"] = req.done_time

    # Fill any trajectories that never finished (engine stalled)
    end_wall = monotonic()
    for tid, info in tct_data.items():
        if info["end"] == 0.0:
            info["end"] = end_wall  # worst-case

    # Compute derived fields
    results = {}
    for tid, info in tct_data.items():
        tct = info["end"] - info["start"]
        missed = info["end"] > info["deadline"]
        results[tid] = {"tct": tct, "template": info["template"], "missed": missed}

    return results


def _make_request(prompt, max_tokens, trajectory_id, step_index, total_steps, deadline) -> Request:
    return Request(
        prompt=prompt,
        token_ids=[ord(c) % 256 for c in prompt],
        max_tokens=max_tokens,
        trajectory_id=trajectory_id,
        step_index=step_index,
        total_steps=total_steps,
        deadline=deadline,
    )


# ── Metrics helpers ──────────────────────────────────────────────────────────

def compute_stats(results: dict[str, dict]) -> dict[str, dict]:
    """Group results by template and compute P50/P99/miss_rate."""
    by_template: dict[str, list] = defaultdict(list)
    for info in results.values():
        by_template[info["template"]].append(info)

    stats = {}
    for tmpl, items in by_template.items():
        tcts = [i["tct"] for i in items]
        misses = sum(1 for i in items if i["missed"])
        stats[tmpl] = {
            "p50": float(np.percentile(tcts, 50)),
            "p99": float(np.percentile(tcts, 99)),
            "mean": float(np.mean(tcts)),
            "miss_rate": misses / len(items),
            "n": len(items),
        }
    return stats


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_p99_tct(all_stats: dict[str, dict[str, dict]], out_path: str) -> None:
    """Grouped bar chart: P99 TCT per policy (groups) × template (bars)."""
    policy_names = list(all_stats.keys())
    n_policies = len(policy_names)
    n_templates = len(TEMPLATES)

    x = np.arange(n_templates)
    bar_width = 0.18
    offsets = np.linspace(-(n_policies - 1) / 2, (n_policies - 1) / 2, n_policies) * bar_width

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (policy, offset) in enumerate(zip(policy_names, offsets)):
        p99s = [all_stats[policy].get(tmpl, {}).get("p99", 0.0) for tmpl in TEMPLATES]
        bars = ax.bar(
            x + offset, p99s, bar_width,
            label=policy,
            color=POLICY_COLOURS.get(policy, f"C{i}"),
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, p99s):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=6.5,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(TEMPLATES, fontsize=11)
    ax.set_ylabel("P99 Trajectory Completion Time (s)", fontsize=11)
    ax.set_title("P99 TCT by Scheduling Policy × Trajectory Template", fontsize=13, pad=12)
    ax.legend(title="Policy", fontsize=9, title_fontsize=9)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AgentServe trajectory scheduling benchmark")
    parser.add_argument("--n-traj", type=int, default=30, help="Trajectories per template")
    parser.add_argument("--out", default="notes/plots", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", default=None, help="Path to real model weights (omit for mock)")
    parser.add_argument("--estimated-tps", type=float, default=500.0,
                        help="Tokens/sec estimate for deadline slack (real model ~500, mock ~50)")
    parser.add_argument("--max-batch", type=int, default=4,
                        help="Max decode batch size per engine step (lower = less VRAM)")
    args = parser.parse_args()

    # Rebuild deadline policy with the right TPS estimate
    POLICIES["traj_deadline"] = TrajectoryDeadlinePolicy(estimated_tps=args.estimated_tps)

    os.makedirs(args.out, exist_ok=True)

    gen = TrajectoryGenerator(seed=args.seed)

    # Pre-generate all trajectory specs (same workload for every policy)
    specs_by_template: dict[str, list[TrajectorySpec]] = {
        tmpl: gen.generate(args.n_traj, tmpl) for tmpl in TEMPLATES
    }
    all_specs = [s for specs in specs_by_template.values() for s in specs]

    model_label = args.model_dir or "mock"
    print(f"\nTrajectory benchmark [{model_label}]: "
          f"{args.n_traj} trajectories × {len(TEMPLATES)} templates × {len(POLICIES)} policies")
    print(f"Total trajectories per policy run: {len(all_specs)}\n")

    # Results: policy_name -> template -> stats
    all_stats: dict[str, dict[str, dict]] = {}
    raw_results: dict[str, dict] = {}

    header = f"{'Policy':<16} {'Template':<14} {'N':>4} {'P50(s)':>8} {'P99(s)':>8} {'Miss%':>7}"
    print(header)
    print("-" * len(header))

    for policy_name, policy_obj in POLICIES.items():
        t0 = time.monotonic()
        results = run_trajectory_workload(
            all_specs, policy=policy_obj,
            model_dir=args.model_dir,
            estimated_tps=args.estimated_tps,
            max_batch_size=args.max_batch,
        )
        elapsed = time.monotonic() - t0

        # Release CUDA memory between policy runs to avoid OOM from accumulation
        if args.model_dir is not None:
            try:
                import gc, torch
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass

        stats = compute_stats(results)
        all_stats[policy_name] = stats
        raw_results[policy_name] = {
            tid: info for tid, info in results.items()
        }

        for tmpl in TEMPLATES:
            s = stats.get(tmpl, {})
            print(
                f"{policy_name:<16} {tmpl:<14} {s.get('n', 0):>4} "
                f"{s.get('p50', 0):>8.3f} {s.get('p99', 0):>8.3f} "
                f"{s.get('miss_rate', 0) * 100:>6.1f}%"
            )
        print(f"  ({elapsed:.1f}s wall time)")

    print()

    # Plot
    plot_path = os.path.join(args.out, "trajectory_p99_tct.png")
    plot_p99_tct(all_stats, plot_path)

    # Save JSON
    json_path = os.path.join(args.out, "trajectory_results.json")
    with open(json_path, "w") as f:
        # raw_results contains Request objects — just save TCT/template/missed
        serialisable = {
            policy: {
                tid: {"tct": v["tct"], "template": v["template"], "missed": v["missed"]}
                for tid, v in per_policy.items()
            }
            for policy, per_policy in raw_results.items()
        }
        json.dump({"stats": all_stats, "per_trajectory": serialisable}, f, indent=2)
    print(f"  saved → {json_path}")


if __name__ == "__main__":
    main()
