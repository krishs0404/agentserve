#!/usr/bin/env python3
"""
Generate the four plots that tell the AgentServe story.

  1. ablation_latency.png   — easy vs hard latency across all 5 scheduling modes
                              (the headline result: what does agent-aware scheduling buy you?)
  2. trajectory_speedup.png — TCT speedup over FIFO per policy × template
                              (trajectory-aware policies: 6× for react, 2.6× for plan-execute)
  3. sensitivity_sweep.png  — scheduling benefit vs classifier noise
                              (robustness: still +65% benefit at 50% random labelling)
  4. latency_cdf.png        — easy-request latency CDF across all modes
                              (shows distribution shift, not just mean)

Usage:
    uv run python scripts/plot_all.py
    uv run python scripts/plot_all.py --ablation notes/results_latest.json
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family":        "sans-serif",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.22,
    "figure.dpi":         150,
})

ROOT   = Path(__file__).parent.parent
OUTDIR = ROOT / "notes" / "plots"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Consistent colours so every plot tells the same visual story
MODE_COLORS = {
    "(a) Baseline FIFO":      "#c0392b",
    "(b) Priority only":      "#e67e22",
    "(c) Priority + Overflow":"#27ae60",
    "(d) All 3 Policies":     "#2980b9",
    "(e) Relative Batching":  "#8e44ad",
}
POLICY_COLORS = {
    "fifo":          "#8c8c8c",
    "priority":      "#4c72b0",
    "traj_progress": "#dd8452",
    "traj_deadline": "#55a868",
}
TEMPLATES       = ["react",      "plan_execute",    "reflect",    "chat"]
TEMPLATE_LABELS = ["ReAct\n(3-step)", "Plan-Execute\n(4-step)",
                   "Reflect\n(3-step)", "Chat\n(4-step)"]


def _save(fig, name: str) -> None:
    path = OUTDIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.relative_to(ROOT)}")


# ── Plot 1: Easy vs Hard latency — the headline ablation ──────────────────────

def plot_ablation_latency(ablation_path: Path) -> None:
    data  = json.loads(ablation_path.read_text())
    modes = [m for m in data.get("ablation", data if isinstance(data, list) else [])
             if isinstance(m, dict) and "label" in m
             and m.get("easy_mean_lat_s", 0) > 0]
    if not modes:
        print("  [skip] no valid ablation modes found")
        return

    labels = [m["label"] for m in modes]
    easy   = [m["easy_mean_lat_s"] for m in modes]
    hard   = [m["hard_mean_lat_s"] for m in modes]
    x, w   = np.arange(len(labels)), 0.32

    fig, ax = plt.subplots(figsize=(12, 5.5))
    b1 = ax.bar(x - w/2, easy, w, label="Easy requests",
                color="#27ae60", alpha=0.88, edgecolor="white")
    b2 = ax.bar(x + w/2, hard, w, label="Hard requests",
                color="#c0392b", alpha=0.88, edgecolor="white")

    for bar in (*b1, *b2):
        h = bar.get_height()
        if h > 0.05:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.06,
                    f"{h:.2f}s", ha="center", va="bottom", fontsize=8)

    # Annotate best easy-latency improvement
    baseline_easy = easy[0]
    best_idx  = int(np.argmin(easy))
    best_easy = easy[best_idx]
    if baseline_easy > 0:
        pct = (1 - best_easy / baseline_easy) * 100
        ax.annotate(
            f"−{pct:.0f}% easy latency",
            xy=(best_idx - w/2, best_easy),
            xytext=(best_idx - w/2 + 0.5, best_easy + 1.8),
            arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.5),
            fontsize=9, color="#27ae60", fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("Mean Request Latency (s)", fontsize=11)
    ax.set_title(
        "Request Latency by Scheduling Mode\n"
        "A10G GPU · Llama 3.2-1B · 100 heterogeneous agent requests",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save(fig, "ablation_latency.png")
    print(f"    → easy −{pct:.0f}%  |  modes: {', '.join(labels)}")


# ── Plot 2: Trajectory speedup — the trajectory scheduling story ───────────────

def plot_trajectory_speedup(trajectory_path: Path) -> None:
    data   = json.loads(trajectory_path.read_text())
    stats  = data.get("trajectories", {}).get("stats", data.get("stats", {}))
    if not stats:
        print("  [skip] no trajectory stats found")
        return

    policies       = ["priority", "traj_progress", "traj_deadline"]
    policy_labels  = ["Priority", "Traj-Progress", "Traj-Deadline"]
    n = len(policies)
    x = np.arange(len(TEMPLATES))
    w = 0.22
    offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * w

    fig, ax = plt.subplots(figsize=(12, 5.5))

    for i, (policy, plabel) in enumerate(zip(policies, policy_labels)):
        speedups = []
        for tmpl in TEMPLATES:
            fifo_p50 = stats.get("fifo",   {}).get(tmpl, {}).get("p50", 0)
            pol_p50  = stats.get(policy,   {}).get(tmpl, {}).get("p50", 0)
            speedups.append(fifo_p50 / pol_p50 if pol_p50 > 0 else 1.0)

        bars = ax.bar(x + offsets[i], speedups, w,
                      label=plabel,
                      color=POLICY_COLORS.get(policy, "#ccc"),
                      alpha=0.88, edgecolor="white")
        for bar, val in zip(bars, speedups):
            if val > 1.05:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.03,
                        f"{val:.1f}×",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.axhline(1.0, color="#c0392b", linewidth=1.8, linestyle="--",
               label="FIFO baseline (1×)", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(TEMPLATE_LABELS, fontsize=10)
    ax.set_ylabel("TCT Speedup over FIFO (×)", fontsize=11)
    ax.set_title(
        "Trajectory Completion Time Speedup over FIFO\n"
        "A10G GPU · Llama 3.2-1B · 20 trajectories per template",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, title="Policy", title_fontsize=9)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    _save(fig, "trajectory_speedup.png")

    react_sp = (stats.get("fifo",{}).get("react",{}).get("p50",0) /
                max(stats.get("traj_progress",{}).get("react",{}).get("p50",1), 1e-9))
    pe_sp    = (stats.get("fifo",{}).get("plan_execute",{}).get("p50",0) /
                max(stats.get("traj_deadline",{}).get("plan_execute",{}).get("p50",1), 1e-9))
    print(f"    → react {react_sp:.1f}×  plan_execute {pe_sp:.1f}×")


# ── Plot 3: Sensitivity sweep — classifier robustness ─────────────────────────

def plot_sensitivity(sensitivity_path: Path) -> None:
    if not sensitivity_path.exists():
        print("  [skip] sensitivity_sweep.json not found")
        return

    data = json.loads(sensitivity_path.read_text())
    sweep = data["sweep"]
    noise_pct  = [s["noise_rate"] * 100 for s in sweep]
    benefit    = [s["easy_improvement_pct"] for s in sweep]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.fill_between(noise_pct, benefit, 0, alpha=0.12, color="#2980b9")
    ax.plot(noise_pct, benefit, "o-", color="#2980b9",
            linewidth=2.5, markersize=8, zorder=3)
    for x, y in zip(noise_pct, benefit):
        ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9.5,
                    fontweight="bold", color="#2980b9")

    ax.axhline(0, color="#c0392b", linewidth=1.5, linestyle="--",
               label="FIFO baseline (0% improvement)")
    ax.set_xlabel("Classifier Noise Rate (% of labels randomly flipped)", fontsize=10)
    ax.set_ylabel("Easy-Request Latency\nImprovement vs FIFO (%)", fontsize=10)
    ax.set_title(
        "Scheduling Benefit is Robust to Classifier Accuracy\n"
        "Benefit stays >60% even when half of all labels are random",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.set_xlim(-2, max(noise_pct) + 3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    _save(fig, "sensitivity_sweep.png")
    print(f"    → {benefit[0]:.0f}% at 0% noise  →  {benefit[-1]:.0f}% at {noise_pct[-1]:.0f}% noise")


# ── Plot 4: Easy-request latency CDF — shows distribution shift ───────────────

def plot_easy_cdf(ablation_path: Path) -> None:
    data  = json.loads(ablation_path.read_text())
    modes = [m for m in data.get("ablation", data if isinstance(data, list) else [])
             if isinstance(m, dict) and "label" in m and m.get("easy_latencies")]
    if not modes:
        print("  [skip] no easy latency data")
        return

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for m in modes:
        lats = sorted(m["easy_latencies"])
        if not lats:
            continue
        ys = [(i+1)/len(lats) for i in range(len(lats))]
        color = MODE_COLORS.get(m["label"], "#95a5a6")
        lw = 2.8 if m["label"] in ("(a) Baseline FIFO", "(e) Relative Batching") else 1.8
        ax.plot(lats, ys, label=m["label"], color=color, linewidth=lw, alpha=0.92)

    ax.set_xlabel("Request Latency (s)", fontsize=10)
    ax.set_ylabel("CDF", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Easy-Request Latency CDF by Scheduling Mode\n"
        "Left = faster. Agent-aware modes shift the entire distribution.",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=8.5, loc="lower right")
    plt.tight_layout()
    _save(fig, "latency_cdf.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation",    default=str(ROOT / "notes/results_20260531_151633.json"))
    ap.add_argument("--trajectory",  default=str(ROOT / "notes/results_20260531_154047.json"))
    ap.add_argument("--sensitivity", default=str(ROOT / "agentserve/engine/sensitivity_sweep.json"))
    args = ap.parse_args()

    print("Generating plots → notes/plots/\n")

    print("1. Ablation latency (easy vs hard, all modes):")
    plot_ablation_latency(Path(args.ablation))

    print("\n2. Trajectory speedup over FIFO:")
    plot_trajectory_speedup(Path(args.trajectory))

    print("\n3. Sensitivity sweep:")
    plot_sensitivity(Path(args.sensitivity))

    print("\n4. Easy-request latency CDF:")
    plot_easy_cdf(Path(args.ablation))

    print(f"\nDone.")


if __name__ == "__main__":
    main()
