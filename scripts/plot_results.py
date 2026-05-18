"""
Generate benchmark plots from bench_ablation.py --output-json output.

Usage:
  python scripts/plot_results.py --results notes/results_ablation.json
  python scripts/plot_results.py --results notes/results_with_vllm.json --out results/plots/

Produces:
  latency_cdf.png          — latency CDFs by difficulty class (easy / medium / hard)
  easy_vs_hard_latency.png — grouped bar: easy vs hard mean latency per mode
  throughput_bar.png       — throughput (tok/s) bar chart including vLLM if present
  ttft_bar.png             — mean TTFT bar chart
  wall_time_bar.png        — total wall time per mode
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 150,
})

# One consistent colour per scheduling mode across all plots
PALETTE = {
    "(a) Baseline FIFO":      "#e74c3c",
    "(b) Priority only":       "#f39c12",
    "(c) Priority + Overflow": "#27ae60",
    "(d) All 3 Policies":      "#2980b9",
    "vLLM (FIFO)":             "#8e44ad",
}
_DEFAULT = "#95a5a6"


def _color(label: str) -> str:
    return PALETTE.get(label, _DEFAULT)


def _load(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ── Plot 1: Latency CDFs by difficulty ────────────────────────────────────────

def plot_latency_cdfs(results: list[dict], out_dir: str) -> None:
    difficulties = [
        ("easy",   "easy_latencies",   "Easy Requests"),
        ("medium", "medium_latencies", "Medium Requests"),
        ("hard",   "hard_latencies",   "Hard Requests"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for ax, (_, key, title) in zip(axes, difficulties):
        plotted = False
        for r in results:
            lats = r.get(key, [])
            if not lats:
                continue
            xs = sorted(lats)
            ys = [(i + 1) / len(xs) for i in range(len(xs))]
            ax.plot(xs, ys, label=r["label"], color=_color(r["label"]),
                    linewidth=2, alpha=0.9)
            plotted = True

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Latency (s)", fontsize=10)
        ax.set_ylim(0, 1.05)
        if plotted:
            ax.legend(fontsize=7, loc="lower right")

    axes[0].set_ylabel("CDF", fontsize=10)
    fig.suptitle("Per-Request Latency CDFs by Difficulty Class", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, os.path.join(out_dir, "latency_cdf.png"))


# ── Plot 2: Easy vs Hard mean latency, grouped bars ───────────────────────────

def plot_easy_vs_hard(results: list[dict], out_dir: str) -> None:
    rows = [
        r for r in results
        if "easy_mean_lat_s" in r and "hard_mean_lat_s" in r
    ]
    if not rows:
        return

    labels = [r["label"] for r in rows]
    easy   = [r["easy_mean_lat_s"] for r in rows]
    hard   = [r["hard_mean_lat_s"] for r in rows]
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x - w / 2, easy, w, label="Easy",  color="#27ae60", alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + w / 2, hard, w, label="Hard",  color="#e74c3c", alpha=0.85, edgecolor="white")

    for bar in (*b1, *b2):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.001,
                    f"{h:.3f}s", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Mean Latency (s)", fontsize=11)
    ax.set_title("Easy vs Hard Request Latency by Scheduling Mode", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "easy_vs_hard_latency.png"))


# ── Plot 3: Throughput bar ────────────────────────────────────────────────────

def plot_throughput(results: list[dict], out_dir: str) -> None:
    rows   = [r for r in results if "throughput_tps" in r]
    labels = [r["label"] for r in rows]
    vals   = [r["throughput_tps"] for r in rows]
    colors = [_color(lbl) for lbl in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(labels)), vals, color=colors, alpha=0.85, edgecolor="white")

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Throughput (tokens / s)", fontsize=11)
    ax.set_title("Throughput Comparison", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "throughput_bar.png"))


# ── Plot 4: Mean TTFT bar ─────────────────────────────────────────────────────

def plot_ttft(results: list[dict], out_dir: str) -> None:
    rows   = [r for r in results if "mean_ttft_s" in r and r["mean_ttft_s"] > 0]
    if not rows:
        return
    labels = [r["label"] for r in rows]
    vals   = [r["mean_ttft_s"] for r in rows]
    colors = [_color(lbl) for lbl in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(labels)), vals, color=colors, alpha=0.85, edgecolor="white")

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.3f}s", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Mean TTFT (s)", fontsize=11)
    ax.set_title("Mean Time-to-First-Token (lower = better)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "ttft_bar.png"))


# ── Plot 5: Total wall time ───────────────────────────────────────────────────

def plot_wall_time(results: list[dict], out_dir: str) -> None:
    rows   = [r for r in results if "wall_s" in r]
    if not rows:
        return
    labels = [r["label"] for r in rows]
    vals   = [r["wall_s"] for r in rows]
    colors = [_color(lbl) for lbl in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(labels)), vals, color=colors, alpha=0.85, edgecolor="white")

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.1f}s", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Total Wall Time (s)", fontsize=11)
    ax.set_title("Task Completion Time (lower = better)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "wall_time_bar.png"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Plot AgentServe ablation results")
    p.add_argument("--results", required=True,
                   help="JSON file produced by bench_ablation.py --output-json")
    p.add_argument("--out", default=None,
                   help="Output directory (default: results/plots/ next to the JSON file)")
    args = p.parse_args()

    if not os.path.exists(args.results):
        print(f"ERROR: results file not found: {args.results}")
        sys.exit(1)

    results = _load(args.results)
    print(f"Loaded {len(results)} result sets from {args.results}")

    if args.out:
        out_dir = os.path.abspath(args.out)
    else:
        base = os.path.dirname(os.path.abspath(args.results))
        out_dir = os.path.join(base, "..", "results", "plots")
        out_dir = os.path.abspath(out_dir)

    os.makedirs(out_dir, exist_ok=True)
    print(f"Writing plots to {out_dir}/\n")

    plot_latency_cdfs(results, out_dir)
    plot_easy_vs_hard(results, out_dir)
    plot_throughput(results, out_dir)
    plot_ttft(results, out_dir)
    plot_wall_time(results, out_dir)

    print(f"\nDone. Open {out_dir}/ to view the plots.")


if __name__ == "__main__":
    main()
