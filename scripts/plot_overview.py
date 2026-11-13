#!/usr/bin/env python3
"""Render the README overview from committed trace-study metrics.

Run from any directory. Writes SVG and PNG; no model or inference required.
"""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
INK = "#172b40"
MUTED = "#526579"
BLUE = "#386ca3"
TEAL = "#237f75"


def main():
    summary = json.loads((ROOT / "experiments/length_prediction/summary.json").read_text())
    splits = summary["splits"]
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "svg.hashsalt": "agentserve-overview",
            "svg.fonttype": "none",
        }
    ):
        fig = plt.figure(figsize=(12, 5.25), facecolor="white")
        fig.text(0.045, 0.94, "AgentServe", fontsize=21, weight="bold")
        fig.text(
            0.045,
            0.882,
            "Scheduling experiments for multi-step agent workloads",
            fontsize=12,
            color=MUTED,
        )
        flow = fig.add_axes([0.035, 0.15, 0.47, 0.66])
        flow.set(xlim=(0, 10), ylim=(0, 7))
        flow.axis("off")
        flow.text(0.1, 7.1, "A   System overview", fontsize=11, weight="bold")

        def box(x, y, w, h, text, face="#f2f6fa", edge="#c9d6e2", size=10):
            flow.add_patch(
                FancyBboxPatch(
                    (x, y),
                    w,
                    h,
                    boxstyle="round,pad=0.06,rounding_size=0.13",
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=1,
                )
            )
            flow.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size)

        def arrow(start, end, **kw):
            flow.add_patch(
                FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=1.3,
                    color=MUTED,
                    **kw,
                )
            )

        flow.add_patch(
            FancyBboxPatch(
                (1.4, 2),
                7.1,
                3.35,
                boxstyle="round,pad=0.09,rounding_size=0.16",
                facecolor="#f7f9fc",
                edgecolor="#d3dde8",
            )
        )
        flow.text(1.65, 5.02, "PyTorch engine", color=MUTED, fontsize=8)
        box(2.2, 5.95, 5.5, 0.75, "Agent session / next LLM call", face="white")
        box(
            2.2,
            3.95,
            5.5,
            0.8,
            "Admission scheduler\nFIFO · priority · trajectory policies",
            size=9.5,
        )
        box(
            2.2,
            2.35,
            5.5,
            1.05,
            "Prefill + batched decode\nOrdinary / paged KV · prefix reuse",
            face="#eaf4f1",
            edge="#b4d4cd",
            size=9.5,
        )
        box(2.2, 0.45, 5.5, 0.85, "Response → agent / tools", face="white")
        arrow((4.95, 5.93), (4.95, 4.8))
        arrow((4.95, 3.94), (4.95, 3.45))
        arrow((4.95, 2.28), (4.95, 1.34))
        arrow((7.75, 0.87), (7.75, 6.32), connectionstyle="arc3,rad=0.55", linestyle="--")
        flow.text(
            9.2,
            3.4,
            "next turn\nafter tool wait",
            fontsize=8,
            ha="center",
            color=MUTED,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
        )
        flow.text(
            4.9,
            -0.05,
            "Conceptual loop; causal replay remains future work",
            fontsize=8,
            color=MUTED,
            ha="center",
        )

        ax = fig.add_axes([0.65, 0.255, 0.285, 0.46])
        fig.text(0.565, 0.827, "B   Can a cheap signal order requests?", fontsize=11, weight="bold")
        names = [
            ("median", "Training median", "#7a8997"),
            ("keyword", "Keywords", BLUE),
            ("previous_turn", "Previous turn", TEAL),
        ]
        ax.axvline(50, color="#a5b2be", linestyle="--", linewidth=1, zorder=0)
        for y, (key, label, color) in enumerate(names):
            values = [s["metrics"][key]["pair_order_accuracy"] * 100 for s in splits]
            mean = statistics.mean(values)
            offsets = [-0.13, -0.065, 0, 0.065, 0.13]
            ax.scatter(
                values, [y + o for o in offsets], color=color, s=26, alpha=0.65, linewidths=0
            )
            ax.scatter(
                [mean],
                [y],
                color=color,
                s=68,
                marker="D",
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            ax.text(
                max(values) + 1.2,
                y,
                f"{mean:.1f}%",
                va="center",
                fontsize=10,
                color=color,
                weight="bold",
            )
        ax.set(
            xlim=(45, 72),
            ylim=(2.55, -0.55),
            yticks=[0, 1, 2],
            yticklabels=[n[1] for n in names],
            xticks=[45, 50, 55, 60, 65, 70],
        )
        ax.set_xlabel("Pair ordering accuracy (%) ↑", labelpad=10)
        ax.tick_params(axis="both", length=0, labelcolor=MUTED, labelsize=9, pad=8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#d5dde6")
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#edf1f5")
        fig.text(
            0.65,
            0.117,
            "Dots: held-out splits   ◆ Mean   |   Ties: ½ credit",
            fontsize=8,
            color=MUTED,
        )
        fig.text(
            0.045,
            0.065,
            f"{summary['rows']:,} recorded turns · {summary['sessions']} sessions · {len(splits)} session-held-out splits",
            fontsize=9,
            weight="bold",
        )
        fig.text(
            0.045,
            0.025,
            "Trace statistics, not GPU speedups. Better ordering does not establish lower session latency.",
            fontsize=9,
            color=MUTED,
        )
        output = ROOT / "docs/figures"
        output.mkdir(exist_ok=True)
        fig.savefig(output / "overview.svg", facecolor="white", metadata={"Date": None})
        fig.savefig(
            output / "overview.png",
            dpi=180,
            facecolor="white",
            metadata={"Software": "AgentServe / Matplotlib"},
        )
        plt.close(fig)
        print("Wrote docs/figures/overview.svg and overview.png")


if __name__ == "__main__":
    main()
