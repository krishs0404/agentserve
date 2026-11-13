#!/usr/bin/env python3
"""CPU trace study; no inference, tokenization, or scheduling simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentserve.engine.difficulty import RequestDifficultyClassifier


def split_sessions(sessions, seed):
    ids = sorted(sessions)
    random.Random(seed).shuffle(ids)
    cut = max(1, int(len(ids) * 0.8))
    return set(ids[:cut]), set(ids[cut:])


def predictions(rows, median):
    """Input file order defines turn order; use only previously observed outputs."""
    classifier = RequestDifficultyClassifier()
    previous = {}
    for row in rows:
        sid = row["session_id"]
        yield (
            row,
            {
                "median": median,
                "keyword": classifier.classify(row["prompt"]).estimated_output_tokens,
                "previous_turn": previous.get(sid, median),
            },
        )
        previous[sid] = row["output_length"]


def pair_ordering(raw, trials=20000):
    """Cross-session random pairs, label ties excluded; prediction ties get half credit.

    This is a workload-wide diagnostic, not a claim that these requests overlap.
    """
    rng = random.Random(90210)
    scores = {p: 0.0 for p in raw[0]["predictions"]}
    ties = {p: 0 for p in scores}
    count = 0
    for _ in range(trials):
        a, b = rng.choices(raw, k=2)
        dy = a["output_length"] - b["output_length"]
        if a["session_id"] == b["session_id"] or dy == 0:
            continue
        count += 1
        for p in scores:
            dp = a["predictions"][p] - b["predictions"][p]
            if dp == 0:
                ties[p] += 1
                scores[p] += 0.5
            elif dp * dy > 0:
                scores[p] += 1
    return {
        p: {
            "pair_order_accuracy": scores[p] / count if count else None,
            "prediction_tie_rate": ties[p] / count if count else None,
            "evaluated_pairs": count,
        }
        for p in scores
    }


def evaluate(rows, train_ids, test_ids):
    median = statistics.median(r["output_length"] for r in rows if r["session_id"] in train_ids)
    errors = {p: defaultdict(list) for p in ("median", "keyword", "previous_turn")}
    long_errors = {p: [] for p in errors}
    # Tail threshold fixed using training data, never test labels.
    train_y = sorted(r["output_length"] for r in rows if r["session_id"] in train_ids)
    threshold = train_y[int(0.9 * (len(train_y) - 1))]
    raw = []
    for row, preds in predictions(rows, median):
        if row["session_id"] not in test_ids:
            continue
        y = row["output_length"]
        raw.append(
            {
                "session_id": row["session_id"],
                "model": row["model"],
                "output_length": y,
                "predictions": preds,
            }
        )
        for p, pred in preds.items():
            errors[p][row["session_id"]].append(abs(y - pred))
            if y >= threshold:
                long_errors[p].append(abs(y - pred))
    metrics = {}
    for p, sessions in errors.items():
        flat = [e for values in sessions.values() for e in values]
        metrics[p] = {
            "request_mae": statistics.mean(flat),
            "session_macro_mae": statistics.mean(statistics.mean(v) for v in sessions.values()),
            "training_p90_tail_mae": statistics.mean(long_errors[p]),
        }
    ordering = pair_ordering(raw)
    for p in metrics:
        metrics[p].update(ordering[p])
    # Paired bootstrap over sessions (not correlated turns), conditional on this split.
    ids = sorted(test_ids)
    rng = random.Random(1234)
    intervals = {}
    for p in ("keyword", "previous_turn"):
        differences = [
            statistics.mean(errors[p][s]) - statistics.mean(errors["median"][s]) for s in ids
        ]
        samples = sorted(statistics.mean(rng.choices(differences, k=len(ids))) for _ in range(1000))
        intervals[p] = {
            "delta_session_mae_vs_median": statistics.mean(differences),
            "bootstrap_95pct": [samples[24], samples[974]],
        }
    buckets = defaultdict(list)
    for r in raw:
        buckets[r["predictions"]["keyword"]].append(r["output_length"])
    bucket_summary = {
        str(k): {
            "count": len(v),
            "observed_median": statistics.median(v),
            "observed_mean": statistics.mean(v),
        }
        for k, v in sorted(buckets.items())
    }
    return {
        "keyword_buckets": bucket_summary,
        "training_median": median,
        "training_p90": threshold,
        "train_sessions": sorted(train_ids),
        "test_sessions": ids,
        "metrics": metrics,
        "paired_session_bootstrap": intervals,
    }, raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("notes/lmcache_lmcache_787_pairs.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("experiments/length_prediction"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    args = parser.parse_args()
    data = args.input.read_bytes()
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    sessions = {r["session_id"] for r in rows}
    if len(sessions) < 2:
        raise ValueError("Need at least two sessions")
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "kind": "recorded_trace_statistics",
        "input": str(args.input),
        "sha256": hashlib.sha256(data).hexdigest(),
        "python": platform.python_version(),
        "rows": len(rows),
        "sessions": len(sessions),
        "models": {},
        "splits": [],
    }
    for model in sorted({r["model"] for r in rows}):
        ys = [r["output_length"] for r in rows if r["model"] == model]
        summary["models"][model] = {"rows": len(ys), "median_output_tokens": statistics.median(ys)}
    with (args.out / "predictions.jsonl").open("w") as out:
        for seed in args.seeds:
            train, test = split_sessions(sessions, seed)
            result, raw = evaluate(rows, train, test)
            summary["splits"].append({"seed": seed, **result})
            for row in raw:
                out.write(json.dumps({"seed": seed, **row}) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                p: {
                    m: statistics.mean(s["metrics"][p][m] for s in summary["splits"])
                    for m in summary["splits"][0]["metrics"][p]
                }
                for p in ("median", "keyword", "previous_turn")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
