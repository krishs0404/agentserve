#!/usr/bin/env python3
"""
Train the learned difficulty classifier from GPU benchmark output.

Usage (after a full modal run has produced notes/results_*.json):
    python scripts/train_classifier.py notes/results_20240101_120000.json

Saves trained weights to agentserve/engine/classifier_weights.json.
Then runs a sensitivity sweep showing scheduling quality vs. classifier noise.

Output:
  - Training stats (MAE, RMSE, bucket accuracy vs. keyword heuristic)
  - Sensitivity table: noise_rate → scheduling benefit
  - agentserve/engine/classifier_weights.json (loadable by LearnedDifficultyClassifier)
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agentserve.engine.learned_difficulty import (
    LearnedDifficultyClassifier,
    extract_features,
)
from agentserve.engine.difficulty import RequestDifficultyClassifier
from scripts.bench_ablation import EASY, MEDIUM, HARD, make_workload


def build_training_pairs(n_per_class: int = 200, seed: int = 42) -> list[tuple[str, int]]:
    """
    Build (prompt, expected_output_tokens) training pairs from the known prompt bank.

    Uses the ground-truth max_tokens per difficulty class as the target label.
    When real GPU results are available, actual_output_tokens can replace these.
    """
    import random
    rng = random.Random(seed)
    pairs = []

    # Easy: ~20 tokens
    for _ in range(n_per_class):
        pairs.append((rng.choice(EASY), 20))

    # Medium: ~100 tokens
    for _ in range(n_per_class):
        pairs.append((rng.choice(MEDIUM), 100))

    # Hard: ~256 tokens
    for _ in range(n_per_class):
        pairs.append((rng.choice(HARD), 256))

    rng.shuffle(pairs)
    return pairs


def print_gpu_results_summary(results_path: Path) -> None:
    """
    Print scheduling headline numbers from a GPU results JSON.
    Training labels still use natural expected lengths (20/100/256 tokens);
    the GPU results validate the scheduling story, not the classifier targets.
    """
    try:
        data = json.loads(results_path.read_text())
    except Exception as e:
        print(f"  Could not read results JSON: {e}")
        return

    ablation = data.get("ablation", data if isinstance(data, list) else [])
    if not ablation:
        return

    baseline = next((m for m in ablation if isinstance(m, dict) and "Baseline" in m.get("label", "")), None)
    best     = next((m for m in ablation if isinstance(m, dict) and "All 3"    in m.get("label", "")), None)
    if baseline and best:
        easy_improvement = (1 - best["easy_mean_lat_s"] / baseline["easy_mean_lat_s"]) * 100
        tps_improvement  = (best["throughput_tps"] / baseline["throughput_tps"] - 1) * 100
        print(f"  GPU results: easy-latency improvement = {easy_improvement:.1f}%  "
              f"throughput improvement = {tps_improvement:.1f}%")
        print(f"  (Training labels use natural expected lengths: easy=20, medium=100, hard=256 tokens)")


def evaluate_keyword_classifier(pairs: list[tuple[str, int]]) -> dict:
    """Measure bucket accuracy of the existing keyword heuristic."""
    clf = RequestDifficultyClassifier()
    correct = 0
    for prompt, actual_tok in pairs:
        diff = clf.classify(prompt)
        predicted_bucket = diff.level.value
        if actual_tok <= 40:
            true_bucket = "easy"
        elif actual_tok >= 150:
            true_bucket = "hard"
        else:
            true_bucket = "medium"
        if predicted_bucket == true_bucket:
            correct += 1
    return {"bucket_accuracy": round(correct / len(pairs), 3), "n": len(pairs)}


def run_sensitivity_sweep(weights_path: Path, noise_rates: list[float]) -> list[dict]:
    """
    For each noise rate, run a mock ablation and measure scheduling benefit.
    Returns a list of dicts with noise_rate + easy/hard latency ratios.
    """
    from agentserve.engine.engine import Engine
    from agentserve.engine.request import Request
    from agentserve.model.config import TinyConfig
    import time, statistics

    N = 60
    MAX_TOK = 32

    def run_with_clf(clf, label):
        engine = Engine(
            config=TinyConfig,
            use_mock=True,
            agent_aware=True,
            max_batch_size=8,
        )
        # Monkey-patch the engine's classifier
        engine.classifier = clf

        tokenize = lambda t: [ord(c) % 256 for c in t]
        workload = make_workload(N, MAX_TOK, tokenize)
        fresh = [Request(prompt=r.prompt, token_ids=list(r.token_ids), max_tokens=r.max_tokens)
                 for r in workload]

        t0 = time.monotonic()
        completed = engine.generate(fresh)
        wall = time.monotonic() - t0

        by_diff = {"easy": [], "medium": [], "hard": []}
        for r in completed:
            by_diff.get(r.difficulty, []).append(r.latency)

        def mean(xs): return statistics.mean(xs) if xs else 0.0
        return {
            "label": label,
            "easy_lat": mean(by_diff["easy"]),
            "hard_lat": mean(by_diff["hard"]),
            "wall": wall,
        }

    # Baseline (FIFO, no classifier)
    from agentserve.engine.engine import Engine as _E
    baseline_engine = _E(config=TinyConfig, use_mock=True, agent_aware=False, max_batch_size=8)
    baseline_engine.classifier = RequestDifficultyClassifier()
    tokenize = lambda t: [ord(c) % 256 for c in t]
    wl = make_workload(N, MAX_TOK, tokenize)
    fresh = [Request(prompt=r.prompt, token_ids=list(r.token_ids), max_tokens=r.max_tokens) for r in wl]
    import time, statistics
    t0 = time.monotonic()
    comp = baseline_engine.generate(fresh)
    by_diff = {"easy": [], "hard": []}
    for r in comp:
        if r.difficulty == "easy": by_diff["easy"].append(r.latency)
        if r.difficulty == "hard": by_diff["hard"].append(r.latency)
    def mean(xs): return statistics.mean(xs) if xs else 0.0
    baseline_easy = mean(by_diff["easy"])
    baseline_hard  = mean(by_diff["hard"])

    results = []
    for rate in noise_rates:
        clf = LearnedDifficultyClassifier.load(weights_path, noise_rate=rate)
        r = run_with_clf(clf, f"noise={rate:.0%}")
        results.append({
            "noise_rate": rate,
            "easy_lat": r["easy_lat"],
            "hard_lat": r["hard_lat"],
            "easy_improvement_pct": round((1 - r["easy_lat"] / baseline_easy) * 100, 1) if baseline_easy > 0 else 0,
        })

    return results, baseline_easy, baseline_hard


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_json", nargs="?", default=None,
                   help="Path to notes/results_*.json from a GPU run (optional)")
    p.add_argument("--n-per-class", type=int, default=200,
                   help="Training examples per difficulty class (synthetic fallback)")
    p.add_argument("--real-pairs", default=None,
                   help="Path to notes/lmcache_training_pairs.jsonl for real-data training")
    p.add_argument("--out", default="agentserve/engine/classifier_weights.json",
                   help="Where to save trained weights")
    p.add_argument("--skip-sweep", action="store_true",
                   help="Skip the sensitivity sweep (faster)")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("─" * 60)
    if args.real_pairs:
        print(f"Loading real training pairs from {args.real_pairs}...")
        pairs = []
        with open(args.real_pairs) as f:
            for line in f:
                rec = json.loads(line.strip())
                pairs.append((rec["prompt"], rec["output_length"]))
        print(f"  {len(pairs)} real (prompt, output_length) pairs loaded")
        # Show distribution
        lens = sorted(p[1] for p in pairs)
        n = len(lens)
        easy   = sum(1 for x in lens if x <= 40)
        medium = sum(1 for x in lens if 40 < x < 150)
        hard   = sum(1 for x in lens if x >= 150)
        print(f"  Median={lens[n//2]}  P95={lens[int(n*0.95)]}  "
              f"easy={easy}({easy/n:.0%}) med={medium}({medium/n:.0%}) hard={hard}({hard/n:.0%})")
    else:
        print("Building training pairs from synthetic prompt bank...")
        pairs = build_training_pairs(n_per_class=args.n_per_class)
        print(f"  {len(pairs)} pairs ({args.n_per_class} per class)")

    if args.results_json:
        print(f"GPU results from {args.results_json}:")
        print_gpu_results_summary(Path(args.results_json))

    print()
    print("─" * 60)
    print("Training learned classifier...")
    clf = LearnedDifficultyClassifier(use_learned=True)
    stats = clf.train(pairs)
    print(f"  n_train        : {stats['n_train']}")
    print(f"  MAE            : {stats['mae_tokens']:.1f} tokens")
    print(f"  RMSE           : {stats['rmse_tokens']:.1f} tokens")
    print(f"  Bucket accuracy: {stats['bucket_accuracy']:.1%}")

    print()
    print("─" * 60)
    print("Evaluating keyword heuristic (baseline)...")
    kw_stats = evaluate_keyword_classifier(pairs)
    print(f"  Bucket accuracy: {kw_stats['bucket_accuracy']:.1%}")

    delta = stats["bucket_accuracy"] - kw_stats["bucket_accuracy"]
    sign = "+" if delta >= 0 else ""
    print(f"  Δ vs. learned  : {sign}{delta:.1%}")

    clf.save(out_path)
    print(f"\nWeights saved to {out_path}")

    if not args.skip_sweep:
        print()
        print("─" * 60)
        print("Sensitivity sweep: scheduling benefit vs. classifier noise...")
        noise_rates = [0.0, 0.1, 0.2, 0.3, 0.5]
        sweep, baseline_easy, baseline_hard = run_sensitivity_sweep(out_path, noise_rates)

        print(f"\n  {'Noise rate':<12}  {'Easy lat':>9}  {'Improvement':>12}  {'Hard lat':>9}")
        print(f"  {'FIFO baseline':<12}  {baseline_easy:>9.4f}s  {'(reference)':>12}  {baseline_hard:>9.4f}s")
        print(f"  {'-'*12}  {'-'*9}  {'-'*12}  {'-'*9}")
        for r in sweep:
            print(f"  {r['noise_rate']:<12.0%}  "
                  f"{r['easy_lat']:>9.4f}s  "
                  f"{r['easy_improvement_pct']:>+11.1f}%  "
                  f"{r['hard_lat']:>9.4f}s")

        sweep_path = out_path.parent / "sensitivity_sweep.json"
        sweep_path.write_text(json.dumps({
            "baseline_easy_lat": baseline_easy,
            "baseline_hard_lat": baseline_hard,
            "sweep": sweep,
        }, indent=2))
        print(f"\nSweep results saved to {sweep_path}")


if __name__ == "__main__":
    main()
