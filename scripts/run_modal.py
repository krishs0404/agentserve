#!/usr/bin/env python3
"""
Modal benchmark runner for AgentServe.

Runs ablation + trajectory benchmarks on an A10G GPU and writes results
to notes/. Model weights are cached in a persistent Modal Volume so
subsequent runs skip the ~2.5 GB download.

Estimated cost: ~$0.60-0.80 per full run (45-60 minutes on A10G).

Setup (one time):
    pip install modal
    modal token new          # authenticates your account
    export HF_TOKEN=hf_...  # HuggingFace token with Llama 3.2 access

Run from project root:
    modal run scripts/run_modal.py

Just verify image + model download (no GPU benchmark):
    modal run scripts/run_modal.py --dry-run

Reduce scope for a quick sanity check (~5 min, ~$0.07):
    modal run scripts/run_modal.py --num-requests 20 --n-traj 5
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).parent.parent

# ── Modal app ──────────────────────────────────────────────────────────────────

app = modal.App("agentserve-bench")

# Persistent volume: weights are downloaded once and reused across invocations.
weights_vol = modal.Volume.from_name("agentserve-weights", create_if_missing=True)
results_vol = modal.Volume.from_name("agentserve-results", create_if_missing=True)

# Image: Python 3.11 + deps + local source embedded at build time.
# add_local_python_source embeds the agentserve package into the image so
# it is importable in every container without PYTHONPATH hacks.
# add_local_dir copies scripts/ so we can run them directly with subprocess.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch==2.4.0",
        "safetensors>=0.4.0",
        "transformers>=4.40.0",
        "huggingface-hub>=0.22.0",
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.29.0",
        "pydantic>=2.0.0",
        "rich>=13.0.0",
        "numpy>=1.26.0",
        "matplotlib>=3.8.0",
        "psutil>=5.9.0",
    ])
    .add_local_python_source("agentserve")
    .add_local_dir("scripts", remote_path="/agentserve_scripts")
)

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_DIR = "/weights/llama-3.2-1b"
RESULTS_DIR = "/results"

# ── GPU benchmark function ─────────────────────────────────────────────────────

@app.function(
    gpu="A10G",
    image=image,
    volumes={
        "/weights": weights_vol,
        "/results": results_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200,
    memory=32768,
)
def run_benchmarks(
    num_requests: int = 100,
    max_tokens: int = 64,
    max_batch: int = 16,
    n_traj: int = 20,
    compare_vllm: bool = False,
    dry_run: bool = False,
) -> dict:
    """Run ablation + trajectory benchmarks on GPU. Returns combined results dict."""
    import subprocess
    import sys
    import json
    import os
    from pathlib import Path

    # ── 1. Download model weights if not cached ────────────────────────────────
    model_dir = Path(MODEL_DIR)
    needs_download = not model_dir.exists() or not list(model_dir.glob("*.safetensors"))

    if needs_download:
        print("Downloading model weights (cached for future runs)...")
        hf_token = os.environ.get("HF_TOKEN", "")
        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN not set. Run: export HF_TOKEN=hf_... then modal run again."
            )
        model_dir.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(model_dir),
            allow_patterns=["*.safetensors", "*.json"],
            token=hf_token,
        )
        weights_vol.commit()
        print(f"Model cached at {model_dir}")
    else:
        shards = list(model_dir.glob("*.safetensors"))
        mb = sum(f.stat().st_size for f in shards) / 1e6
        print(f"Using cached model ({mb:.0f} MB, {len(shards)} shard(s))")

    if dry_run:
        print("\nDry run complete — image and model are ready.")
        return {"dry_run": True, "model_dir": str(model_dir)}

    results: dict = {}

    # ── 2. Ablation benchmark ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"ABLATION BENCHMARK  ({num_requests} req × 5 modes)")
    print(f"{'='*60}")

    ablation_json = f"{RESULTS_DIR}/ablation.json"
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    ablation_cmd = [
        sys.executable, "/agentserve_scripts/bench_ablation.py",
        "--model-dir", str(model_dir),
        "--model-size", "1b",
        "--num-requests", str(num_requests),
        "--max-tokens", str(max_tokens),
        "--max-batch", str(max_batch),
        "--output-json", ablation_json,
    ]
    if compare_vllm:
        ablation_cmd.append("--compare-vllm")

    subprocess.run(ablation_cmd)

    try:
        with open(ablation_json) as f:
            results["ablation"] = json.load(f)
    except FileNotFoundError:
        results["ablation_error"] = "output JSON was not produced"

    # ── 3. Trajectory benchmark ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"TRAJECTORY BENCHMARK  ({n_traj} traj × 4 templates × 4 policies)")
    print(f"{'='*60}")

    traj_out_dir = f"{RESULTS_DIR}/traj_plots"
    Path(traj_out_dir).mkdir(parents=True, exist_ok=True)

    subprocess.run([
        sys.executable, "/agentserve_scripts/bench_trajectories.py",
        "--model-dir", str(model_dir),
        "--n-traj", str(n_traj),
        "--out", traj_out_dir,
        "--estimated-tps", "400",
        "--max-batch", "4",   # smaller batch = less peak VRAM during long trajectories
    ])

    try:
        traj_json = f"{traj_out_dir}/trajectory_results.json"
        with open(traj_json) as f:
            results["trajectories"] = json.load(f)
    except FileNotFoundError:
        results["traj_error"] = "trajectory JSON was not produced"

    results_vol.commit()
    return results


# ── Local entrypoint ───────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    dry_run: bool = False,
    num_requests: int = 100,
    max_tokens: int = 64,
    max_batch: int = 16,
    n_traj: int = 20,
    compare_vllm: bool = False,
):
    """
    Launch AgentServe benchmarks on Modal A10G.

    Pass --dry-run to just build the image and cache the model without running
    benchmarks. Useful for verifying setup before a full run.
    """
    if not dry_run:
        print("AgentServe Benchmark Run")
        print(f"  Ablation  : {num_requests} requests × 5 modes (a–e), batch={max_batch}")
        print(f"  Trajectory: {n_traj} trajectories × 4 templates × 4 policies")
        print(f"  GPU       : A10G (24 GB VRAM)")
        print(f"  Est. time : 40-60 min")
        print(f"  Est. cost : ~$0.60-0.80")
        print()

    results = run_benchmarks.remote(
        num_requests=num_requests,
        max_tokens=max_tokens,
        max_batch=max_batch,
        n_traj=n_traj,
        compare_vllm=compare_vllm,
        dry_run=dry_run,
    )

    if dry_run:
        print("Dry run finished. Run without --dry-run to execute benchmarks.")
        return

    # Save results locally under notes/
    notes_dir = PROJECT_ROOT / "notes"
    notes_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = notes_dir / f"results_{timestamp}.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_file}")

    _print_summary(results)


def _print_summary(results: dict) -> None:
    if "ablation" in results:
        modes = results["ablation"]
        print("\n── Ablation Results ──────────────────────────────────────────")
        print(f"  {'Mode':<26}  {'Easy lat':>9}  {'Hard lat':>9}  {'TPS':>7}  {'TTFT':>7}")
        print(f"  {'-'*26}  {'-'*9}  {'-'*9}  {'-'*7}  {'-'*7}")
        for m in modes:
            if not isinstance(m, dict) or "label" not in m:
                continue
            print(
                f"  {m['label']:<26}  "
                f"{m.get('easy_mean_lat_s', 0):>9.3f}s  "
                f"{m.get('hard_mean_lat_s', 0):>9.3f}s  "
                f"{m.get('throughput_tps', 0):>7.0f}  "
                f"{m.get('mean_ttft_s', 0):>7.3f}s"
            )

    if "trajectories" in results and "stats" in results.get("trajectories", {}):
        stats = results["trajectories"]["stats"]
        print("\n── Trajectory P50 / P99 TCT ──────────────────────────────────")
        print(f"  {'Policy':<16}  {'Template':<14}  {'P50':>7}  {'P99':>7}  {'Miss%':>6}")
        print(f"  {'-'*16}  {'-'*14}  {'-'*7}  {'-'*7}  {'-'*6}")
        for policy, by_tmpl in stats.items():
            for tmpl, s in by_tmpl.items():
                miss = s.get("miss_rate", 0) * 100
                print(
                    f"  {policy:<16}  {tmpl:<14}  "
                    f"{s.get('p50', 0):>7.2f}s  "
                    f"{s.get('p99', 0):>7.2f}s  "
                    f"{miss:>5.1f}%"
                )
    print()
