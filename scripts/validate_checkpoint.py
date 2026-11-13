#!/usr/bin/env python3
"""Check a local supported Llama checkpoint against Hugging Face before timing it."""

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import replace
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentserve.model.config import ModelConfig
from agentserve.model.llama import LlamaModel
from agentserve.model.loader import load_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("/tmp/agentserve-checkpoint-parity.json"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    config_bytes = (args.model_dir / "config.json").read_bytes()
    cfg = ModelConfig.from_hf_dict(json.loads(config_bytes))
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    prompts = [
        "Explain why scheduling affects latency.",
        "An agent reads a file, calls a tool, and then writes tests. " * 8,
    ]
    sequences = [tokenizer.encode(p, add_special_tokens=True) for p in prompts]
    limit = max(map(len, sequences)) + args.steps
    if limit > cfg.max_seq_len:
        raise ValueError("Validation input exceeds checkpoint context limit")
    torch.set_float32_matmul_precision("highest")
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = LlamaModel(replace(cfg, max_seq_len=limit)).eval()
    load_weights(model, str(args.model_dir))
    model = model.to(args.device)
    reference = (
        AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            local_files_only=True,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        .to(args.device)
        .eval()
    )
    report = {
        "kind": "checkpoint_numerical_validation",
        "passed": False,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "weights": {p.name: p.stat().st_size for p in sorted(args.model_dir.glob("*.safetensors"))},
        "dtype": "float32",
        "atol": 2e-4,
        "rtol": 2e-4,
        "initial_token_ids": [list(seq) for seq in sequences],
        "checks": [],
    }
    caches = []

    def check(actual, expected, phase, request):
        report["checks"].append(
            {
                "phase": phase,
                "request": request,
                "max_abs_error": float((actual - expected).abs().max()),
                "greedy_match": bool(torch.equal(actual.argmax(-1), expected.argmax(-1))),
            }
        )
        torch.testing.assert_close(actual, expected, atol=report["atol"], rtol=report["rtol"])
        if not report["checks"][-1]["greedy_match"]:
            raise AssertionError("Greedy tokens diverged")

    try:
        with torch.inference_mode():
            for i, seq in enumerate(sequences):
                actual, cache = model.forward(seq)
                expected = reference(
                    torch.tensor([seq], device=args.device), use_cache=False
                ).logits[0]
                check(actual, expected, "prefill", i)
                caches.append(cache)
            for step in range(args.steps):
                # Identical reference-greedy inputs; every position's argmax is checked.
                next_tokens = []
                for seq in sequences:
                    logits = reference(
                        torch.tensor([seq], device=args.device), use_cache=False
                    ).logits[0, -1]
                    next_tokens.append(int(logits.argmax()))
                actual, caches = model.forward_decode_batch(
                    next_tokens, caches, list(map(len, sequences))
                )
                for i, token in enumerate(next_tokens):
                    sequences[i].append(token)
                    expected = reference(
                        torch.tensor([sequences[i]], device=args.device), use_cache=False
                    ).logits[0, -1]
                    check(actual[i], expected, f"decode_{step}", i)
        report["passed"] = True
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {len(report['checks'])} logit and greedy-token comparisons; {args.out}")


if __name__ == "__main__":
    main()
