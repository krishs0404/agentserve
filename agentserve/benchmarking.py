"""Shared benchmark input preparation for local and packaged runners."""

from typing import Callable


def build_tokenizer(model_dir: str | None, use_mock: bool) -> Callable[[str], list[int]]:
    """Use synthetic character IDs only for explicit mock runs.

    Real-model runs require the model's tokenizer. Fail instead of silently
    benchmarking unrelated character IDs when tokenizer loading fails.
    """
    if use_mock:
        return lambda text: [ord(c) % 256 for c in text]
    if not model_dir:
        raise ValueError("Real-model tokenization requires a model directory.")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir)
    except Exception as exc:
        raise RuntimeError(f"Cannot load tokenizer for real-model benchmark: {model_dir}") from exc
    print(f"  Loaded tokenizer from {model_dir} (vocab size {tok.vocab_size})")
    return lambda text: tok.encode(text, add_special_tokens=False)
