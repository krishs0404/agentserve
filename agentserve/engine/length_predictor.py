"""
Online output-length predictor for relative batch scheduling.

Replaces the three-bin keyword classifier with a continuous estimate ŷ of
expected output tokens. The scheduler uses ŷ to group requests with similar
predicted lengths into the same decode batch, reducing KV-padding waste.

The predictor uses a small feature vector with hand-coded initial weights
and updates them online via SGD after every completed request. This means
it adapts to the actual workload being served — if the traffic shifts from
synthetic heterogeneous to SWE-bench medium/hard dominated, the predictor
learns that within a few hundred requests.
"""

from __future__ import annotations
import math
import re


# ── Feature extraction ─────────────────────────────────────────────────────────

_CODE_KW    = {"implement", "write a function", "write a class", "write a script",
               "write a program", "write a module", "refactor", "debug", "fix the bug",
               "write unit tests", "write tests", "generate tests", "create an api"}
_EXPLAIN_KW = {"explain", "describe", "summarize", "what is", "how does", "why does",
               "walk me through", "break down", "elaborate"}
_LIST_KW    = {"list ", "enumerate", "what are the", "give me", "name the"}
_CONSTRAINT = {"in one word", "one word", "yes or no", "true or false", "answer with",
               "answer in one", "briefly", "concisely", "one sentence", "in one line",
               "single word"}


def featurize(prompt: str) -> dict[str, float]:
    """
    Return a fixed feature dict for a prompt string.

    Features are designed to be informative regardless of prompt length, so
    the predictor works for both single-turn and multi-turn conversations.
    """
    is_multi = "<assistant>" in prompt or "<tool>" in prompt
    # For multi-turn: classify on the tail where the current instruction lives
    window = prompt[-800:] if len(prompt) > 800 else prompt
    p = window.lower()

    words = window.split()
    n_words = max(len(words), 1)

    return {
        "log_prompt_len":  math.log1p(len(prompt.split())),   # full length for prefill cost
        "has_code":        float(any(kw in p for kw in _CODE_KW)),
        "has_explain":     float(any(kw in p for kw in _EXPLAIN_KW)),
        "has_list":        float(any(kw in p for kw in _LIST_KW)),
        "has_constraint":  float(any(kw in p for kw in _CONSTRAINT)),
        "ends_question":   float(window.rstrip().endswith("?")),
        "is_multi_turn":   float(is_multi),
        "short_tail":      float(n_words < 6),                 # very terse instruction
    }


N_FEATURES = 8


# ── Predictor ──────────────────────────────────────────────────────────────────

class OutputLengthPredictor:
    """
    Predicts expected output length (tokens) for a request.

    Initial weights are informed by our empirical findings:
      - Code generation  → ~256 tokens
      - Explanation      → ~120 tokens
      - List / summary   → ~80 tokens
      - Constrained      → ~20 tokens
      - Multi-turn tool  → ~100 tokens (median of real SWE-bench traces)

    Online SGD updates the weights after every completed request so the
    predictor adapts to the actual workload distribution.
    """

    # (feature_name → initial weight)
    _INIT_WEIGHTS: dict[str, float] = {
        "log_prompt_len":  5.0,    # longer prompt → somewhat longer output
        "has_code":        140.0,  # code gen is the heaviest task
        "has_explain":     70.0,   # explanations are moderately long
        "has_list":        40.0,   # lists are moderate
        "has_constraint":  -55.0,  # explicit brevity signals → short output
        "ends_question":   -20.0,  # direct question → often a short answer
        "is_multi_turn":   -10.0,  # tool calls in agentic loops are short
        "short_tail":      -25.0,  # terse instruction → terse response
    }
    _INIT_BIAS = 80.0
    _LR = 0.005          # conservative learning rate — stable but adapts over ~200 updates
    _MIN_PRED = 5.0      # never predict fewer than 5 tokens

    def __init__(self):
        self.weights: dict[str, float] = dict(self._INIT_WEIGHTS)
        self.bias: float = self._INIT_BIAS
        self.n_updates: int = 0

    # ── Public API ────────────────────────────────────────────────────────

    def predict(self, prompt: str) -> float:
        """Return predicted output token count for this prompt."""
        feats = featurize(prompt)
        raw = sum(self.weights[k] * v for k, v in feats.items()) + self.bias
        return max(self._MIN_PRED, raw)

    def update(self, prompt: str, actual_output_len: int) -> None:
        """SGD update: adjust weights toward actual observed output length."""
        feats = featurize(prompt)
        pred  = self.predict(prompt)
        err   = actual_output_len - pred          # positive → we under-predicted
        for k, v in feats.items():
            self.weights[k] += self._LR * err * v
        self.bias += self._LR * err
        self.n_updates += 1

    def bucket(self, predicted: float) -> str:
        """Map continuous prediction to a difficulty label (for metrics only)."""
        if predicted <= 40:
            return "easy"
        if predicted >= 150:
            return "hard"
        return "medium"
