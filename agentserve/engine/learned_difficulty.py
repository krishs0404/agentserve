"""
Learned output-length predictor for request difficulty classification.

Replaces the keyword heuristic in difficulty.py with a two-stage model:

  Stage 1 — Feature extraction (sub-millisecond, no ML model needed):
    Structural features computed directly from the prompt string:
    prompt length, question word count, imperative verb detection, code
    block presence, punctuation density, etc. These capture the same
    signal as keywords but generalize to novel phrasings.

  Stage 2 — Linear regression head:
    A small weight vector maps the feature vector to predicted output
    token count. Trained from real (prompt, actual_output_tokens) pairs
    collected during a GPU benchmark run.

  Online calibration:
    After each request completes, the calibrator updates a per-bucket
    correction factor (predicted / actual ratio). Future predictions in
    that bucket are scaled by the rolling correction. This lets the
    predictor adapt if the real workload distribution shifts.

The classifier implements the same interface as RequestDifficultyClassifier
so it can be swapped in without changing the engine or scheduler.

Training workflow:
    1. Run bench_ablation.py once with the real model to collect
       (prompt, actual_output_tokens) pairs (saved to JSON by collect_training_data).
    2. Call LearnedDifficultyClassifier.train(pairs) to fit the linear head.
    3. Call .save(path) / .load(path) to persist the weights.
    4. Instantiate with use_learned=True in the engine for subsequent runs.

Sensitivity analysis support:
    Pass noise_rate=0.1 to randomly flip 10% of predictions to a wrong
    bucket — useful for ablating how robust scheduling is to classifier error.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

from agentserve.engine.difficulty import Difficulty, DifficultyLevel


# ---------------------------------------------------------------------------
# Feature extraction (no ML dependency — runs in < 0.5 ms)
# ---------------------------------------------------------------------------

_IMPERATIVE_VERBS = {
    "write", "implement", "create", "build", "design", "generate", "make",
    "develop", "refactor", "optimize", "fix", "debug", "rewrite", "add",
    "convert", "translate", "list", "explain", "describe", "summarize",
    "classify", "extract", "identify", "label", "compare", "analyse", "analyze",
}

_QUESTION_WORDS = {"what", "why", "how", "when", "where", "who", "which", "is", "are", "does", "do"}

_CODE_SIGNALS = {"def ", "class ", "```", "function ", "import ", "return ", "->", "==", "!="}


def extract_features(prompt: str) -> list[float]:
    """
    Extract a fixed-length feature vector from a prompt string.

    Features (11 total):
      0  log(prompt_char_len)       — longer prompts → longer answers
      1  log(prompt_word_count)     — word-level length signal
      2  has_code_block             — ``` or indented code → usually long output
      3  has_code_signals           — def/class/import/etc in prompt
      4  first_verb_is_imperative   — "write a function" → hard; "classify" → easy
      5  question_word_ratio        — fraction of words that are question words
      6  sentence_count             — multiple sentences → richer context → longer reply
      7  avg_word_length            — longer words → more technical → longer reply
      8  has_constraint_words       — "in one word", "briefly", "concisely" → easy
      9  has_list_signal            — "list N", "enumerate", "bullet" → medium-hard
      10 prompt_ends_with_question  — final char is "?" → answer might be short
    """
    p = prompt.strip()
    words = p.split()
    n_words = max(len(words), 1)
    n_chars = max(len(p), 1)

    # 0, 1 — length signals
    log_chars = math.log1p(n_chars)
    log_words = math.log1p(n_words)

    # 2 — explicit code block
    has_code_block = float("```" in p)

    # 3 — code-like tokens anywhere in prompt
    p_lower = p.lower()
    has_code_signals = float(any(s in p for s in _CODE_SIGNALS))

    # 4 — first content word is an imperative verb
    first_word = words[0].lower().rstrip(".,!?:") if words else ""
    first_is_imperative = float(first_word in _IMPERATIVE_VERBS)

    # 5 — fraction of words that are question words
    q_count = sum(1 for w in words if w.lower().rstrip(".,!?:") in _QUESTION_WORDS)
    question_ratio = q_count / n_words

    # 6 — sentence count (rough: split on . ? !)
    sentences = [s.strip() for s in re.split(r"[.?!]+", p) if s.strip()]
    sentence_count = math.log1p(max(len(sentences), 1))

    # 7 — average word length (technical prompts have longer words)
    avg_word_len = sum(len(w) for w in words) / n_words / 10.0  # normalise to ~1

    # 8 — constraint words that signal a short expected output
    constraint_patterns = [
        "in one word", "in a single word", "one word", "one sentence",
        "in one line", "yes or no", "true or false", "briefly", "concisely",
        "answer with", "answer in",
    ]
    has_constraint = float(any(cp in p_lower for cp in constraint_patterns))

    # 9 — list/enumeration signals
    list_patterns = ["list ", "enumerate", "bullet", "step by step", "steps to"]
    has_list = float(any(lp in p_lower for lp in list_patterns))

    # 10 — ends with question mark
    ends_with_question = float(p.rstrip().endswith("?"))

    return [
        log_chars,
        log_words,
        has_code_block,
        has_code_signals,
        first_is_imperative,
        question_ratio,
        sentence_count,
        avg_word_len,
        has_constraint,
        has_list,
        ends_with_question,
    ]


N_FEATURES = 11


# ---------------------------------------------------------------------------
# Linear regression head (no external ML deps)
# ---------------------------------------------------------------------------

class LinearPredictor:
    """
    Tiny linear model: predicted_tokens = dot(weights, features) + bias.

    Trained with ordinary least squares via normal equations, so no gradient
    descent loop is needed. Works well with the small datasets (200-2000
    examples) we get from one GPU benchmark run.
    """

    def __init__(self):
        self.weights: list[float] = [0.0] * N_FEATURES
        self.bias: float = 80.0  # reasonable prior: ~80 tokens on average
        self.fitted: bool = False

    def predict(self, features: list[float]) -> float:
        raw = sum(w * f for w, f in zip(self.weights, features)) + self.bias
        return max(1.0, raw)  # output length is always >= 1

    def fit(self, X: list[list[float]], y: list[float]) -> None:
        """
        Fit using normal equations: w = (X^T X)^{-1} X^T y.

        Adds a bias column (all-ones) to X before solving.
        """
        n = len(X)
        if n < N_FEATURES + 2:
            raise ValueError(f"Need at least {N_FEATURES + 2} training examples, got {n}")

        # Augment with bias column
        Xa = [row + [1.0] for row in X]
        p = N_FEATURES + 1

        # Compute X^T X  (p × p)
        XtX = [[0.0] * p for _ in range(p)]
        for row in Xa:
            for i in range(p):
                for j in range(p):
                    XtX[i][j] += row[i] * row[j]

        # Regularise diagonal (ridge, λ=1) so we don't blow up on collinear features
        lam = 1.0
        for i in range(p):
            XtX[i][i] += lam

        # Compute X^T y  (p-vector)
        Xty = [0.0] * p
        for row, yi in zip(Xa, y):
            for i in range(p):
                Xty[i] += row[i] * yi

        # Solve (X^T X) w = X^T y via Gaussian elimination with partial pivoting
        sol = _solve_linear(XtX, Xty)

        self.weights = sol[:N_FEATURES]
        self.bias = sol[N_FEATURES]
        self.fitted = True

    def to_dict(self) -> dict:
        return {"weights": self.weights, "bias": self.bias, "fitted": self.fitted}

    @classmethod
    def from_dict(cls, d: dict) -> "LinearPredictor":
        lp = cls()
        lp.weights = d["weights"]
        lp.bias = d["bias"]
        lp.fitted = d["fitted"]
        return lp


def _solve_linear(A: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting. Operates in-place on copies."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    for col in range(n):
        # Partial pivot
        max_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[max_row] = M[max_row], M[col]

        pivot = M[col][col]
        if abs(pivot) < 1e-12:
            continue  # singular column — skip (regularisation handles this)

        for row in range(col + 1, n):
            factor = M[row][col] / pivot
            for j in range(col, n + 1):
                M[row][j] -= factor * M[col][j]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = M[i][n]
        for j in range(i + 1, n):
            x[i] -= M[i][j] * x[j]
        if abs(M[i][i]) > 1e-12:
            x[i] /= M[i][i]

    return x


# ---------------------------------------------------------------------------
# Online calibration
# ---------------------------------------------------------------------------

@dataclass
class _BucketCalibrator:
    """Rolling correction factor per difficulty bucket."""
    n: int = 0
    sum_ratio: float = 0.0   # sum of (actual / predicted)

    @property
    def correction(self) -> float:
        if self.n == 0:
            return 1.0
        return self.sum_ratio / self.n

    def update(self, predicted: float, actual: float) -> None:
        if predicted > 0:
            self.sum_ratio += actual / predicted
            self.n += 1


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

class LearnedDifficultyClassifier:
    """
    Drop-in replacement for RequestDifficultyClassifier.

    Uses structural features + a trained linear head to predict output token
    count, then maps the prediction to an easy/medium/hard label.

    Falls back gracefully to the keyword heuristic when unfitted (e.g., first
    run before training data is available).

    Args:
        noise_rate: Fraction of predictions to randomly flip to a wrong bucket.
                    Use for sensitivity analysis experiments (0.0 = clean).
        use_learned: If False, always uses the keyword heuristic (baseline mode).
    """

    # Bucket boundaries (in expected output tokens)
    EASY_MAX = 40      # <= 40 tokens → easy
    HARD_MIN = 150     # >= 150 tokens → hard

    def __init__(self, noise_rate: float = 0.0, use_learned: bool = True):
        self.noise_rate = noise_rate
        self.use_learned = use_learned
        self._predictor = LinearPredictor()
        self._calibrators: dict[str, _BucketCalibrator] = {
            "easy":   _BucketCalibrator(),
            "medium": _BucketCalibrator(),
            "hard":   _BucketCalibrator(),
        }
        # Fall back to keyword heuristic when not yet fitted
        from agentserve.engine.difficulty import RequestDifficultyClassifier
        self._fallback = RequestDifficultyClassifier()
        self._rng = random.Random(0)

    # ── Public API (matches RequestDifficultyClassifier) ──────────────────

    def classify(self, prompt: str) -> Difficulty:
        if not self.use_learned or not self._predictor.fitted:
            return self._fallback.classify(prompt)

        features = extract_features(prompt)
        raw_pred = self._predictor.predict(features)

        # Apply per-bucket calibration (correction is 1.0 until enough data)
        bucket = self._bucket_for(raw_pred)
        correction = self._calibrators[bucket].correction
        pred_tokens = raw_pred * correction

        # Optional noise injection for sensitivity experiments
        if self.noise_rate > 0 and self._rng.random() < self.noise_rate:
            buckets = ["easy", "medium", "hard"]
            buckets.remove(bucket)
            bucket = self._rng.choice(buckets)

        return self._bucket_to_difficulty(bucket, int(pred_tokens))

    def on_request_complete(self, prompt: str, actual_output_tokens: int) -> None:
        """
        Update online calibration after a request finishes.

        Call this from engine._ingest_incoming or engine.step after each
        completion to keep the correction factors fresh.
        """
        if not self._predictor.fitted:
            return
        features = extract_features(prompt)
        pred = self._predictor.predict(features)
        bucket = self._bucket_for(pred)
        self._calibrators[bucket].update(pred, actual_output_tokens)

    # ── Training ──────────────────────────────────────────────────────────

    def train(self, pairs: list[tuple[str, int]]) -> dict:
        """
        Fit the linear head from (prompt, actual_output_tokens) pairs.

        Args:
            pairs: list of (prompt_text, actual_token_count) tuples.

        Returns:
            dict with training stats (mae, rmse, per-bucket accuracy).
        """
        X = [extract_features(p) for p, _ in pairs]
        y = [float(t) for _, t in pairs]

        self._predictor.fit(X, y)

        # Compute training stats
        preds = [self._predictor.predict(x) for x in X]
        errors = [abs(p - t) for p, t in zip(preds, y)]
        mae = sum(errors) / len(errors)
        rmse = math.sqrt(sum(e**2 for e in errors) / len(errors))

        # Per-bucket accuracy (did we predict the right bucket?)
        correct = 0
        for (_, actual), pred in zip(pairs, preds):
            if self._bucket_for(pred) == self._bucket_for(actual):
                correct += 1
        bucket_accuracy = correct / len(pairs)

        return {
            "n_train": len(pairs),
            "mae_tokens": round(mae, 1),
            "rmse_tokens": round(rmse, 1),
            "bucket_accuracy": round(bucket_accuracy, 3),
        }

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        data = {
            "predictor": self._predictor.to_dict(),
            "calibrators": {
                k: {"n": c.n, "sum_ratio": c.sum_ratio}
                for k, c in self._calibrators.items()
            },
            "easy_max": self.EASY_MAX,
            "hard_min": self.HARD_MIN,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "LearnedDifficultyClassifier":
        data = json.loads(Path(path).read_text())
        obj = cls(**kwargs)
        obj._predictor = LinearPredictor.from_dict(data["predictor"])
        for k, c in data["calibrators"].items():
            obj._calibrators[k] = _BucketCalibrator(n=c["n"], sum_ratio=c["sum_ratio"])
        return obj

    # ── Internal helpers ──────────────────────────────────────────────────

    def _bucket_for(self, token_count: float) -> str:
        if token_count <= self.EASY_MAX:
            return "easy"
        if token_count >= self.HARD_MIN:
            return "hard"
        return "medium"

    def _bucket_to_difficulty(self, bucket: str, estimated_tokens: int) -> Difficulty:
        if bucket == "easy":
            return Difficulty(DifficultyLevel.EASY, estimated_tokens, priority=0)
        if bucket == "hard":
            return Difficulty(DifficultyLevel.HARD, estimated_tokens, priority=2)
        return Difficulty(DifficultyLevel.MEDIUM, estimated_tokens, priority=1)


# ---------------------------------------------------------------------------
# Training data collection helper
# ---------------------------------------------------------------------------

def collect_training_data(results_json_path: str | Path) -> list[tuple[str, int]]:
    """
    Extract (prompt, actual_output_tokens) pairs from a bench_ablation results JSON.

    The ablation benchmark saves per-request latency arrays but not raw prompts.
    This function extracts what's available. For richer training data, instrument
    the engine to log (prompt, num_output_tokens) per completed request directly.
    """
    data = json.loads(Path(results_json_path).read_text())
    pairs = []

    for mode in data:
        if not isinstance(mode, dict):
            continue
        # The ablation JSON has per-difficulty latency arrays but not prompts.
        # We use the known prompt bank to reconstruct approximate pairs.
        # For real training, instrument engine.py to emit these directly.
        n_easy   = mode.get("n_easy",   0)
        n_medium = mode.get("n_medium", 0)
        n_hard   = mode.get("n_hard",   0)

        # Approximate: use median latency × throughput to estimate token counts
        # This is a fallback — direct logging is more accurate
        tps = mode.get("throughput_tps", 100)
        for _ in range(n_easy):
            lat = mode.get("easy_mean_lat_s", 0.2)
            pairs.append(("__easy__", int(lat * tps)))
        for _ in range(n_medium):
            lat = mode.get("med_mean_lat_s", 0.5)
            pairs.append(("__medium__", int(lat * tps)))
        for _ in range(n_hard):
            lat = mode.get("hard_mean_lat_s", 1.0)
            pairs.append(("__hard__", int(lat * tps)))

    return pairs
