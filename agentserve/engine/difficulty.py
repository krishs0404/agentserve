"""
Request difficulty classifier.

Agents produce heterogeneous request streams: some calls are trivial
("classify this as positive/negative"), others are expensive ("write a
2,000-line module with tests"). A generic scheduler treats them the same.
Classifying difficulty lets us schedule easy requests first, which unblocks
downstream tool calls faster.

Classification is based on simple heuristic rules over the prompt text —
no ML model required, sub-millisecond cost.

Levels:
  EASY   — classify/extract/yes-no, expected output < 20 tokens
  MEDIUM — summarize, short code, explanation — 100 tokens expected
  HARD   — long code gen, multi-step planning, long prompts — 256 tokens expected
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class DifficultyLevel(Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


@dataclass
class Difficulty:
    level: DifficultyLevel
    estimated_output_tokens: int
    priority: int   # 0 = highest urgency (easy), 2 = lowest (hard)


# Keywords that signal a cheap, narrow-output task
_EASY_KEYWORDS = [
    "classify",
    "label",
    "yes or no",
    "true or false",
    "is it",
    "answer with yes or no",
    "answer yes or no",
    "answer in one word",
    "answer in a single word",
    "answer with a single",
    "one word",
    "one sentence",
    "in one line",
    "extract the",
    "what is the sentiment",
    "sentiment analysis",
    "fill in the json",
    '"$schema"',
    '"type": "object"',
]

# Keywords that signal a heavy, long-output task
_HARD_KEYWORDS = [
    "write a function",
    "write a class",
    "write a program",
    "write a script",
    "write a module",
    "implement",
    "debug this",
    "fix the bug",
    "refactor",
    "optimize this",
    "multi-step",
    "step by step plan",
    "design a system",
    "architecture for",
    "create a rest api",
    "create an api",
    "write unit tests",
    "write tests",
    "generate tests",
]

# Token count above which a prompt is treated as hard (expensive prefill)
_HARD_PROMPT_TOKEN_THRESHOLD = 2000

# Rough word-to-token ratio for the threshold estimate
_WORDS_PER_TOKEN = 0.75


class RequestDifficultyClassifier:
    """Classifies a prompt string into EASY / MEDIUM / HARD."""

    def classify(self, prompt: str) -> Difficulty:
        p = prompt.lower()

        # Estimate prompt token length from word count (cheap heuristic)
        approx_tokens = int(len(prompt.split()) / _WORDS_PER_TOKEN)

        # Hard: explicit code-gen / long planning keywords OR very long prompt
        if approx_tokens > _HARD_PROMPT_TOKEN_THRESHOLD:
            return Difficulty(DifficultyLevel.HARD, estimated_output_tokens=256, priority=2)
        for kw in _HARD_KEYWORDS:
            if kw in p:
                return Difficulty(DifficultyLevel.HARD, estimated_output_tokens=256, priority=2)

        # Easy: narrow-output keywords
        for kw in _EASY_KEYWORDS:
            if kw in p:
                return Difficulty(DifficultyLevel.EASY, estimated_output_tokens=20, priority=0)

        # Easy: very short prompt (< 4 words) — terse input almost certainly expects a terse answer.
        # Use word count to avoid misclassifying short-but-substantive prompts.
        if len(prompt.split()) < 4:
            return Difficulty(DifficultyLevel.EASY, estimated_output_tokens=20, priority=0)

        # Default: medium
        return Difficulty(DifficultyLevel.MEDIUM, estimated_output_tokens=100, priority=1)
