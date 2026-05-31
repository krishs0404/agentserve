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
        # Detect multi-turn conversations by the presence of assistant or tool
        # role markers. In multi-turn mode the prompt is a growing context dump
        # (system prompt + prior turns + tool results) where accumulated length
        # tells us nothing about the CURRENT response difficulty — a 20K-token
        # conversation context can still produce a 30-token tool call next.
        is_multi_turn = "<assistant>" in prompt or "<tool>" in prompt

        # Classify on the tail of the prompt: this is always where the current
        # instruction lives, regardless of how long the conversation history is.
        classify_window = prompt[-800:] if len(prompt) > 800 else prompt
        p = classify_window.lower()

        # Length threshold only applies to true single-turn prompts. For multi-turn
        # conversations, skip it — prompt length is anti-correlated with output
        # difficulty (later turns have longer contexts but often shorter responses).
        if not is_multi_turn:
            full_tokens = int(len(prompt.split()) / _WORDS_PER_TOKEN)
            if full_tokens > _HARD_PROMPT_TOKEN_THRESHOLD:
                return Difficulty(DifficultyLevel.HARD, estimated_output_tokens=256, priority=2)

        for kw in _HARD_KEYWORDS:
            if kw in p:
                return Difficulty(DifficultyLevel.HARD, estimated_output_tokens=256, priority=2)

        for kw in _EASY_KEYWORDS:
            if kw in p:
                return Difficulty(DifficultyLevel.EASY, estimated_output_tokens=20, priority=0)

        if len(classify_window.split()) < 4:
            return Difficulty(DifficultyLevel.EASY, estimated_output_tokens=20, priority=0)

        return Difficulty(DifficultyLevel.MEDIUM, estimated_output_tokens=100, priority=1)
