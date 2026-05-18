"""
Trajectory workload generator for AgentServe benchmarks.

A trajectory is a sequence of LLM calls where each step depends on the
previous one (sequential dependency).  Four templates cover common
agentic patterns:

  react          — Thought → Action → Observation loop (3 steps)
  plan_execute   — Plan upfront, execute sub-tasks (4 steps)
  reflect        — Answer → Critique → Revise (3 steps)
  chat           — Multi-turn conversation (4 turns)
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import List


@dataclass
class StepSpec:
    prompt: str
    max_tokens: int = 100


@dataclass
class TrajectorySpec:
    trajectory_id: str
    template: str
    steps: List[StepSpec]

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def total_output_tokens(self) -> int:
        return sum(s.max_tokens for s in self.steps)


_QUESTIONS = [
    "what is the capital of France",
    "explain how neural networks learn",
    "describe the water cycle",
    "what is photosynthesis",
    "how does GPS work",
    "what is recursion in programming",
    "explain machine learning to a beginner",
    "what causes earthquakes",
    "how do vaccines work",
    "describe the theory of relativity",
]

_OBSERVATIONS = [
    "tool returned: Paris is the capital of France, population 2.1M",
    "tool returned: search result snippet about neural network backprop",
    "tool returned: Wikipedia excerpt on the water cycle",
    "tool returned: definition of photosynthesis from biology database",
]


class TrajectoryGenerator:
    """Generates batches of TrajectorySpec objects for a given template."""

    TEMPLATES = ("react", "plan_execute", "reflect", "chat")

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def generate(self, n: int, template: str) -> List[TrajectorySpec]:
        if template not in self.TEMPLATES:
            raise ValueError(f"Unknown template '{template}'. Choose from {self.TEMPLATES}")
        gen = getattr(self, f"_{template}")
        return [gen() for _ in range(n)]

    # ── Templates ────────────────────────────────────────────────────────────

    def _react(self) -> TrajectorySpec:
        q = self._rng.choice(_QUESTIONS)
        obs = self._rng.choice(_OBSERVATIONS)
        return TrajectorySpec(
            trajectory_id=str(uuid.uuid4()),
            template="react",
            steps=[
                StepSpec(
                    f"You are a ReAct agent. Question: '{q}'. "
                    "Think step-by-step and decide what tool to call next.",
                    max_tokens=80,
                ),
                StepSpec(
                    f"Observation: {obs}. "
                    "Continue reasoning based on this result. What is your next action?",
                    max_tokens=60,
                ),
                StepSpec(
                    "Based on all observations, synthesize a concise final answer.",
                    max_tokens=120,
                ),
            ],
        )

    def _plan_execute(self) -> TrajectorySpec:
        q = self._rng.choice(_QUESTIONS)
        return TrajectorySpec(
            trajectory_id=str(uuid.uuid4()),
            template="plan_execute",
            steps=[
                StepSpec(
                    f"Create a step-by-step plan to answer: '{q}'. List 3-5 concrete sub-tasks.",
                    max_tokens=150,
                ),
                StepSpec(
                    f"Execute sub-task 1 from the plan about '{q}'. Return the result concisely.",
                    max_tokens=100,
                ),
                StepSpec(
                    f"Execute sub-task 2 from the plan about '{q}'. Return the result concisely.",
                    max_tokens=100,
                ),
                StepSpec(
                    "Combine all sub-task results and produce the final answer.",
                    max_tokens=150,
                ),
            ],
        )

    def _reflect(self) -> TrajectorySpec:
        q = self._rng.choice(_QUESTIONS)
        return TrajectorySpec(
            trajectory_id=str(uuid.uuid4()),
            template="reflect",
            steps=[
                StepSpec(
                    f"Answer the following question thoroughly: '{q}'",
                    max_tokens=200,
                ),
                StepSpec(
                    "Critique the answer above: identify what is missing, incorrect, or unclear.",
                    max_tokens=100,
                ),
                StepSpec(
                    "Produce an improved, complete answer incorporating the critique.",
                    max_tokens=200,
                ),
            ],
        )

    def _chat(self) -> TrajectorySpec:
        msgs = self._rng.sample(_QUESTIONS, 3)
        return TrajectorySpec(
            trajectory_id=str(uuid.uuid4()),
            template="chat",
            steps=[
                StepSpec(f"User: {msgs[0]}\nAssistant:", max_tokens=80),
                StepSpec(f"User: can you elaborate on that?\nAssistant:", max_tokens=80),
                StepSpec(f"User: {msgs[1]}\nAssistant:", max_tokens=100),
                StepSpec("User: Can you summarize what we discussed?\nAssistant:", max_tokens=120),
            ],
        )
