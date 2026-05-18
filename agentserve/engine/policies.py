"""
Plug-in scheduling policies for the AgentServe Scheduler.

Each policy implements SchedulerPolicy and controls two things:
  priority_key(req) -> tuple   — sort key; lower = schedule sooner
  on_request_complete(req)     — optional state update hook on completion

The Scheduler accepts a policy via its `policy` parameter.  When set,
it bypasses the built-in three-deque priority logic and uses a single
sorted-list pending queue keyed by priority_key().

Existing behaviour (baseline_mode / enable_priority / enable_overflow /
enable_preemption) is completely unaffected when policy=None.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import monotonic
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentserve.engine.request import Request


class SchedulerPolicy(ABC):
    """Interface for pluggable scheduling policies.

    Implement priority_key() to define ordering.
    Override on_request_complete() to track per-trajectory state if needed.
    """

    @abstractmethod
    def priority_key(self, req: "Request") -> tuple:
        """Return a sort key for the pending queue.  Lower → scheduled sooner."""
        ...

    def on_request_complete(self, req: "Request") -> None:
        """Called by the scheduler each time a request finishes.  Optional."""
        pass


# ── Baseline policies ─────────────────────────────────────────────────────────

class FifoPolicy(SchedulerPolicy):
    """Strict arrival-order FIFO.  Reproduces baseline_mode=True behaviour."""

    def priority_key(self, req: "Request") -> tuple:
        return (req.arrival_time,)


class PriorityPolicy(SchedulerPolicy):
    """Per-request difficulty priority (easy first), FIFO within the same tier.

    Reproduces the existing enable_priority=True behaviour as a policy object.
    """

    def priority_key(self, req: "Request") -> tuple:
        # priority: 0=easy (highest), 1=medium, 2=hard (lowest)
        return (req.priority, req.arrival_time)


# ── Trajectory-aware policies ─────────────────────────────────────────────────

class TrajectoryProgressPolicy(SchedulerPolicy):
    """Prioritise requests from trajectories that are ≥50 % complete by step.

    Rationale: a trajectory past its midpoint is closer to yielding a
    usable result.  Finishing it sooner reduces TCT and frees its slot.

    Within the same progress group, tie-break by:
      1. Predicted output length (shortest first — minimises time-to-release)
      2. Arrival time (FIFO among ties)
    """

    def priority_key(self, req: "Request") -> tuple:
        total = max(req.total_steps - 1, 1)   # denominator: steps 0..N-1
        progress = req.step_index / total
        group = 0 if progress >= 0.5 else 1   # 0 = high-progress (schedule first)
        return (group, req.estimated_output_tokens, req.arrival_time)


class TrajectoryDeadlinePolicy(SchedulerPolicy):
    """Schedule by urgency = remaining_work / time_remaining.

    Higher urgency → scheduled sooner.

    Falls back to TrajectoryProgressPolicy ordering when there is no deadline
    pressure — specifically when time_remaining > 2 × estimated_remaining_time.
    This avoids penalising requests that are comfortably ahead of schedule.

    Args:
        estimated_tps: rough tokens-per-second throughput estimate used to
                       convert token budgets to time estimates.  Only affects
                       the deadline-pressure threshold, not actual scheduling.
    """

    _PROGRESS = TrajectoryProgressPolicy()

    def __init__(self, estimated_tps: float = 50.0):
        self._tps = max(estimated_tps, 1.0)

    def priority_key(self, req: "Request") -> tuple:
        now = monotonic()
        time_remaining = req.deadline - now

        # Remaining token budget: this step + all future steps (rough estimate)
        steps_left = req.total_steps - req.step_index   # includes current step
        remaining_tokens = req.estimated_output_tokens * steps_left

        if time_remaining <= 0:
            # Past deadline — treat as maximally urgent
            return (0, float("-inf"), req.arrival_time)

        urgency = remaining_tokens / time_remaining
        estimated_remaining_s = remaining_tokens / self._tps
        under_pressure = time_remaining < 2.0 * estimated_remaining_s

        if under_pressure:
            # Group 0: sort by urgency descending (negate so lower key = higher urgency)
            return (0, -urgency, req.arrival_time)
        else:
            # Group 1: fall back to progress ordering (all group-0 items sort first)
            pg = self._PROGRESS.priority_key(req)
            return (1, *pg)
