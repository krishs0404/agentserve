"""
Continuous-batching scheduler with agent-aware policies.

The key insight driving this design: agents emit heterogeneous request
streams.  A single agent task may fire 5 classification calls, 2 extraction
calls, and 1 code-generation call simultaneously.  A generic FIFO scheduler
runs them in arrival order, so the code-gen occupies a batch slot for
hundreds of steps while the 5 trivial classifiers queue behind it.

Each trivial classifier is a *dependency* in the agent's DAG: the agent
can't proceed until it has all 5 answers.  Making the code-gen wait 20
steps is fine; making the classifiers wait 200 steps (blocked by code-gen)
is catastrophic for overall task completion time.

Three agent-aware policies (all disabled in baseline_mode=True):

  Policy 1 — Priority scheduling
    Requests are sorted by difficulty (easy=0 < medium=1 < hard=2).
    Easy requests move to the front of the pending queue regardless of
    arrival order.  Within the same difficulty level, FIFO is preserved.

  Policy 2 — Soft batch overflow for easy requests
    When the decode batch is at max_batch_size and a new easy request
    arrives, we admit it anyway — up to max_batch_size * overflow_factor.
    The marginal GPU cost of one extra easy request is tiny because it
    will finish generating in ~20 tokens.

  Policy 3 — Preemption of young hard requests
    If ALL decode slots are occupied by hard requests AND an easy request
    is waiting, we preempt the YOUNGEST hard request (fewest output tokens
    generated so far) to free a slot for the easy one.  We only preempt
    if the hard request has generated < preempt_after_tokens tokens — it
    hasn't invested much compute yet, so the re-prefill cost is low.
    The preempted request goes back to pending at the front of its priority
    bucket so it's rescheduled soon.
"""

from __future__ import annotations
from collections import deque
from typing import List, Tuple

from agentserve.engine.request import Request, RequestStatus


class Scheduler:

    def __init__(
        self,
        max_batch_size: int = 8,
        max_prefill_per_step: int = 4,
        overflow_factor: float = 1.25,
        preempt_after_tokens: int = 10,
        baseline_mode: bool = False,
    ):
        """
        Args:
            max_batch_size:        Maximum decode-phase requests per step.
            max_prefill_per_step:  Max new requests to admit to prefill each step.
            overflow_factor:       Soft overflow limit = max_batch_size * factor.
            preempt_after_tokens:  Only preempt hard requests with < this many
                                   output tokens generated.
            baseline_mode:         If True, disable all three policies and use
                                   plain FIFO scheduling (for benchmarking).
        """
        self.max_batch_size = max_batch_size
        self.max_prefill_per_step = max_prefill_per_step
        self.overflow_factor = overflow_factor
        self.preempt_after_tokens = preempt_after_tokens
        self.baseline_mode = baseline_mode

        self.soft_cap = int(max_batch_size * overflow_factor)

        # Queues
        self.pending: deque[Request] = deque()   # waiting to be prefilled
        self.decoding: List[Request] = []         # generating tokens
        self.completed: List[Request] = []        # done this epoch

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add(self, request: Request) -> None:
        """Add a new request to the pending queue."""
        request.status = RequestStatus.PENDING
        if self.baseline_mode:
            self.pending.append(request)
        else:
            self._insert_priority(request)

    def is_finished(self) -> bool:
        return not self.pending and not self.decoding

    def get_prefill_batch(self) -> List[Request]:
        """Return up to max_prefill_per_step requests to prefill this step.

        Also applies Policy 2 (overflow) and Policy 3 (preemption).
        """
        if not self.pending:
            return []

        # Policy 3: check whether we should preempt a young hard request
        if not self.baseline_mode:
            self._maybe_preempt()

        batch: List[Request] = []
        while self.pending and len(batch) < self.max_prefill_per_step:
            # Policy 2: respect batch capacity (with soft overflow for easy)
            decode_count = len(self.decoding)
            candidate = self.pending[0]
            is_easy = (candidate.priority == 0)

            cap = self.soft_cap if (not self.baseline_mode and is_easy) else self.max_batch_size
            if decode_count + len(batch) >= cap:
                break  # at capacity (or over soft cap for easy)

            req = self.pending.popleft()
            req.mark_prefill_start()
            batch.append(req)

        return batch

    def get_decode_batch(self) -> List[Request]:
        """Return all requests currently in decode phase."""
        return list(self.decoding)

    def on_prefill_complete(self, request: Request) -> None:
        """Called by engine after a request's prefill step finishes."""
        request.mark_first_token()
        self.decoding.append(request)

    def on_decode_step(self, request: Request, token_id: int, eos_token_id: int = 1) -> bool:
        """
        Process one decode step for a request.

        Returns True if the request is now complete.
        """
        request.output_token_ids.append(token_id)

        done = (
            len(request.output_token_ids) >= request.max_tokens
            or token_id == eos_token_id
        )
        if done:
            request.mark_done()
            self.decoding.remove(request)
            self.completed.append(request)

        return done

    def pop_completed(self) -> List[Request]:
        """Drain and return all completed requests from this epoch."""
        done = list(self.completed)
        self.completed.clear()
        return done

    # ------------------------------------------------------------------
    # Policy helpers
    # ------------------------------------------------------------------

    def _insert_priority(self, request: Request) -> None:
        """Insert a request into pending, maintaining priority order.

        Priority 0 (easy) goes before priority 1 (medium) before priority 2 (hard).
        Within same priority, FIFO is preserved (insert at the last position with
        the same or higher priority).
        """
        p = request.priority
        # Find insertion point: after all existing entries with priority <= p
        # so that within same priority we keep arrival order.
        insert_at = len(self.pending)
        for i in range(len(self.pending) - 1, -1, -1):
            if self.pending[i].priority <= p:
                insert_at = i + 1
                break
            else:
                insert_at = i

        lst = list(self.pending)
        lst.insert(insert_at, request)
        self.pending = deque(lst)

    def _maybe_preempt(self) -> None:
        """Policy 3: if the front of pending is easy and all decode slots are
        occupied by hard requests, preempt the youngest hard request."""
        if not self.pending:
            return
        if self.pending[0].priority != 0:  # front is not easy
            return
        if len(self.decoding) < self.max_batch_size:
            return  # room available, no preemption needed

        # Find the youngest hard request in decoding (fewest output tokens)
        hard_candidates = [
            r for r in self.decoding
            if r.priority == 2 and r.num_output_tokens < self.preempt_after_tokens
        ]
        if not hard_candidates:
            return  # no preemptable hard requests

        # Youngest = fewest output tokens
        youngest = min(hard_candidates, key=lambda r: r.num_output_tokens)
        self._preempt(youngest)

    def _preempt(self, request: Request) -> None:
        """Return a decode-phase request to the front of pending."""
        self.decoding.remove(request)
        request.status = RequestStatus.PENDING
        # Discard generated tokens so far and re-prefill from scratch
        request.output_token_ids.clear()
        request.kv_cache = None
        request.num_cached_tokens = 0
        request.first_token_time = 0.0
        # Insert at the front of its priority bucket (re-runs soon)
        self._insert_priority(request)

    # ------------------------------------------------------------------
    # Introspection for tests and benchmarks
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "pending": len(self.pending),
            "decoding": len(self.decoding),
            "completed_this_epoch": len(self.completed),
            "baseline_mode": self.baseline_mode,
        }
