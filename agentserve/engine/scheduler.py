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
    Implementation: three separate FIFO deques, one per priority level.
    add() is O(1); dequeue is O(1) (check easy → medium → hard).

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

import bisect
import statistics
import time
from collections import deque
from typing import List, Optional, TYPE_CHECKING

from agentserve.engine.request import Request, RequestStatus

if TYPE_CHECKING:
    from agentserve.engine.policies import SchedulerPolicy


class Scheduler:

    def __init__(
        self,
        max_batch_size: int = 8,
        max_prefill_per_step: int = 4,
        overflow_factor: float = 1.25,
        preempt_after_tokens: int = 10,
        baseline_mode: bool = False,
        enable_priority: bool = True,
        enable_overflow: bool = True,
        enable_preemption: bool = True,
        policy: "Optional[SchedulerPolicy]" = None,
        use_relative_batching: bool = False,
        use_combined_batching: bool = False,
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
            enable_priority:       Enable Policy 1 (priority ordering).
            enable_overflow:       Enable Policy 2 (soft overflow for easy requests).
            enable_preemption:     Enable Policy 3 (preempt young hard requests).
        """
        self.max_batch_size = max_batch_size
        self.max_prefill_per_step = max_prefill_per_step
        self.overflow_factor = overflow_factor
        self.preempt_after_tokens = preempt_after_tokens
        self.baseline_mode = baseline_mode
        # Granular policy toggles for ablation studies
        self.enable_priority   = enable_priority   and not baseline_mode
        self.enable_overflow   = enable_overflow   and not baseline_mode
        self.enable_preemption = enable_preemption and not baseline_mode

        self.soft_cap = int(max_batch_size * overflow_factor)
        self.use_relative_batching = use_relative_batching
        self.use_combined_batching = use_combined_batching

        # Flat list for relative-batching mode (sliding window needs random access)
        self._pending_flat: List[Request] = []

        # Pluggable policy (when set, bypasses the three-deque logic entirely)
        self.policy = policy
        # Sorted list of (key_tuple, sequence_number, req) — key from policy.priority_key
        self._policy_pending: list = []
        self._policy_seq: int = 0

        # Three O(1) FIFO queues for agent-aware mode, one per priority level
        self._pending_easy:   deque[Request] = deque()
        self._pending_medium: deque[Request] = deque()
        self._pending_hard:   deque[Request] = deque()
        self._priority_queues = {
            0: self._pending_easy,
            1: self._pending_medium,
            2: self._pending_hard,
        }

        # Single FIFO for baseline mode (strict arrival order across all priorities)
        self._pending_baseline: deque[Request] = deque()

        self.decoding: List[Request] = []
        self.completed: List[Request] = []

    # ------------------------------------------------------------------
    # Backward-compatible pending property (used by tests and introspection)
    # ------------------------------------------------------------------

    @property
    def pending(self) -> deque:
        """Combined view of all pending requests in priority order."""
        if self.use_relative_batching:
            return deque(self._pending_flat)
        if self.policy is not None:
            return deque(item[2] for item in self._policy_pending)
        if self.baseline_mode or not self.enable_priority:
            return deque(self._pending_baseline)
        combined = deque()
        for q in (self._pending_easy, self._pending_medium, self._pending_hard):
            combined.extend(q)
        return combined

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add(self, request: Request) -> None:
        """Add a new request to the pending queue."""
        request.status = RequestStatus.PENDING
        if self.use_relative_batching:
            self._pending_flat.append(request)
            return
        if self.policy is not None:
            key = self.policy.priority_key(request)
            bisect.insort(self._policy_pending, (key, self._policy_seq, request))
            self._policy_seq += 1
            return
        if self.baseline_mode or not self.enable_priority:
            self._pending_baseline.append(request)
        else:
            self._priority_queues[request.priority].append(request)

    def is_finished(self) -> bool:
        if self.use_relative_batching:
            return not self._pending_flat and not self.decoding
        if self.policy is not None:
            return not self._policy_pending and not self.decoding
        if self.baseline_mode or not self.enable_priority:
            return not self._pending_baseline and not self.decoding
        return (
            not self._pending_easy
            and not self._pending_medium
            and not self._pending_hard
            and not self.decoding
        )

    def get_prefill_batch(self) -> List[Request]:
        """Return up to max_prefill_per_step requests to prefill this step."""
        if self.use_relative_batching:
            return self._get_prefill_batch_relative()
        if self.use_combined_batching:
            return self._get_prefill_batch_combined()
        if self._next_pending_candidate() is None:
            return []

        # Policy 3: check whether we should preempt a young hard request
        if self.enable_preemption:
            self._maybe_preempt()

        batch: List[Request] = []
        while len(batch) < self.max_prefill_per_step:
            candidate = self._next_pending_candidate()
            if candidate is None:
                break

            # Policy 2: respect batch capacity (with soft overflow for easy)
            decode_count = len(self.decoding)
            is_easy = (candidate.priority == 0)
            cap = self.soft_cap if (self.enable_overflow and is_easy) else self.max_batch_size
            if decode_count + len(batch) >= cap:
                break  # at capacity (or over soft cap for easy)

            req = self._pop_front_pending()
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
            if self.policy is not None:
                self.policy.on_request_complete(request)

        return done

    def pop_completed(self) -> List[Request]:
        """Drain and return all completed requests from this epoch."""
        done = list(self.completed)
        self.completed.clear()
        return done

    # ------------------------------------------------------------------
    # Internal queue helpers (O(1))
    # ------------------------------------------------------------------

    def _next_pending_candidate(self) -> Optional[Request]:
        """Peek at the highest-priority pending request without removing it."""
        if self.use_relative_batching:
            return self._pending_flat[0] if self._pending_flat else None
        if self.policy is not None:
            return self._policy_pending[0][2] if self._policy_pending else None
        if self.baseline_mode or not self.enable_priority:
            return self._pending_baseline[0] if self._pending_baseline else None
        for q in (self._pending_easy, self._pending_medium, self._pending_hard):
            if q:
                return q[0]
        return None

    def _pop_front_pending(self) -> Request:
        """Remove and return the highest-priority pending request."""
        if self.policy is not None:
            return self._policy_pending.pop(0)[2]
        if self.baseline_mode or not self.enable_priority:
            return self._pending_baseline.popleft()
        for q in (self._pending_easy, self._pending_medium, self._pending_hard):
            if q:
                return q.popleft()
        raise RuntimeError("pop_front_pending called on empty pending queue")

    # ------------------------------------------------------------------
    # Relative-batching helpers
    # ------------------------------------------------------------------

    def _get_prefill_batch_relative(self) -> List[Request]:
        """
        Sliding-window batch selection: pick up to max_prefill_per_step requests
        whose predicted output lengths form the tightest cluster in the pending queue.

        Why this helps: our decode forward pass pads all KV caches to the longest
        sequence in the batch. Pairing a 5-token request with a 500-token request
        wastes 99% of that slot's compute. By grouping similar-length requests we
        reduce max(seq_lens) within each decode step.

        The age penalty prevents starvation: requests that have waited longest
        push their window's score down even if the variance is slightly higher.
        """
        if not self._pending_flat:
            return []

        slots = max(0, self.max_batch_size - len(self.decoding))
        if slots == 0:
            return []

        n = min(self.max_prefill_per_step, slots, len(self._pending_flat))
        if n == 0:
            return []

        # Sort by predicted output length (continuous ŷ from the predictor)
        sorted_pending = sorted(self._pending_flat,
                                key=lambda r: r.estimated_output_tokens)

        best_batch: List[Request] = sorted_pending[:n]
        best_score = float("inf")
        now = time.monotonic()

        for i in range(len(sorted_pending) - n + 1):
            window = sorted_pending[i: i + n]
            lengths = [r.estimated_output_tokens for r in window]
            variance = statistics.variance(lengths) if len(lengths) > 1 else 0.0
            # Age penalty: reward windows that contain older (more urgent) requests
            age_penalty = sum(now - r.arrival_time for r in window)
            score = variance - 0.15 * age_penalty
            if score < best_score:
                best_score = score
                best_batch = window

        # Remove selected requests from the flat list and mark prefill start
        selected_ids = {id(r) for r in best_batch}
        self._pending_flat = [r for r in self._pending_flat
                              if id(r) not in selected_ids]
        for req in best_batch:
            req.mark_prefill_start()

        return best_batch

    def _get_prefill_batch_combined(self) -> List[Request]:
        """
        Combined mode: priority ordering (easy → medium → hard) with
        relative batching applied *within* each tier.

        Easy requests still always go before medium, medium before hard —
        so agent DAG dependencies get unblocked first. But within the easy
        tier, instead of strict FIFO we pick the requests whose predicted
        output lengths cluster most tightly, minimising KV-padding waste
        during decode without sacrificing the priority bias.
        """
        slots = max(0, self.max_batch_size - len(self.decoding))
        if slots == 0:
            return []

        n = min(self.max_prefill_per_step, slots)
        if self.enable_preemption:
            self._maybe_preempt()

        batch: List[Request] = []
        for q in (self._pending_easy, self._pending_medium, self._pending_hard):
            if len(batch) >= n or not q:
                continue
            remaining = n - len(batch)
            selected = self._select_by_similarity(list(q), remaining)
            for req in selected:
                q.remove(req)
                req.mark_prefill_start()
                batch.append(req)
        return batch

    def _select_by_similarity(self, candidates: List[Request], n: int) -> List[Request]:
        """
        From candidates, return n requests with minimum output-length variance.
        Falls back to FIFO order when n >= len(candidates).
        """
        if len(candidates) <= n:
            return candidates

        sorted_c = sorted(candidates, key=lambda r: r.estimated_output_tokens)
        best, best_score = sorted_c[:n], float("inf")
        now = time.monotonic()

        for i in range(len(sorted_c) - n + 1):
            window = sorted_c[i: i + n]
            lengths = [r.estimated_output_tokens for r in window]
            variance = statistics.variance(lengths) if len(lengths) > 1 else 0.0
            age_penalty = sum(now - r.arrival_time for r in window)
            score = variance - 0.15 * age_penalty
            if score < best_score:
                best_score, best = score, window

        return best

    def _preempt_relative(self) -> None:
        """
        Continuous preemption: among young decoding requests, preempt the one
        most dissimilar from the current batch's median predicted output length.
        More principled than the keyword-based 'youngest hard request' heuristic.
        """
        if not self._pending_flat or not self.decoding:
            return
        if len(self.decoding) < self.max_batch_size:
            return

        median_len = statistics.median(r.estimated_output_tokens for r in self.decoding)
        candidates = [r for r in self.decoding
                      if r.num_output_tokens < self.preempt_after_tokens]
        if not candidates:
            return

        outlier = max(candidates, key=lambda r: abs(r.estimated_output_tokens - median_len))
        self._preempt(outlier)
        # Re-insert into flat list so relative batching can pick it up later
        self._pending_flat.append(outlier)

    # ------------------------------------------------------------------
    # Policy helpers
    # ------------------------------------------------------------------

    def _maybe_preempt(self) -> None:
        """Policy 3: if the front of pending is easy and all decode slots are
        occupied by hard requests, preempt the youngest hard request."""
        if not self._pending_easy:
            return  # no easy requests waiting
        if len(self.decoding) < self.max_batch_size:
            return  # room available, no preemption needed

        # Find the youngest hard request in decoding (fewest output tokens)
        hard_candidates = [
            r for r in self.decoding
            if r.priority == 2 and r.num_output_tokens < self.preempt_after_tokens
        ]
        if not hard_candidates:
            return  # no preemptable hard requests

        youngest = min(hard_candidates, key=lambda r: r.num_output_tokens)
        self._preempt(youngest)

    def _preempt(self, request: Request) -> None:
        """Return a decode-phase request to the front of its priority bucket."""
        self.decoding.remove(request)
        request.status = RequestStatus.PENDING
        # Discard generated tokens so far and re-prefill from scratch
        request.output_token_ids.clear()
        request.kv_cache = None
        request.num_cached_tokens = 0
        request.first_token_time = 0.0
        # Re-insert at the front of its priority queue so it runs soon
        self._priority_queues[request.priority].appendleft(request)

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
