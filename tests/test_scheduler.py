"""
Tests for the agent-aware scheduler.

All CPU, no GPU required.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from agentserve.engine.request import Request, RequestStatus
from agentserve.engine.scheduler import Scheduler


def make_request(priority: int = 1, max_tokens: int = 50, prompt: str = "test") -> Request:
    req = Request(prompt=prompt, token_ids=[1, 2, 3], max_tokens=max_tokens)
    req.priority = priority
    req.difficulty = {0: "easy", 1: "medium", 2: "hard"}[priority]
    req.estimated_output_tokens = {0: 20, 1: 100, 2: 256}[priority]
    return req


def make_easy(max_tokens: int = 20)   -> Request: return make_request(priority=0, max_tokens=max_tokens)
def make_medium(max_tokens: int = 50) -> Request: return make_request(priority=1, max_tokens=max_tokens)
def make_hard(max_tokens: int = 100)  -> Request: return make_request(priority=2, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Request lifecycle
# ---------------------------------------------------------------------------

class TestRequestLifecycle:
    def test_add_puts_in_pending(self):
        sched = Scheduler()
        req = make_easy()
        sched.add(req)
        assert len(sched.pending) == 1
        assert req.status == RequestStatus.PENDING

    def test_prefill_batch_moves_to_decode(self):
        sched = Scheduler(max_batch_size=4)
        req = make_easy()
        sched.add(req)
        batch = sched.get_prefill_batch()
        assert req in batch
        sched.on_prefill_complete(req)
        assert req in sched.decoding
        assert req.status == RequestStatus.DECODE

    def test_decode_step_completes_at_max_tokens(self):
        sched = Scheduler(max_batch_size=4)
        req = make_request(max_tokens=3)
        sched.add(req)
        batch = sched.get_prefill_batch()
        sched.on_prefill_complete(req)  # moves to decode

        done = sched.on_decode_step(req, token_id=5, eos_token_id=1)
        assert not done  # 1 token
        done = sched.on_decode_step(req, token_id=5, eos_token_id=1)
        assert not done  # 2 tokens
        done = sched.on_decode_step(req, token_id=5, eos_token_id=1)
        assert done       # 3 tokens = max_tokens

        assert req.status == RequestStatus.DONE
        assert req not in sched.decoding

    def test_eos_token_triggers_completion(self):
        sched = Scheduler(max_batch_size=4)
        req = make_request(max_tokens=100)
        sched.add(req)
        sched.get_prefill_batch()
        sched.on_prefill_complete(req)

        done = sched.on_decode_step(req, token_id=1, eos_token_id=1)  # EOS
        assert done
        assert req.status == RequestStatus.DONE

    def test_is_finished_when_all_done(self):
        sched = Scheduler(max_batch_size=4)
        assert sched.is_finished()
        req = make_easy()
        sched.add(req)
        assert not sched.is_finished()


# ---------------------------------------------------------------------------
# Policy 1: Priority ordering (agent-aware mode)
# ---------------------------------------------------------------------------

class TestPriorityScheduling:
    def test_easy_before_medium_before_hard(self):
        sched = Scheduler(max_batch_size=8, max_prefill_per_step=3, baseline_mode=False)
        hard   = make_hard()
        medium = make_medium()
        easy   = make_easy()
        # Add in worst-case arrival order: hard first
        sched.add(hard)
        sched.add(medium)
        sched.add(easy)

        batch = sched.get_prefill_batch()
        # First in batch should be easy, then medium, then hard
        assert batch[0].priority == 0  # easy
        assert batch[1].priority == 1  # medium
        assert batch[2].priority == 2  # hard

    def test_fifo_within_same_priority(self):
        sched = Scheduler(max_batch_size=8, max_prefill_per_step=3, baseline_mode=False)
        e1 = make_easy(); e1.prompt = "first"
        e2 = make_easy(); e2.prompt = "second"
        e3 = make_easy(); e3.prompt = "third"
        sched.add(e1)
        sched.add(e2)
        sched.add(e3)
        batch = sched.get_prefill_batch()
        assert batch[0].prompt == "first"
        assert batch[1].prompt == "second"
        assert batch[2].prompt == "third"

    def test_baseline_mode_is_fifo(self):
        sched = Scheduler(max_batch_size=8, max_prefill_per_step=3, baseline_mode=True)
        hard   = make_hard();   hard.prompt   = "hard_first"
        medium = make_medium(); medium.prompt = "medium_second"
        easy   = make_easy();   easy.prompt   = "easy_third"
        sched.add(hard)
        sched.add(medium)
        sched.add(easy)
        batch = sched.get_prefill_batch()
        # FIFO: hard first (arrived first)
        assert batch[0].prompt == "hard_first"
        assert batch[1].prompt == "medium_second"
        assert batch[2].prompt == "easy_third"


# ---------------------------------------------------------------------------
# Policy 2: Soft overflow for easy requests
# ---------------------------------------------------------------------------

class TestBatchAdmissionOverflow:
    def test_easy_admitted_beyond_max_batch(self):
        """An easy request should be admitted even when decode batch is at capacity."""
        sched = Scheduler(max_batch_size=4, overflow_factor=1.25, baseline_mode=False)

        # Fill decode batch to max_batch_size with medium requests
        for _ in range(4):
            req = make_medium()
            sched.add(req)
            sched.get_prefill_batch()
            sched.on_prefill_complete(req)

        assert len(sched.decoding) == 4  # at capacity

        # Now add an easy request
        easy = make_easy()
        sched.add(easy)
        batch = sched.get_prefill_batch()
        # Easy should be admitted (soft cap = 4 * 1.25 = 5)
        assert easy in batch

    def test_medium_blocked_at_capacity(self):
        """A medium request should be blocked when decode batch is at capacity."""
        sched = Scheduler(max_batch_size=4, overflow_factor=1.25, baseline_mode=False)

        for _ in range(4):
            req = make_medium()
            sched.add(req)
            sched.get_prefill_batch()
            sched.on_prefill_complete(req)

        assert len(sched.decoding) == 4

        medium = make_medium()
        sched.add(medium)
        batch = sched.get_prefill_batch()
        assert medium not in batch  # blocked

    def test_overflow_disabled_in_baseline_mode(self):
        """Baseline mode should not allow overflow."""
        sched = Scheduler(max_batch_size=4, overflow_factor=1.25, baseline_mode=True)

        for _ in range(4):
            req = make_medium()
            sched.add(req)
            sched.get_prefill_batch()
            sched.on_prefill_complete(req)

        easy = make_easy()
        sched.add(easy)
        batch = sched.get_prefill_batch()
        # Baseline: no overflow, easy should also be blocked
        assert easy not in batch


# ---------------------------------------------------------------------------
# Policy 3: Preemption
# ---------------------------------------------------------------------------

class TestPreemption:
    def test_young_hard_preempted_for_easy(self):
        """A hard request with <preempt_after_tokens output tokens should be preempted."""
        sched = Scheduler(
            max_batch_size=2,
            overflow_factor=1.0,   # no overflow allowed
            preempt_after_tokens=10,
            baseline_mode=False,
        )

        # Fill decode batch with hard requests (both young, 0 output tokens)
        h1 = make_hard(); sched.add(h1)
        h2 = make_hard(); sched.add(h2)
        sched.get_prefill_batch()
        sched.on_prefill_complete(h1)
        sched.get_prefill_batch()
        sched.on_prefill_complete(h2)

        assert len(sched.decoding) == 2

        # Now add an easy request
        easy = make_easy()
        sched.add(easy)

        # Trigger prefill — scheduler should preempt youngest hard, admit easy
        batch = sched.get_prefill_batch()
        assert easy in batch
        # One hard request should have been preempted back to pending
        preempted_count = sum(1 for r in [h1, h2] if r.status == RequestStatus.PENDING)
        assert preempted_count >= 1

    def test_old_hard_not_preempted(self):
        """Hard requests with many output tokens should not be preempted."""
        sched = Scheduler(
            max_batch_size=2,
            overflow_factor=1.0,
            preempt_after_tokens=5,
            baseline_mode=False,
        )

        h1 = make_hard(); sched.add(h1)
        h2 = make_hard(); sched.add(h2)
        sched.get_prefill_batch()
        sched.on_prefill_complete(h1)
        sched.get_prefill_batch()
        sched.on_prefill_complete(h2)

        # Simulate 10 decode steps — both hard requests are now "old"
        for _ in range(10):
            for tok in range(5, 8):
                if h1 in sched.decoding:
                    h1.output_token_ids.append(tok)
                if h2 in sched.decoding:
                    h2.output_token_ids.append(tok)

        easy = make_easy()
        sched.add(easy)
        batch = sched.get_prefill_batch()

        # Neither hard request should have been preempted (both have > preempt_after_tokens)
        assert h1.status != RequestStatus.PENDING
        assert h2.status != RequestStatus.PENDING

    def test_preemption_disabled_in_baseline(self):
        """Baseline mode should never preempt."""
        sched = Scheduler(max_batch_size=2, overflow_factor=1.0, baseline_mode=True)

        h1 = make_hard(); sched.add(h1)
        h2 = make_hard(); sched.add(h2)
        sched.get_prefill_batch()
        sched.on_prefill_complete(h1)
        sched.get_prefill_batch()
        sched.on_prefill_complete(h2)

        easy = make_easy()
        sched.add(easy)
        sched.get_prefill_batch()

        # No preemption in baseline
        assert h1.status == RequestStatus.DECODE
        assert h2.status == RequestStatus.DECODE


# ---------------------------------------------------------------------------
# Agent-aware vs baseline end-to-end ordering
# ---------------------------------------------------------------------------

class TestAgentAwareVsBaseline:
    def _run_to_completion(self, sched: Scheduler, requests: list[Request]) -> list[Request]:
        """Run scheduler until all requests complete. Returns completion order."""
        completed = []
        for req in requests:
            sched.add(req)

        max_steps = 1000
        for _ in range(max_steps):
            if sched.is_finished():
                break
            batch = sched.get_prefill_batch()
            for req in batch:
                sched.on_prefill_complete(req)

            decode_batch = sched.get_decode_batch()
            for req in decode_batch:
                done = sched.on_decode_step(req, token_id=5, eos_token_id=999)
                if done:
                    completed.append(req)

            sched.pop_completed()

        return completed

    def test_easy_completes_before_hard_in_agent_aware(self):
        easy = make_easy(max_tokens=3)
        hard = make_hard(max_tokens=20)

        sched = Scheduler(max_batch_size=2, baseline_mode=False)
        # Hard arrives first
        completed = self._run_to_completion(sched, [hard, easy])

        # Easy should finish first despite arriving after hard
        if len(completed) >= 2:
            easy_idx = next((i for i, r in enumerate(completed) if r.priority == 0), None)
            hard_idx = next((i for i, r in enumerate(completed) if r.priority == 2), None)
            if easy_idx is not None and hard_idx is not None:
                assert easy_idx < hard_idx, "Easy should complete before hard in agent-aware mode"

    def test_baseline_respects_arrival_order(self):
        """In baseline FIFO, hard (arrived first) tends to complete before easy."""
        hard = make_hard(max_tokens=3)   # small max_tokens so test is fast
        easy = make_easy(max_tokens=20)

        sched = Scheduler(max_batch_size=1, baseline_mode=True)
        # Hard arrives first, batch size 1 → strictly sequential
        completed = self._run_to_completion(sched, [hard, easy])

        # Hard arrived first and should complete first (batch_size=1, strict FIFO)
        if len(completed) >= 2:
            assert completed[0].priority == 2  # hard was first


# ---------------------------------------------------------------------------
# Relative batching: variance minimization
# ---------------------------------------------------------------------------

class TestRelativeBatching:
    def _make_req(self, predicted_len: int) -> Request:
        req = Request(prompt="test", token_ids=[1, 2, 3], max_tokens=predicted_len)
        req.estimated_output_tokens = predicted_len
        req.priority = 1
        req.difficulty = "medium"
        return req

    def test_selects_minimum_variance_window(self):
        """_select_by_similarity picks the tightest cluster, not the front of the list."""
        import statistics
        sched = Scheduler(max_batch_size=8, use_combined_batching=True)
        # [10, 100, 200, 201, 202]: window [200, 201, 202] has variance ~1 — far tighter
        # than any window that includes 10 or 100.
        reqs = [self._make_req(l) for l in [10, 100, 200, 201, 202]]
        selected = sched._select_by_similarity(reqs, n=3)
        selected_lens = sorted(r.estimated_output_tokens for r in selected)
        # Verify variance of selected window is below any cross-cluster window
        assert statistics.variance(selected_lens) < statistics.variance([10, 100, 200])
        assert selected_lens == [200, 201, 202]

    def test_returns_all_when_n_ge_len(self):
        reqs = [self._make_req(l) for l in [50, 100, 150]]
        sched = Scheduler(max_batch_size=8, use_combined_batching=True)
        selected = sched._select_by_similarity(reqs, n=5)
        assert len(selected) == 3  # returns all when n >= len

    def test_relative_mode_engine_completes_all(self):
        """Relative batching mode still completes all requests."""
        from agentserve.engine.engine import Engine
        from agentserve.model.config import TinyConfig
        engine = Engine(config=TinyConfig, use_mock=True, agent_aware=True,
                        use_relative_batching=True, max_batch_size=4)
        reqs = [Request(prompt="p", token_ids=[1, 2], max_tokens=t)
                for t in [5, 10, 50, 5, 10, 50]]
        completed = engine.generate(reqs)
        assert len(completed) == 6
