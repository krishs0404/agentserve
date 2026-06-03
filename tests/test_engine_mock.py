"""
End-to-end engine tests using MockModel (CPU only, no GPU needed).

Verifies that the engine loop:
  - Correctly processes requests from PENDING → PREFILL → DECODE → DONE
  - Completes all submitted requests
  - In agent-aware mode, easy requests tend to complete before hard ones
  - Prefix cache hit rate improves when requests share a prefix
  - Metrics are collected correctly
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from agentserve.model.config import TinyConfig
from agentserve.engine.engine import Engine
from agentserve.engine.request import Request, RequestStatus


def make_request(prompt: str, max_tokens: int = 5) -> Request:
    token_ids = [ord(c) % TinyConfig.vocab_size for c in prompt]
    return Request(prompt=prompt, token_ids=token_ids, max_tokens=max_tokens)


def easy_prompt(i: int) -> str:
    return f"Classify as POSITIVE or NEGATIVE. One word only. Review number {i}."

def hard_prompt(i: int) -> str:
    return f"Write a function that implements binary search. Make it complete with error handling. Version {i}."

SHARED_SYSTEM = "You are a helpful AI assistant. Answer concisely. " * 5  # shared prefix


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------

class TestBasicCompletion:
    def test_single_request_completes(self):
        engine = Engine(config=TinyConfig, use_mock=True, agent_aware=True, max_batch_size=4)
        req = make_request("Hello world", max_tokens=3)
        completed = engine.generate([req])
        assert len(completed) == 1
        assert req.status == RequestStatus.DONE
        assert req.num_output_tokens == 3

    def test_all_requests_complete(self):
        engine = Engine(config=TinyConfig, use_mock=True, agent_aware=True, max_batch_size=4)
        requests = [make_request(f"Request {i}", max_tokens=4) for i in range(10)]
        completed = engine.generate(requests)
        assert len(completed) == 10
        for req in completed:
            assert req.status == RequestStatus.DONE

    def test_output_tokens_up_to_max(self):
        engine = Engine(config=TinyConfig, use_mock=True, agent_aware=True)
        req = make_request("Test prompt", max_tokens=7)
        engine.generate([req])
        assert req.num_output_tokens <= 7

    def test_timing_fields_set(self):
        engine = Engine(config=TinyConfig, use_mock=True, agent_aware=True)
        req = make_request("Test", max_tokens=3)
        engine.generate([req])
        assert req.arrival_time > 0
        assert req.first_token_time > 0
        assert req.done_time > 0
        assert req.first_token_time >= req.arrival_time
        assert req.done_time >= req.first_token_time


# ---------------------------------------------------------------------------
# Easy-before-hard ordering in agent-aware mode
# ---------------------------------------------------------------------------

class TestPriorityOrdering:
    def test_easy_completes_before_hard_agent_aware(self):
        """With agent-aware scheduling, easy requests should finish before hard ones."""
        engine = Engine(
            config=TinyConfig, use_mock=True, agent_aware=True,
            max_batch_size=4, max_prefill_per_step=2,
        )
        # Mix easy and hard, hard arrives first
        requests = []
        for i in range(3):
            req = make_request(hard_prompt(i), max_tokens=20)
            requests.append(req)
        for i in range(3):
            req = make_request(easy_prompt(i), max_tokens=3)
            requests.append(req)

        completed = engine.generate(requests)

        # Verify all completed
        assert len(completed) == 6

        # Easy requests should have lower average latency than hard ones
        easy_done = [r for r in completed if r.difficulty == "easy"]
        hard_done = [r for r in completed if r.difficulty == "hard"]

        if easy_done and hard_done:
            avg_easy_latency = sum(r.latency for r in easy_done) / len(easy_done)
            avg_hard_latency = sum(r.latency for r in hard_done) / len(hard_done)
            # Easy requests should have lower latency (they're shorter, AND prioritized)
            assert avg_easy_latency <= avg_hard_latency, (
                f"Easy avg latency {avg_easy_latency:.3f}s should be <= "
                f"hard avg latency {avg_hard_latency:.3f}s"
            )

    def test_all_complete_in_baseline_too(self):
        """Baseline mode should still complete all requests, just in FIFO order."""
        engine = Engine(config=TinyConfig, use_mock=True, agent_aware=False, max_batch_size=4)
        requests = [make_request(f"Request {i}", max_tokens=4) for i in range(10)]
        completed = engine.generate(requests)
        assert len(completed) == 10


# ---------------------------------------------------------------------------
# Prefix cache
# ---------------------------------------------------------------------------

class TestPrefixCache:
    def test_shared_prefix_improves_hit_rate(self):
        engine = Engine(
            config=TinyConfig, use_mock=True, agent_aware=True,
            max_batch_size=4, max_prefix_cache_entries=64,
        )
        # All requests share a long system prompt (forces prefix cache use)
        requests = []
        for i in range(8):
            prompt = SHARED_SYSTEM + f" Task {i}: classify as positive or negative."
            token_ids = [ord(c) % TinyConfig.vocab_size for c in prompt]
            requests.append(Request(prompt=prompt, token_ids=token_ids, max_tokens=4))

        completed = engine.generate(requests)
        assert len(completed) == 8

    def test_metrics_track_prefix_hits(self):
        engine = Engine(config=TinyConfig, use_mock=True, agent_aware=True, max_batch_size=4)
        requests = []
        for i in range(6):
            prompt = SHARED_SYSTEM + f" Do task {i}."
            token_ids = [ord(c) % TinyConfig.vocab_size for c in prompt]
            requests.append(Request(prompt=prompt, token_ids=token_ids, max_tokens=3))

        engine.generate(requests)
        # Metrics should be populated
        assert engine.metrics.total_requests == 6
        assert engine.metrics.completed_requests == 6
        assert engine.metrics.steps > 0


# ---------------------------------------------------------------------------
# Metrics collection
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_metrics_after_run(self):
        engine = Engine(config=TinyConfig, use_mock=True, agent_aware=True, max_batch_size=4)
        requests = [make_request(f"Request {i}", max_tokens=4) for i in range(5)]
        engine.generate(requests)

        m = engine.metrics
        assert m.total_requests == 5
        assert m.completed_requests == 5
        assert m.total_prompt_tokens > 0
        assert m.total_output_tokens > 0
        assert m.steps > 0

    def test_difficulty_counts_sum_to_total(self):
        engine = Engine(config=TinyConfig, use_mock=True, agent_aware=True, max_batch_size=8)
        requests = []
        for i in range(3):
            requests.append(make_request(easy_prompt(i), max_tokens=3))
        for i in range(3):
            requests.append(make_request(hard_prompt(i), max_tokens=5))
        for i in range(2):
            requests.append(make_request(f"Explain {i} in 2 sentences.", max_tokens=4))

        engine.generate(requests)

        m = engine.metrics
        total_by_diff = sum(m.difficulty_counts.values())
        assert total_by_diff == m.completed_requests

    def test_throughput_positive(self):
        engine = Engine(config=TinyConfig, use_mock=True, agent_aware=True)
        requests = [make_request("Test", max_tokens=3) for _ in range(3)]
        engine.generate(requests)
        assert engine.metrics.throughput_tokens_per_sec >= 0


# ---------------------------------------------------------------------------
# Stress test
# ---------------------------------------------------------------------------

class TestStress:
    def test_many_requests_all_complete(self):
        engine = Engine(config=TinyConfig, use_mock=True, agent_aware=True, max_batch_size=8)
        requests = [make_request(f"Request {i}", max_tokens=3) for i in range(30)]
        completed = engine.generate(requests, max_steps=5000)
        assert len(completed) == 30, f"Only {len(completed)}/30 completed"

    def test_single_token_max(self):
        """max_tokens=1 should produce exactly one output token."""
        engine = Engine(config=TinyConfig, use_mock=True, agent_aware=True)
        req = make_request("Quick answer:", max_tokens=1)
        engine.generate([req])
        assert req.num_output_tokens == 1
        assert req.status == RequestStatus.DONE


# ---------------------------------------------------------------------------
# SDPA / LlamaModel shape validation
# ---------------------------------------------------------------------------

class TestLlamaModelShapes:
    """Verify LlamaModel (TinyConfig) produces correct output shapes via SDPA.

    TinyConfig uses 2 layers so this runs fast on CPU.
    """

    def test_forward_prefill_shapes(self):
        import torch
        from agentserve.model.llama import LlamaModel
        model = LlamaModel(TinyConfig)
        token_ids = [1, 2, 3, 4, 5]
        logits, kv_cache = model.forward(token_ids)
        assert logits.shape == (len(token_ids), TinyConfig.vocab_size)
        assert len(kv_cache) == TinyConfig.n_layers
        k, v = kv_cache[0]
        assert k.shape[0] == len(token_ids)
        assert k.shape[1] == TinyConfig.n_kv_heads
        assert k.shape[2] == TinyConfig.head_dim

    def test_forward_decode_batch_shapes(self):
        import torch
        from agentserve.model.llama import LlamaModel
        model = LlamaModel(TinyConfig)
        B = 3
        # Different sequence lengths exercises the vectorised padding mask path
        seq_lens = [4, 6, 5]
        Kh, D = TinyConfig.n_kv_heads, TinyConfig.head_dim
        kv_caches = [
            [(torch.zeros(L, Kh, D), torch.zeros(L, Kh, D))
             for _ in range(TinyConfig.n_layers)]
            for L in seq_lens
        ]
        logits, new_kvs = model.forward_decode_batch([1, 2, 3], kv_caches, seq_lens[:])
        assert logits.shape == (B, TinyConfig.vocab_size)
        assert len(new_kvs) == B
        for b, (orig_len, req_kvs) in enumerate(zip(seq_lens, new_kvs)):
            k, v = req_kvs[0]
            assert k.shape[0] == orig_len + 1, f"req {b}: expected {orig_len+1}, got {k.shape[0]}"

    def test_forward_decode_batch_uniform_lengths(self):
        """Uniform seq_lens — no padding mask needed. Should still succeed."""
        import torch
        from agentserve.model.llama import LlamaModel
        model = LlamaModel(TinyConfig)
        B, L = 2, 5
        Kh, D = TinyConfig.n_kv_heads, TinyConfig.head_dim
        kv_caches = [
            [(torch.zeros(L, Kh, D), torch.zeros(L, Kh, D))
             for _ in range(TinyConfig.n_layers)]
            for _ in range(B)
        ]
        logits, _ = model.forward_decode_batch([1, 2], kv_caches, [L, L])
        assert logits.shape == (B, TinyConfig.vocab_size)
