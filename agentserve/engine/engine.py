"""
Main engine loop.

Ties together the model, scheduler, block allocator, prefix cache, and
difficulty classifier into a single inference engine.

Engine step loop:
  1. Accept newly submitted requests from the input queue
  2. Classify each new request's difficulty
  3. Query the prefix cache — skip re-computing KV for matched prefix tokens
  4. Ask the scheduler which requests to prefill this step
  5. Prefill selected requests (full prompt forward pass)
  6. Decode all active decode-phase requests by one token each
  7. Check for completions (max_tokens reached or EOS)
  8. Free block-allocator resources for completed requests
  9. Accumulate metrics

The engine can run with a real LlamaModel (GPU) or a MockModel (CPU).
Set use_mock=True for testing without any GPU.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import torch

from agentserve.model.config import ModelConfig, TinyConfig
from agentserve.model.llama import LlamaModel, MockModel
from agentserve.engine.request import Request, RequestStatus
from agentserve.engine.scheduler import Scheduler
from agentserve.engine.cache import BlockAllocator
from agentserve.engine.prefix_cache import PrefixCache
from agentserve.engine.difficulty import RequestDifficultyClassifier
from agentserve.engine.sampling import sample


@dataclass
class EngineMetrics:
    total_requests: int = 0
    completed_requests: int = 0
    total_prompt_tokens: int = 0
    total_output_tokens: int = 0
    prefix_cache_hits: int = 0
    prefix_cache_misses: int = 0
    prefix_tokens_saved: int = 0
    steps: int = 0
    wall_time: float = 0.0
    difficulty_counts: Dict[str, int] = field(default_factory=lambda: {"easy": 0, "medium": 0, "hard": 0})

    @property
    def throughput_tokens_per_sec(self) -> float:
        if self.wall_time <= 0:
            return 0.0
        return (self.total_prompt_tokens + self.total_output_tokens) / self.wall_time

    @property
    def prefix_hit_rate(self) -> float:
        total = self.prefix_cache_hits + self.prefix_cache_misses
        return self.prefix_cache_hits / total if total > 0 else 0.0


class Engine:
    """
    The AgentServe inference engine.

    Usage:
        engine = Engine(config=TinyConfig, use_mock=True, agent_aware=True)
        engine.submit(request)
        engine.run_until_done()
        results = engine.completed_requests
    """

    def __init__(
        self,
        config: ModelConfig = TinyConfig,
        use_mock: bool = True,
        agent_aware: bool = True,
        max_batch_size: int = 8,
        max_prefill_per_step: int = 4,
        num_cache_blocks: int = 256,
        block_size: int = 16,
        eos_token_id: int = 1,
        max_prefix_cache_entries: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ):
        self.config = config
        self.eos_token_id = eos_token_id
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        # Model: real or mock
        if use_mock:
            self.model = MockModel(config)
        else:
            self.model = LlamaModel(config)

        # Scheduler: agent-aware or plain FIFO baseline
        self.scheduler = Scheduler(
            max_batch_size=max_batch_size,
            max_prefill_per_step=max_prefill_per_step,
            baseline_mode=not agent_aware,
        )

        # KV-cache block allocator
        self.allocator = BlockAllocator(
            num_blocks=num_cache_blocks,
            block_size=block_size,
            num_layers=config.n_layers,
            num_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
        )

        # Prefix cache (LFU eviction)
        self.prefix_cache = PrefixCache(
            block_size=block_size,
            max_entries=max_prefix_cache_entries,
        )

        # Difficulty classifier
        self.classifier = RequestDifficultyClassifier()

        # Metrics
        self.metrics = EngineMetrics()
        self.completed_requests: List[Request] = []

        # Internal: newly submitted requests waiting for classification
        self._incoming: deque[Request] = deque()

    # ------------------------------------------------------------------
    # Public submission API
    # ------------------------------------------------------------------

    def submit(self, request: Request) -> None:
        """Submit a request to the engine. Thread-safe in single-threaded mode."""
        self._incoming.append(request)

    def submit_many(self, requests: List[Request]) -> None:
        for r in requests:
            self.submit(r)

    # ------------------------------------------------------------------
    # Engine loop
    # ------------------------------------------------------------------

    def run_until_done(self, max_steps: int = 10_000) -> List[Request]:
        """Run the engine until all submitted requests complete or max_steps hit."""
        start = time.monotonic()
        for _ in range(max_steps):
            if self._is_idle():
                break
            self.step()
        self.metrics.wall_time = time.monotonic() - start
        return self.completed_requests

    def _is_idle(self) -> bool:
        return not self._incoming and self.scheduler.is_finished()

    def step(self) -> List[Request]:
        """
        One engine step.  Returns requests that completed this step.
        """
        self.metrics.steps += 1

        # 1. Classify and enqueue newly arrived requests
        self._ingest_incoming()

        # 2. Prefill: scheduler picks which pending requests to start
        prefill_batch = self.scheduler.get_prefill_batch()
        for req in prefill_batch:
            self._prefill(req)

        # 3. Decode: one token per active decode-phase request
        decode_batch = self.scheduler.get_decode_batch()
        newly_done: List[Request] = []
        for req in decode_batch:
            done = self._decode_step(req)
            if done:
                newly_done.append(req)

        # 4. Free block-allocator resources for completed requests
        for req in newly_done:
            self.allocator.free(req.request_id)

        # 5. Collect completions from scheduler and accumulate metrics
        completed_this_step = self.scheduler.pop_completed()
        for req in completed_this_step:
            self.metrics.completed_requests += 1
            self.metrics.total_output_tokens += req.num_output_tokens
            self.metrics.difficulty_counts[req.difficulty] = (
                self.metrics.difficulty_counts.get(req.difficulty, 0) + 1
            )
            self.completed_requests.append(req)

        return completed_this_step

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ingest_incoming(self) -> None:
        """Classify and schedule all newly arrived requests."""
        while self._incoming:
            req = self._incoming.popleft()

            # Classify difficulty and set priority
            diff = self.classifier.classify(req.prompt)
            req.difficulty = diff.level.value
            req.priority = diff.priority
            req.estimated_output_tokens = diff.estimated_output_tokens

            # Check prefix cache to skip redundant prefill
            matched_len, _ = self.prefix_cache.find_longest_prefix(req.token_ids)
            req.num_cached_tokens = matched_len
            if matched_len > 0:
                self.metrics.prefix_cache_hits += 1
                self.metrics.prefix_tokens_saved += matched_len
            else:
                self.metrics.prefix_cache_misses += 1

            # Allocate KV-cache blocks (logical bookkeeping)
            blocks_needed = self.allocator.blocks_needed(len(req.token_ids) + req.max_tokens)
            if self.allocator.can_allocate(blocks_needed):
                self.allocator.allocate(req.request_id, blocks_needed)
            # (if not enough blocks, still schedule — mock model doesn't need them)

            self.metrics.total_requests += 1
            self.metrics.total_prompt_tokens += req.num_prompt_tokens

            self.scheduler.add(req)

    def _prefill(self, req: Request) -> None:
        """Run the full-prompt forward pass for a request."""
        # Tokens to actually process (skip prefix-cached ones)
        input_ids = req.token_ids[req.num_cached_tokens:]
        if not input_ids:
            # Entire prompt was cached — treat as if prefill succeeded with no work
            input_ids = req.token_ids[-1:]  # feed at least the last token

        logits, kv_cache = self.model.forward(
            token_ids=input_ids,
            kv_cache=req.kv_cache,
            position_offset=req.num_cached_tokens,
        )
        req.kv_cache = kv_cache
        req.num_cached_tokens = len(req.token_ids)

        # Sample the first output token from the last logit position
        first_token = int(self._sample(logits[-1:]).item())

        # Update scheduler state and request lifecycle
        self.scheduler.on_prefill_complete(req)  # moves to decoding, sets first_token_time
        req.output_token_ids.append(first_token)

        # If max_tokens == 1, we're done immediately
        if req.num_output_tokens >= req.max_tokens or first_token == self.eos_token_id:
            req.mark_done()
            self.scheduler.decoding.remove(req)
            self.scheduler.completed.append(req)

        # Store prefix in cache so future requests with same prompt skip prefill
        complete_blocks_len = (len(req.token_ids) // self.prefix_cache.block_size) * self.prefix_cache.block_size
        if complete_blocks_len > 0:
            block_ids = self.allocator.blocks_for(req.request_id)
            self.prefix_cache.store(req.token_ids[:complete_blocks_len], block_ids)

    def _decode_step(self, req: Request) -> bool:
        """Generate one more token for a decode-phase request.

        Returns True if the request is now complete.
        """
        # Feed last generated token
        last_token = req.output_token_ids[-1]
        position = req.num_cached_tokens + req.num_output_tokens - 1

        logits, kv_cache = self.model.forward(
            token_ids=[last_token],
            kv_cache=req.kv_cache,
            position_offset=position,
        )
        req.kv_cache = kv_cache

        # Sample next token
        next_token = int(self._sample(logits[-1:]).item())

        done = self.scheduler.on_decode_step(req, next_token, self.eos_token_id)
        return done

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample token(s) from logits using engine-level sampling params."""
        if isinstance(logits, torch.Tensor):
            return sample(logits, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k)
        raise TypeError(f"Expected torch.Tensor, got {type(logits)}")

    # ------------------------------------------------------------------
    # Convenience: run a list of requests in one call
    # ------------------------------------------------------------------

    def generate(self, requests: List[Request], max_steps: int = 10_000) -> List[Request]:
        """Submit requests and run until all complete. Returns completed requests."""
        self.submit_many(requests)
        return self.run_until_done(max_steps=max_steps)
