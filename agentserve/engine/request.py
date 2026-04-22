"""
Request dataclass representing one inference request through its lifecycle.

Lifecycle:
  PENDING  → request arrived, waiting for scheduler
  PREFILL  → currently being prefilled (prompt is being processed)
  DECODE   → prompt done, generating tokens one at a time
  DONE     → generation complete (hit max_tokens or EOS)

Timing fields for metrics:
  arrival_time    - wall clock when request was submitted
  prefill_start   - when prefill began (used to compute queue wait time)
  first_token_time - when first output token was generated (TTFT)
  done_time       - when generation finished
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from time import monotonic
from itertools import count
import uuid


class RequestStatus(Enum):
    PENDING = auto()
    PREFILL = auto()
    DECODE  = auto()
    DONE    = auto()


_id_counter = count()


@dataclass
class Request:
    prompt: str
    token_ids: list[int]
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    stop_token_id: int | None = None

    # Set by the engine, not the caller
    request_id: str = field(default_factory=lambda: str(next(_id_counter)))
    status: RequestStatus = RequestStatus.PENDING

    # Timing (seconds, monotonic clock)
    arrival_time: float = field(default_factory=monotonic)
    prefill_start: float = 0.0
    first_token_time: float = 0.0
    done_time: float = 0.0

    # Difficulty classification (set by engine before scheduling)
    difficulty: str = "medium"          # "easy" | "medium" | "hard"
    estimated_output_tokens: int = 100
    priority: int = 1                   # 0 = highest (easy), 2 = lowest (hard)

    # Generated output
    output_token_ids: list[int] = field(default_factory=list)

    # KV cache stored per-request (list of (k, v) tensors, one per layer)
    # None until prefill completes; populated after first forward pass.
    kv_cache: list | None = None

    # Block IDs from the paged allocator (logical bookkeeping)
    block_ids: list[int] = field(default_factory=list)

    # Tokens already processed by prefill (for prefix cache hit tracking)
    num_cached_tokens: int = 0

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def is_done(self) -> bool:
        return self.status == RequestStatus.DONE

    @property
    def ttft(self) -> float:
        """Time-to-first-token in seconds."""
        if self.first_token_time == 0.0:
            return 0.0
        return self.first_token_time - self.arrival_time

    @property
    def latency(self) -> float:
        """Total request latency in seconds."""
        if self.done_time == 0.0:
            return 0.0
        return self.done_time - self.arrival_time

    def mark_prefill_start(self) -> None:
        self.status = RequestStatus.PREFILL
        self.prefill_start = monotonic()

    def mark_first_token(self) -> None:
        self.status = RequestStatus.DECODE
        self.first_token_time = monotonic()

    def mark_done(self) -> None:
        self.status = RequestStatus.DONE
        self.done_time = monotonic()
