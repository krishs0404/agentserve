"""
Prefix cache with agent-aware eviction.

Many agent calls share the same system prompt (e.g., "You are a helpful
assistant working with tools...").  Computing attention over that prefix
on every request is wasteful; we can save and reuse the KV-cache blocks.

Key insight: standard LRU eviction is wrong for agents.
  - LRU evicts entries not recently used.
  - A system prompt might not be used for 30 seconds, but it will be used
    10,000 more times in the session.
  - We evict by LOWEST reuse_count first (frequency-based), not by recency.
  - Entries with high reuse_count are effectively "pinned" — they survive
    memory pressure as long as cheaper entries exist.

API:
  store(token_ids, kv_blocks)
  find_longest_prefix(token_ids) → (prefix_len, block_ids)
  update_reuse(token_ids_prefix)   — call on cache hit to increment counter
  evict_lfu(n)                    — evict n entries by lowest reuse_count

Hashing:
  We hash complete blocks of tokens using a chained hash (each block's hash
  depends on the previous block's hash + its tokens).  This lets us do
  variable-length prefix matching at block granularity.
"""

from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class PrefixEntry:
    token_ids: List[int]       # full token sequence for this cached prefix
    block_ids: List[int]       # KV-cache block IDs holding the KV tensors
    reuse_count: int = 0       # how many times this prefix has been hit
    last_access_step: int = 0  # engine step counter, for tie-breaking


@dataclass
class PrefixCacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    tokens_saved: int = 0      # prompt tokens skipped due to prefix hit

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def bytes_saved(self) -> int:
        # rough: each saved token avoided one attention computation
        # real savings depend on model size; report token count instead
        return self.tokens_saved


def _hash_block(token_ids: List[int], prefix_hash: int = 0) -> int:
    """Stable hash of a token block, chained from the previous block's hash."""
    h = hashlib.blake2b(digest_size=8)
    h.update(prefix_hash.to_bytes(8, "little", signed=False))
    for tid in token_ids:
        h.update(tid.to_bytes(4, "little"))
    return int.from_bytes(h.digest(), "little")


class PrefixCache:
    """
    Stores KV blocks keyed by token prefix hash.

    All operations are at block granularity.  block_size must match the
    block allocator's block_size.
    """

    def __init__(self, block_size: int = 16, max_entries: int = 1024):
        self.block_size = block_size
        self.max_entries = max_entries
        self.stats = PrefixCacheStats()
        self._step: int = 0

        # hash → PrefixEntry; keyed on the LAST block's chained hash
        self._cache: Dict[int, PrefixEntry] = {}
        # token_ids_tuple → hash, for lookup by token sequence
        self._token_to_hash: Dict[tuple, int] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _block_hashes(self, token_ids: List[int]) -> List[int]:
        """Compute chained hashes for every complete block in token_ids."""
        hashes = []
        h = 0
        # Process only complete blocks (floor division gives count of full blocks)
        n_complete = (len(token_ids) // self.block_size) * self.block_size
        for start in range(0, n_complete, self.block_size):
            block = token_ids[start: start + self.block_size]
            h = _hash_block(block, h)
            hashes.append(h)
        return hashes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(self, token_ids: List[int], block_ids: List[int]) -> None:
        """Store a computed prefix in the cache.

        token_ids must be a multiple of block_size (complete blocks only).
        block_ids corresponds 1-to-1 with the blocks covering token_ids.
        """
        if len(token_ids) == 0:
            return
        hashes = self._block_hashes(token_ids)
        if not hashes:
            return

        # We key on the final chained hash, which covers the whole prefix.
        final_hash = hashes[-1]
        key = tuple(token_ids)

        if final_hash in self._cache:
            # Update metadata only
            self._cache[final_hash].reuse_count += 1
            self._cache[final_hash].last_access_step = self._step
            return

        # Evict if at capacity
        if len(self._cache) >= self.max_entries:
            self._evict_lfu(1)

        entry = PrefixEntry(
            token_ids=list(token_ids),
            block_ids=list(block_ids),
            reuse_count=1,
            last_access_step=self._step,
        )
        self._cache[final_hash] = entry
        self._token_to_hash[key] = final_hash

    def find_longest_prefix(
        self, token_ids: List[int]
    ) -> Tuple[int, List[int]]:
        """Find the longest cached prefix of token_ids.

        Returns (matched_token_count, block_ids_for_matched_prefix).
        matched_token_count is a multiple of block_size.
        Returns (0, []) on complete miss.
        """
        self._step += 1

        hashes = self._block_hashes(token_ids)
        if not hashes:
            self.stats.misses += 1
            return 0, []

        # Walk backwards from longest to shortest to find best match
        for i in range(len(hashes) - 1, -1, -1):
            h = hashes[i]
            if h in self._cache:
                entry = self._cache[h]
                matched_tokens = (i + 1) * self.block_size
                # Verify the token content (hash collision guard)
                if entry.token_ids == token_ids[:matched_tokens]:
                    entry.reuse_count += 1
                    entry.last_access_step = self._step
                    self.stats.hits += 1
                    self.stats.tokens_saved += matched_tokens
                    return matched_tokens, list(entry.block_ids)

        self.stats.misses += 1
        return 0, []

    def _evict_lfu(self, n: int) -> None:
        """Evict n entries with the lowest reuse_count (LFU policy)."""
        if not self._cache:
            return
        # Sort by (reuse_count, last_access_step) ascending — lowest reuse first,
        # then oldest access as tie-breaker.
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda kv: (kv[1].reuse_count, kv[1].last_access_step),
        )
        to_remove = sorted_entries[:n]
        for h, entry in to_remove:
            del self._cache[h]
            key = tuple(entry.token_ids)
            self._token_to_hash.pop(key, None)
            self.stats.evictions += 1

    def evict_lfu(self, n: int = 1) -> None:
        """Public eviction method for external memory pressure triggers."""
        self._evict_lfu(n)

    def size(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()
        self._token_to_hash.clear()
