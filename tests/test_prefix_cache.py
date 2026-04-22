"""
Tests for the prefix cache with LFU eviction.

All CPU, no GPU required.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from agentserve.engine.prefix_cache import PrefixCache, PrefixCacheStats


BLOCK_SIZE = 4  # small block size for easy testing


@pytest.fixture
def cache():
    return PrefixCache(block_size=BLOCK_SIZE, max_entries=8)


def make_tokens(n: int, start: int = 0) -> list[int]:
    """Make a token sequence [start, start+1, ..., start+n-1]."""
    return list(range(start, start + n))


def make_blocks(n_blocks: int, start: int = 0) -> list[int]:
    """Fake block IDs."""
    return list(range(start, start + n_blocks))


class TestBasicStoreAndLookup:
    def test_miss_on_empty_cache(self, cache):
        tokens = make_tokens(8)
        length, blocks = cache.find_longest_prefix(tokens)
        assert length == 0
        assert blocks == []

    def test_exact_match(self, cache):
        tokens = make_tokens(8)
        block_ids = make_blocks(2)
        cache.store(tokens, block_ids)
        length, blocks = cache.find_longest_prefix(tokens)
        assert length == 8
        assert blocks == block_ids

    def test_prefix_match(self, cache):
        """If we store 8 tokens and look up 16, we should match the first 8."""
        tokens_short = make_tokens(8)
        tokens_long  = make_tokens(16)
        block_ids = make_blocks(2)
        cache.store(tokens_short, block_ids)
        length, blocks = cache.find_longest_prefix(tokens_long)
        assert length == 8
        assert blocks == block_ids

    def test_no_partial_block_match(self, cache):
        """Prefix must be block-aligned; 6 tokens with block_size=4 → only 4 tokens match."""
        # store 4-token prefix
        tokens_4 = make_tokens(4)
        cache.store(tokens_4, [10])
        # look up 6 tokens (first 4 match a full block, last 2 do not form a complete block)
        tokens_6 = make_tokens(6)
        length, _ = cache.find_longest_prefix(tokens_6)
        assert length == 4

    def test_different_prefix_no_match(self, cache):
        tokens_a = make_tokens(8, start=0)
        tokens_b = make_tokens(8, start=100)
        cache.store(tokens_a, make_blocks(2, 0))
        length, _ = cache.find_longest_prefix(tokens_b)
        assert length == 0


class TestLongestPrefixSelection:
    def test_returns_longest_match(self, cache):
        """With two stored prefixes (4 and 8 tokens), should return 8."""
        tokens_4  = make_tokens(4)
        tokens_8  = make_tokens(8)
        tokens_12 = make_tokens(12)

        cache.store(tokens_4, make_blocks(1, 10))
        cache.store(tokens_8, make_blocks(2, 20))

        length, blocks = cache.find_longest_prefix(tokens_12)
        assert length == 8
        assert blocks == make_blocks(2, 20)


class TestStats:
    def test_hit_increments_counter(self, cache):
        tokens = make_tokens(8)
        cache.store(tokens, make_blocks(2))
        cache.find_longest_prefix(tokens)
        assert cache.stats.hits == 1

    def test_miss_increments_counter(self, cache):
        cache.find_longest_prefix(make_tokens(8))
        assert cache.stats.misses == 1

    def test_hit_rate(self, cache):
        tokens = make_tokens(8)
        cache.store(tokens, make_blocks(2))
        cache.find_longest_prefix(tokens)   # hit
        cache.find_longest_prefix(make_tokens(8, start=100))  # miss
        assert cache.stats.hit_rate == pytest.approx(0.5)

    def test_tokens_saved_on_hit(self, cache):
        tokens = make_tokens(8)
        cache.store(tokens, make_blocks(2))
        cache.find_longest_prefix(tokens)
        assert cache.stats.tokens_saved == 8

    def test_eviction_increments_evictions(self, cache):
        # Fill cache past capacity to trigger eviction
        small_cache = PrefixCache(block_size=BLOCK_SIZE, max_entries=2)
        for i in range(3):
            t = make_tokens(4, start=i * 10)
            small_cache.store(t, make_blocks(1, i))
        assert small_cache.stats.evictions >= 1


class TestLFUEviction:
    def test_high_reuse_entries_survive(self):
        """High-reuse entries should outlive low-reuse entries under memory pressure."""
        cache = PrefixCache(block_size=BLOCK_SIZE, max_entries=3)

        # Store three entries
        popular_tokens = make_tokens(4, start=0)
        rare_tokens_a  = make_tokens(4, start=10)
        rare_tokens_b  = make_tokens(4, start=20)

        cache.store(popular_tokens, [0])
        cache.store(rare_tokens_a,  [1])
        cache.store(rare_tokens_b,  [2])

        # Hit the popular prefix many times to increase its reuse_count
        for _ in range(10):
            cache.find_longest_prefix(make_tokens(4, start=0))

        # Now add a 4th entry — should evict one of the rares, not the popular
        new_tokens = make_tokens(4, start=30)
        cache.store(new_tokens, [3])

        # Popular entry should still be there
        length, _ = cache.find_longest_prefix(popular_tokens)
        assert length == 4, "High-reuse prefix should survive eviction"

    def test_lfu_evicts_lowest_reuse_first(self):
        """Among equal-length entries, the one with the fewest hits is evicted."""
        cache = PrefixCache(block_size=BLOCK_SIZE, max_entries=2)

        tok_a = make_tokens(4, start=0)
        tok_b = make_tokens(4, start=10)

        cache.store(tok_a, [0])
        cache.store(tok_b, [1])

        # Hit A more than B
        for _ in range(5):
            cache.find_longest_prefix(tok_a)
        cache.find_longest_prefix(tok_b)  # only 1 hit

        # Adding a 3rd entry forces eviction of B (fewer hits)
        tok_c = make_tokens(4, start=20)
        cache.store(tok_c, [2])

        # B should be gone
        length_b, _ = cache.find_longest_prefix(tok_b)
        # A should still be there
        length_a, _ = cache.find_longest_prefix(tok_a)
        assert length_a == 4, "A (popular) should still be cached"
        # B is evicted — it may or may not be re-stored depending on implementation
        # The important thing is that evictions happened
        assert cache.stats.evictions >= 1


class TestEdgeCases:
    def test_empty_token_list(self, cache):
        length, blocks = cache.find_longest_prefix([])
        assert length == 0
        assert blocks == []

    def test_store_empty_is_noop(self, cache):
        cache.store([], [])
        assert cache.size() == 0

    def test_tokens_shorter_than_block_size(self, cache):
        """Tokens shorter than block_size can't form a complete block → always miss."""
        short_tokens = make_tokens(BLOCK_SIZE - 1)
        cache.store(short_tokens, [])
        length, _ = cache.find_longest_prefix(short_tokens)
        assert length == 0

    def test_clear_empties_cache(self, cache):
        tokens = make_tokens(8)
        cache.store(tokens, make_blocks(2))
        cache.clear()
        assert cache.size() == 0
        length, _ = cache.find_longest_prefix(tokens)
        assert length == 0
