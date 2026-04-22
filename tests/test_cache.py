"""
Tests for the paged KV-cache block allocator.

All CPU, no GPU required.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from agentserve.engine.cache import BlockAllocator


@pytest.fixture
def alloc():
    return BlockAllocator(num_blocks=32, block_size=16)


class TestBasicAllocation:
    def test_starts_fully_free(self, alloc):
        assert alloc.num_free_blocks() == 32
        assert alloc.num_used_blocks() == 0

    def test_allocate_single_block(self, alloc):
        ids = alloc.allocate("req1", 1)
        assert len(ids) == 1
        assert alloc.num_free_blocks() == 31
        assert alloc.num_used_blocks() == 1

    def test_allocate_multiple_blocks(self, alloc):
        ids = alloc.allocate("req1", 4)
        assert len(ids) == 4
        assert alloc.num_free_blocks() == 28

    def test_free_returns_blocks(self, alloc):
        alloc.allocate("req1", 4)
        alloc.free("req1")
        assert alloc.num_free_blocks() == 32
        assert alloc.num_used_blocks() == 0

    def test_blocks_are_unique(self, alloc):
        ids_a = alloc.allocate("req1", 4)
        ids_b = alloc.allocate("req2", 4)
        assert set(ids_a).isdisjoint(set(ids_b))

    def test_free_nonexistent_is_noop(self, alloc):
        alloc.free("nonexistent")  # should not raise
        assert alloc.num_free_blocks() == 32


class TestMemoryUsage:
    def test_memory_usage_zero_initially(self, alloc):
        assert alloc.memory_usage() == 0.0

    def test_memory_usage_half_full(self, alloc):
        alloc.allocate("req1", 16)
        assert alloc.memory_usage() == pytest.approx(0.5)

    def test_memory_usage_full(self, alloc):
        alloc.allocate("req1", 32)
        assert alloc.memory_usage() == pytest.approx(1.0)


class TestExhaustion:
    def test_raises_when_out_of_blocks(self, alloc):
        with pytest.raises(MemoryError):
            alloc.allocate("req1", 33)  # more than pool size

    def test_exact_pool_size_succeeds(self, alloc):
        ids = alloc.allocate("req1", 32)
        assert len(ids) == 32

    def test_can_allocate_after_free(self, alloc):
        alloc.allocate("req1", 32)
        alloc.free("req1")
        ids = alloc.allocate("req2", 32)
        assert len(ids) == 32


class TestChurn:
    def test_no_fragmentation_after_churn(self, alloc):
        """Allocate and free many requests; pool should always recover to 32 free."""
        for i in range(20):
            alloc.allocate(f"req{i}", 2)
            alloc.free(f"req{i}")
        assert alloc.num_free_blocks() == 32

    def test_interleaved_alloc_free(self, alloc):
        for i in range(16):
            alloc.allocate(f"req{i}", 1)
        for i in range(8):
            alloc.free(f"req{i}")
        assert alloc.num_free_blocks() == 24
        # Allocate the freed blocks again
        alloc.allocate("req_new", 8)
        assert alloc.num_free_blocks() == 16


class TestBlocksNeeded:
    def test_exact_multiple(self, alloc):
        assert alloc.blocks_needed(16) == 1
        assert alloc.blocks_needed(32) == 2

    def test_rounds_up(self, alloc):
        assert alloc.blocks_needed(1) == 1
        assert alloc.blocks_needed(17) == 2
        assert alloc.blocks_needed(31) == 2

    def test_zero_tokens(self, alloc):
        assert alloc.blocks_needed(0) == 0


class TestBlocksFor:
    def test_blocks_for_returns_owned(self, alloc):
        ids = alloc.allocate("req1", 3)
        assert set(alloc.blocks_for("req1")) == set(ids)

    def test_blocks_for_empty_after_free(self, alloc):
        alloc.allocate("req1", 3)
        alloc.free("req1")
        assert alloc.blocks_for("req1") == []
