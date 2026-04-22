"""
Paged KV-cache block allocator.

Paging avoids KV-cache fragmentation: instead of one contiguous tensor per
request, memory is divided into fixed-size blocks.  Requests are assigned
blocks on demand; when they finish, the blocks are returned to the free pool.

Physical layout (when running with a real model on GPU):
  kv_pool: tensor of shape
    [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
  kv_pool[0] = keys, kv_pool[1] = values.

In CPU / mock-model mode the tensor pool is optional — the allocator still
tracks block ownership so the scheduler can reason about memory pressure.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Optional
import torch


@dataclass
class Block:
    block_id: int
    ref_count: int = 0

    def is_free(self) -> bool:
        return self.ref_count == 0


class BlockAllocator:
    """
    Manages a pool of KV-cache blocks.

    Blocks are identified by integer IDs in [0, num_blocks).
    A request holds one or more block IDs; when it finishes, all its blocks
    are returned to the free pool.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int = 16,
        num_layers: int = 1,
        num_kv_heads: int = 1,
        head_dim: int = 64,
        allocate_tensor: bool = False,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size

        self._blocks: List[Block] = [Block(i) for i in range(num_blocks)]
        self._free: deque[int] = deque(range(num_blocks))
        self._used: set[int] = set()
        # request_id → list of block IDs owned by that request
        self._request_blocks: Dict[str, List[int]] = {}

        # Optional physical KV tensor pool (for GPU operation)
        self.kv_pool: Optional[torch.Tensor] = None
        if allocate_tensor:
            # Shape: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
            self.kv_pool = torch.zeros(
                2, num_layers, num_blocks, block_size, num_kv_heads, head_dim
            )

    # ------------------------------------------------------------------
    # Core allocation
    # ------------------------------------------------------------------

    def num_free_blocks(self) -> int:
        return len(self._free)

    def num_used_blocks(self) -> int:
        return len(self._used)

    def memory_usage(self) -> float:
        """Fraction of blocks currently allocated, in [0, 1]."""
        return self.num_used_blocks() / self.num_blocks

    def can_allocate(self, num_blocks_needed: int = 1) -> bool:
        return len(self._free) >= num_blocks_needed

    def allocate(self, request_id: str, num_blocks: int = 1) -> List[int]:
        """Allocate num_blocks for a request. Returns the allocated block IDs."""
        if not self.can_allocate(num_blocks):
            raise MemoryError(
                f"Out of KV-cache blocks: need {num_blocks}, have {len(self._free)}"
            )
        allocated = []
        for _ in range(num_blocks):
            block_id = self._free.popleft()
            block = self._blocks[block_id]
            assert block.is_free(), f"Block {block_id} is not free"
            block.ref_count = 1
            self._used.add(block_id)
            allocated.append(block_id)

        self._request_blocks.setdefault(request_id, []).extend(allocated)
        return allocated

    def free(self, request_id: str) -> None:
        """Return all blocks owned by request_id to the free pool."""
        block_ids = self._request_blocks.pop(request_id, [])
        for block_id in block_ids:
            block = self._blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._used.discard(block_id)
                self._free.append(block_id)

    def blocks_for(self, request_id: str) -> List[int]:
        """Return block IDs currently owned by a request."""
        return list(self._request_blocks.get(request_id, []))

    def blocks_needed(self, num_tokens: int) -> int:
        """How many blocks are required to store num_tokens?"""
        return (num_tokens + self.block_size - 1) // self.block_size
