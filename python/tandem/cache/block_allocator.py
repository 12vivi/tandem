from __future__ import annotations

from collections import deque
from typing import Iterable


class NoFreeBlocksError(RuntimeError):
    pass


class BlockAllocator:
    """Reference-counted pool of physical KV blocks (PagedAttention, SOSP'23).

    Owners of a reference: each request that uses the block, plus the radix prefix
    cache when the block is cached. A block returns to the free list at refcount 0.
    """

    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.num_blocks = num_blocks
        self._free: deque[int] = deque(range(num_blocks))
        self._ref = [0] * num_blocks

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self, n: int) -> list[int]:
        if n > len(self._free):
            raise NoFreeBlocksError(f"need {n} blocks, only {len(self._free)} free")
        ids = [self._free.popleft() for _ in range(n)]
        for i in ids:
            self._ref[i] = 1
        return ids

    def incref(self, ids: Iterable[int]) -> None:
        for i in ids:
            if self._ref[i] <= 0:
                raise RuntimeError(f"incref on free block {i}")
            self._ref[i] += 1

    def decref(self, ids: Iterable[int]) -> None:
        for i in ids:
            if self._ref[i] <= 0:
                raise RuntimeError(f"decref on free block {i}")
            self._ref[i] -= 1
            if self._ref[i] == 0:
                self._free.append(i)

    def ref_count(self, i: int) -> int:
        return self._ref[i]
