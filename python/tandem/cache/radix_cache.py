"""Page-granular radix tree for KV prefix reuse (RadixAttention, SGLang, NeurIPS'24).

Each node stores a run of token ids (a multiple of ``block_size``) and the KV block ids
holding them. Ownership rules:

* The tree holds one allocator reference per cached block.
* A request that uses cached blocks holds its own reference and locks the path from its
  deepest matched node to the root (``lock_ref``); locked nodes are never evicted.
* Locks are path-wise, so every node with ``lock_ref == 0`` sits in a fully unlocked
  subtree and its blocks are referenced by the tree only. Eviction frees LRU leaves.
"""

from __future__ import annotations

import heapq
import itertools
import time
from typing import Optional, Sequence

from tandem.cache.block_allocator import BlockAllocator


class RadixNode:
    __slots__ = ("key", "value", "children", "parent", "lock_ref", "last_access")

    def __init__(
        self,
        key: tuple[int, ...] = (),
        value: Optional[list[int]] = None,
        parent: Optional["RadixNode"] = None,
    ) -> None:
        self.key = key
        self.value: list[int] = value if value is not None else []
        self.children: dict[tuple[int, ...], RadixNode] = {}
        self.parent = parent
        self.lock_ref = 0
        self.last_access = time.monotonic()


class RadixCache:
    def __init__(self, block_size: int, allocator: BlockAllocator) -> None:
        self.bs = block_size
        self.allocator = allocator
        self.root = RadixNode()
        self.root.lock_ref = 1  # never evicted
        self.num_cached_blocks = 0
        self.num_query_tokens = 0
        self.num_hit_tokens = 0

    # ----------------------------------------------------------------- helpers
    def _first_block(self, key: Sequence[int], start: int = 0) -> tuple[int, ...]:
        return tuple(key[start : start + self.bs])

    def _match_blocks(self, node_key: tuple[int, ...], tokens: Sequence[int], start: int) -> int:
        """Number of whole blocks shared by ``node_key`` and ``tokens[start:]``."""
        bs = self.bs
        limit = min(len(node_key), len(tokens) - start) // bs
        n = 0
        while n < limit:
            if node_key[n * bs : (n + 1) * bs] != tuple(tokens[start + n * bs : start + (n + 1) * bs]):
                break
            n += 1
        return n

    def _split(self, child: RadixNode, n_blocks: int) -> RadixNode:
        """Split ``child`` so that a new parent node holds its first ``n_blocks`` blocks."""
        bs = self.bs
        assert child.parent is not None
        new = RadixNode(child.key[: n_blocks * bs], child.value[:n_blocks], child.parent)
        new.lock_ref = child.lock_ref
        new.last_access = child.last_access
        child.parent.children[self._first_block(new.key)] = new
        child.key = child.key[n_blocks * bs :]
        child.value = child.value[n_blocks:]
        child.parent = new
        new.children[self._first_block(child.key)] = child
        return new

    # ------------------------------------------------------------------ public
    def match_prefix(self, tokens: Sequence[int]) -> tuple[list[int], RadixNode]:
        """Longest cached prefix of ``tokens`` in whole blocks -> (block ids, deepest node)."""
        bs = self.bs
        n_full = len(tokens) // bs * bs
        node, blocks, i = self.root, [], 0
        now = time.monotonic()
        while i < n_full:
            child = node.children.get(self._first_block(tokens, i))
            if child is None:
                break
            m = self._match_blocks(child.key, tokens, i)
            if m < len(child.value):
                child = self._split(child, m)
            child.last_access = now
            blocks.extend(child.value)
            node = child
            i += m * bs
        self.num_query_tokens += len(tokens)
        self.num_hit_tokens += i
        return blocks, node

    def insert(self, tokens: Sequence[int], block_ids: Sequence[int]) -> tuple[RadixNode, list[int]]:
        """Insert whole blocks of ``tokens``.

        Returns the deepest node covering the inserted range and the canonical block ids
        for it: existing tree blocks where the content was already cached, otherwise the
        caller's blocks (which the tree now references).
        """
        bs = self.bs
        n_blocks = min(len(tokens) // bs, len(block_ids))
        node, canonical, i, b = self.root, [], 0, 0
        now = time.monotonic()
        while b < n_blocks:
            first = self._first_block(tokens, i)
            child = node.children.get(first)
            if child is None:
                new = RadixNode(tuple(tokens[i : n_blocks * bs]), list(block_ids[b:n_blocks]), node)
                new.last_access = now
                node.children[first] = new
                self.allocator.incref(new.value)
                self.num_cached_blocks += len(new.value)
                canonical.extend(new.value)
                return new, canonical
            m = min(self._match_blocks(child.key, tokens[: n_blocks * bs], i), n_blocks - b)
            if m < len(child.value):
                child = self._split(child, m)
            child.last_access = now
            canonical.extend(child.value)
            node = child
            i += m * bs
            b += m
        return node, canonical

    def inc_lock(self, node: RadixNode) -> None:
        while node is not None and node is not self.root:
            node.lock_ref += 1
            node = node.parent  # type: ignore[assignment]

    def dec_lock(self, node: RadixNode) -> None:
        while node is not None and node is not self.root:
            node.lock_ref -= 1
            if node.lock_ref < 0:
                raise RuntimeError("radix lock underflow")
            node = node.parent  # type: ignore[assignment]

    def _iter_nodes(self):
        stack = list(self.root.children.values())
        while stack:
            node = stack.pop()
            yield node
            stack.extend(node.children.values())

    def num_evictable_blocks(self) -> int:
        return sum(len(n.value) for n in self._iter_nodes() if n.lock_ref == 0)

    def evict(self, num_blocks: int) -> int:
        """Free at least ``num_blocks`` blocks from unlocked LRU leaves (if possible)."""
        tie = itertools.count()
        heap = [
            (n.last_access, next(tie), n) for n in self._iter_nodes() if not n.children and n.lock_ref == 0
        ]
        heapq.heapify(heap)
        freed = 0
        while freed < num_blocks and heap:
            _, _, leaf = heapq.heappop(heap)
            parent = leaf.parent
            if leaf.children or leaf.lock_ref > 0 or parent is None:
                continue
            for blk in leaf.value:
                if self.allocator.ref_count(blk) != 1:
                    raise RuntimeError(f"evicting block {blk} that is still referenced")
            self.allocator.decref(leaf.value)
            freed += len(leaf.value)
            self.num_cached_blocks -= len(leaf.value)
            del parent.children[self._first_block(leaf.key)]
            leaf.parent = None
            if parent is not self.root and not parent.children and parent.lock_ref == 0:
                heapq.heappush(heap, (parent.last_access, next(tie), parent))
        return freed

    @property
    def hit_rate(self) -> float:
        return self.num_hit_tokens / self.num_query_tokens if self.num_query_tokens else 0.0
