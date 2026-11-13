"""Page-granular radix tree for KV prefix reuse (RadixAttention, SGLang, NeurIPS'24), with an
optional host-memory tier (L2; design doc §4.4, docs/P3实施计划.md; the structure of SGLang's
HiCache).

Each node stores a run of token ids (a multiple of ``block_size``) and where their KV lives:
``value`` = GPU block ids (L1) or None, ``host_value`` = host page ids (L2) or None. A node is
in L1, in L2, or in both; a node without either is removed. Without a host pool every node is
in L1 and the tree behaves as a plain GPU prefix cache.

Ownership rules:

* The tree holds one allocator reference per cached GPU block and per host page.
* A request that uses cached blocks holds its own reference and locks the path from its
  deepest matched node to the root (``lock_ref``); locked nodes are never evicted.
* Locks are path-wise, so every node with ``lock_ref == 0`` sits in a fully unlocked subtree.

Tiers:

* L1 nodes are prefix-closed (a node in L1 has its parent in L1), so a lookup returns a run of
  L1 nodes followed by a run of L2-only nodes.
* ``write_event`` marks a GPU -> host copy in flight (the GPU blocks must stay); ``load_event`` a
  host -> GPU copy in flight (the GPU blocks are not valid yet). ``None``: done (or synchronous).
* L1 eviction takes LRU nodes in L1 without children in L1. With a host pool, a node whose KV
  is on the host drops its GPU copy; one that is not is written back first if free host pages
  are left; otherwise the node and its (L2-only) subtree are removed.
* L2 eviction takes LRU nodes with a host copy: L2-only leaves are removed, nodes also in L1
  drop their host copy.
"""

from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from tandem.cache.block_allocator import BlockAllocator


def _done(event: Any) -> bool:
    return event is None or event.query()


class RadixNode:
    __slots__ = ("key", "value", "host_value", "children", "parent", "lock_ref", "last_access", "write_event",
                 "load_event")

    def __init__(
        self,
        key: tuple[int, ...] = (),
        value: Optional[list[int]] = None,
        parent: Optional["RadixNode"] = None,
        host_value: Optional[list[int]] = None,
    ) -> None:
        self.key = key
        self.value = value
        self.host_value = host_value
        self.children: dict[tuple[int, ...], RadixNode] = {}
        self.parent = parent
        self.lock_ref = 0
        self.last_access = time.monotonic()
        self.write_event: Any = None
        self.load_event: Any = None

    @property
    def on_gpu(self) -> bool:
        return self.value is not None


@dataclass
class PrefixMatch:
    """A lookup: the L1 run (``blocks``, deepest node ``node``), the L2-only nodes after it, and
    the loads still in flight into the L1 run."""

    blocks: list[int]
    node: RadixNode
    host_nodes: list[RadixNode] = field(default_factory=list)
    pending: list[Any] = field(default_factory=list)

    @property
    def num_host_blocks(self) -> int:
        return sum(len(n.host_value) for n in self.host_nodes)


class RadixCache:
    def __init__(self, block_size: int, allocator: BlockAllocator, host: Optional[Any] = None) -> None:
        """``host``: a tandem.cache.host_pool.HostKVPool (its ``allocator`` hands out host pages,
        ``to_host`` / ``to_gpu`` copy pages and return events)."""
        self.bs = block_size
        self.allocator = allocator
        self.host = host
        self.root = RadixNode(value=[])
        self.root.lock_ref = 1  # never evicted
        self.num_cached_blocks = 0  # GPU blocks held by the tree
        self.num_host_pages = 0  # host pages held by the tree
        self.num_query_tokens = 0
        self.num_hit_tokens = 0  # found in L1
        self.num_host_hit_tokens = 0  # found in L2 only

    # ----------------------------------------------------------------- helpers
    def _first_block(self, key: Sequence[int], start: int = 0) -> tuple[int, ...]:
        return tuple(key[start : start + self.bs])

    def _num_blocks(self, node: RadixNode) -> int:
        return len(node.key) // self.bs

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
        new = RadixNode(child.key[: n_blocks * bs], None if child.value is None else child.value[:n_blocks],
                        child.parent, None if child.host_value is None else child.host_value[:n_blocks])
        new.lock_ref = child.lock_ref
        new.last_access = child.last_access
        new.write_event, new.load_event = child.write_event, child.load_event  # same copies
        child.parent.children[self._first_block(new.key)] = new
        child.key = child.key[n_blocks * bs :]
        if child.value is not None:
            child.value = child.value[n_blocks:]
        if child.host_value is not None:
            child.host_value = child.host_value[n_blocks:]
        child.parent = new
        new.children[self._first_block(child.key)] = child
        return new

    def _gpu_children(self, node: RadixNode) -> bool:
        return any(c.value is not None for c in node.children.values())

    # ------------------------------------------------------------------ lookup
    def match(self, tokens: Sequence[int], count: bool = True) -> PrefixMatch:
        """Longest cached prefix of ``tokens`` in whole blocks, across both tiers. ``count``:
        add to the hit statistics (not for a repeated lookup of the same request)."""
        bs = self.bs
        n_full = len(tokens) // bs * bs
        node, i = self.root, 0
        out = PrefixMatch(blocks=[], node=self.root)
        now = time.monotonic()
        while i < n_full:
            child = node.children.get(self._first_block(tokens, i))
            if child is None:
                break
            m = self._match_blocks(child.key, tokens, i)
            if m < self._num_blocks(child):
                child = self._split(child, m)
            child.last_access = now
            if child.value is not None and not out.host_nodes:
                out.blocks.extend(child.value)
                out.node = child
                if child.load_event is not None:
                    if _done(child.load_event):
                        child.load_event = None
                    elif child.load_event not in out.pending:
                        out.pending.append(child.load_event)
            else:  # L2-only (the L1 run has ended: below it everything is L2-only)
                out.host_nodes.append(child)
            node = child
            i += m * bs
        if count:
            self.num_query_tokens += len(tokens)
            self.num_hit_tokens += len(out.blocks) * bs
            self.num_host_hit_tokens += out.num_host_blocks * bs
        return out

    def match_prefix(self, tokens: Sequence[int]) -> tuple[list[int], RadixNode]:
        """Longest prefix of ``tokens`` cached in L1 -> (block ids, deepest node)."""
        m = self.match(tokens)
        return m.blocks, m.node

    # ------------------------------------------------------------------ insert
    def insert(self, tokens: Sequence[int], block_ids: Sequence[int]) -> tuple[RadixNode, list[int]]:
        """Insert whole blocks of ``tokens`` computed into ``block_ids``.

        Returns the deepest node covering the inserted range and the canonical block ids
        for it: existing tree blocks where the content was already cached in L1, otherwise
        the caller's blocks (which the tree now references). An L2-only node on the way adopts
        the caller's blocks as its L1 copy.
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
            if m < self._num_blocks(child):
                child = self._split(child, m)
            child.last_access = now
            if child.value is None:  # L2-only: the caller's blocks become its L1 copy
                child.value = list(block_ids[b : b + m])
                self.allocator.incref(child.value)
                self.num_cached_blocks += m
            if _done(child.load_event):
                child.load_event = None
                canonical.extend(child.value)
            else:  # still loading: the caller keeps its own (valid) blocks
                canonical.extend(block_ids[b : b + m])
            node = child
            i += m * bs
            b += m
        return node, canonical

    # ------------------------------------------------------------------- locks
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
        return sum(len(n.value) for n in self._iter_nodes() if n.lock_ref == 0 and n.value is not None)

    # ------------------------------------------------------------------- host copies
    def _alloc_host(self, n: int, make_room: bool) -> Optional[list[int]]:
        free = self.host.allocator.num_free
        if n > free and make_room:
            self.evict_host(n - free)
        if n > self.host.allocator.num_free:
            return None
        self.num_host_pages += n
        return self.host.allocator.allocate(n)

    def _write(self, nodes: list[RadixNode], make_room: bool) -> list[RadixNode]:
        """Copy the GPU blocks of ``nodes`` (L1 without a host copy) to newly allocated host pages
        in one batch; the prefix of ``nodes`` that fits. ``make_room``: evict LRU host copies
        for them (write-through of fresh KV), else only free pages (write-back of an LRU node must
        not push out host copies more recent than itself). Returns the nodes being written."""
        written, gpu, pages = [], [], []
        for node in nodes:
            got = self._alloc_host(len(node.value), make_room)
            if got is None:
                break
            node.host_value = got
            written.append(node)
            gpu.extend(node.value)
            pages.extend(got)
        if written:
            event = self.host.to_host(gpu, pages)
            for node in written:
                node.write_event = event
        return written

    def write_through(self, node: RadixNode) -> int:
        """Start copying the L1 nodes on the path root -> ``node`` that have no host copy yet
        (root first, so a full host pool keeps the shorter prefix). Returns blocks written."""
        if self.host is None:
            return 0
        path = []
        while node is not None and node is not self.root:
            if node.value is not None and node.host_value is None and _done(node.load_event):
                path.append(node)
            node = node.parent  # type: ignore[assignment]
        path.reverse()
        return sum(len(n.value) for n in self._write(path, make_room=True))

    def load(self, nodes: Sequence[RadixNode]) -> tuple[list[RadixNode], Any]:
        """Bring L2-only ``nodes`` (a lookup's ``host_nodes``, path order) into L1: allocate GPU
        blocks (evicting L1 if needed) and start the copy. Loads the prefix of ``nodes`` that
        fits; returns (loaded nodes, event). The caller locks the path first, so the L1 run of
        the lookup is not evicted to make room."""
        need = sum(len(n.host_value) for n in nodes)
        if need > self.allocator.num_free:
            self.evict(need - self.allocator.num_free)
        loaded, gpu, pages = [], [], []
        budget = self.allocator.num_free
        for node in nodes:
            k = len(node.host_value)
            if k > budget:
                break
            budget -= k
            loaded.append(node)
            pages.extend(node.host_value)
        if not loaded:
            return [], None
        gpu = self.allocator.allocate(len(pages))
        self.num_cached_blocks += len(gpu)
        event = self.host.to_gpu(pages, gpu)
        at = 0
        for node in loaded:
            k = len(node.host_value)
            node.value = gpu[at : at + k]
            node.load_event = event
            at += k
        return loaded, event

    # ------------------------------------------------------------------ eviction
    def _drop_gpu(self, node: RadixNode) -> int:
        for blk in node.value:
            if self.allocator.ref_count(blk) != 1:
                raise RuntimeError(f"evicting block {blk} that is still referenced")
        self.allocator.decref(node.value)
        n = len(node.value)
        self.num_cached_blocks -= n
        node.value = None
        return n

    def _remove(self, node: RadixNode) -> int:
        """Remove ``node`` and its subtree from the tree (all unlocked); returns GPU blocks freed."""
        freed, stack = 0, [node]
        while stack:
            n = stack.pop()
            stack.extend(n.children.values())
            if n.value is not None:
                freed += self._drop_gpu(n)
            if n.host_value is not None:
                self.host.allocator.decref(n.host_value)
                self.num_host_pages -= len(n.host_value)
                n.host_value = None
        parent = node.parent
        del parent.children[self._first_block(node.key)]
        node.parent = None
        return freed

    def evict(self, num_blocks: int) -> int:
        """Free at least ``num_blocks`` GPU blocks from unlocked LRU L1 nodes (if possible)."""
        tie = itertools.count()

        def candidate(n: RadixNode) -> bool:
            return (n is not self.root and n.parent is not None and n.value is not None and n.lock_ref == 0
                    and _done(n.load_event) and not self._gpu_children(n))

        heap = [(n.last_access, next(tie), n) for n in self._iter_nodes() if candidate(n)]
        heapq.heapify(heap)
        freed, writing = 0, []
        while freed < num_blocks and heap:
            _, _, node = heapq.heappop(heap)
            if not candidate(node):
                continue
            parent = node.parent
            if self.host is not None and node.host_value is not None:
                if not _done(node.write_event):
                    writing.append(node)
                    continue
                node.write_event = None
                freed += self._drop_gpu(node)  # stays in L2
            elif self.host is not None and self._write([node], make_room=False):
                writing.append(node)  # written back below once the copy lands
                continue
            else:
                freed += self._remove(node)
            if parent is not self.root and candidate(parent):
                heapq.heappush(heap, (parent.last_access, next(tie), parent))
        for node in writing:  # under pressure: wait for write-backs in flight
            if freed >= num_blocks:
                break
            if node.write_event is not None:
                node.write_event.synchronize()
                node.write_event = None
            if candidate(node):
                freed += self._drop_gpu(node)
        return freed

    def evict_host(self, num_pages: int) -> int:
        """Free at least ``num_pages`` host pages from unlocked LRU nodes (if possible)."""
        tie = itertools.count()

        def candidate(n: RadixNode) -> bool:
            return (n.parent is not None and n.host_value is not None and n.lock_ref == 0
                    and _done(n.write_event) and (n.value is not None or not n.children))

        heap = [(n.last_access, next(tie), n) for n in self._iter_nodes() if candidate(n)]
        heapq.heapify(heap)
        freed = 0
        while freed < num_pages and heap:
            _, _, node = heapq.heappop(heap)
            if not candidate(node):
                continue
            k = len(node.host_value)
            parent = node.parent
            if node.value is not None:  # also in L1: drop the host copy only
                self.host.allocator.decref(node.host_value)
                self.num_host_pages -= k
                node.host_value = None
                node.write_event = None
            else:  # an L2-only leaf
                self._remove(node)
            freed += k
            if parent is not self.root and candidate(parent):
                heapq.heappush(heap, (parent.last_access, next(tie), parent))
        return freed

    @property
    def hit_rate(self) -> float:
        return self.num_hit_tokens / self.num_query_tokens if self.num_query_tokens else 0.0

    @property
    def host_hit_rate(self) -> float:
        return self.num_host_hit_tokens / self.num_query_tokens if self.num_query_tokens else 0.0
