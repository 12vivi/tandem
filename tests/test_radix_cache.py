from __future__ import annotations

from tandem.cache.block_allocator import BlockAllocator
from tandem.cache.radix_cache import RadixCache


def _setup(num_blocks=16, bs=4):
    alloc = BlockAllocator(num_blocks)
    return alloc, RadixCache(bs, alloc)


def test_insert_and_match_whole_blocks():
    alloc, rc = _setup()
    blocks = alloc.allocate(3)
    tokens = list(range(12))
    rc.insert(tokens, blocks)
    got, _ = rc.match_prefix(list(range(8)) + [99, 98, 97, 96])
    assert got == blocks[:2]
    got, _ = rc.match_prefix(list(range(11)))  # partial last block is not matched
    assert got == blocks[:2]


def test_split_on_divergence_shares_common_block():
    alloc, rc = _setup()
    a = alloc.allocate(2)
    rc.insert([1, 2, 3, 4, 5, 6, 7, 8], a)
    b = alloc.allocate(2)
    _, canonical = rc.insert([1, 2, 3, 4, 9, 10, 11, 12], b)
    assert canonical == [a[0], b[1]]  # first block already cached under a[0]
    got, _ = rc.match_prefix([1, 2, 3, 4, 9, 10, 11, 12])
    assert got == [a[0], b[1]]
    got, _ = rc.match_prefix([1, 2, 3, 4, 5, 6, 7, 8])
    assert got == a


def test_locked_nodes_are_not_evicted_and_unlock_frees():
    alloc, rc = _setup(num_blocks=4)
    blocks = alloc.allocate(2)
    node, _ = rc.insert([1, 2, 3, 4, 5, 6, 7, 8], blocks)
    alloc.decref(blocks)  # the inserting request finished; the tree holds the only reference
    assert alloc.num_free == 2
    rc.inc_lock(node)
    assert rc.evict(2) == 0
    rc.dec_lock(node)
    assert rc.evict(2) == 2
    assert alloc.num_free == 4
    assert rc.match_prefix([1, 2, 3, 4])[0] == []


def test_evicts_least_recently_used_leaf_first():
    alloc, rc = _setup(num_blocks=8)
    a, b = alloc.allocate(1), alloc.allocate(1)
    rc.insert([1, 1, 1, 1], a)
    rc.insert([2, 2, 2, 2], b)
    alloc.decref(a + b)
    rc.match_prefix([1, 1, 1, 1, 0])  # touch a
    assert rc.evict(1) == 1
    assert rc.match_prefix([2, 2, 2, 2, 0])[0] == []  # b (older) was evicted
    assert rc.match_prefix([1, 1, 1, 1, 0])[0] == a


# ------------------------------------------------------------------ host tier (L2)
class FakeHost:
    """Stand-in for tandem.cache.host_pool.HostKVPool: pages from an allocator; copies are
    recorded and complete at once (no event)."""

    def __init__(self, pages: int) -> None:
        self.allocator = BlockAllocator(pages)
        self.copies = []

    def to_host(self, gpu, pages):
        self.copies.append(("to_host", list(gpu), list(pages)))

    def to_gpu(self, pages, gpu):
        self.copies.append(("to_gpu", list(pages), list(gpu)))


A, B = list(range(1, 9)), list(range(9, 17))  # two prefixes of 2 blocks each


def _tiered(num_blocks=8, pages=8, bs=4):
    alloc, host = BlockAllocator(num_blocks), FakeHost(pages)
    return alloc, host, RadixCache(bs, alloc, host)


def _cached(alloc, rc, tokens):
    """Insert ``tokens`` as a finished request would: the tree keeps the only reference."""
    blocks = alloc.allocate(len(tokens) // rc.bs)
    node, _ = rc.insert(tokens, blocks)
    alloc.decref(blocks)
    return node, blocks


def test_evicted_prefix_stays_on_host_and_loads_back():
    alloc, host, rc = _tiered()
    node, blocks = _cached(alloc, rc, A)
    assert rc.write_through(node) == 2 and host.copies[-1] == ("to_host", blocks, [0, 1])
    assert rc.evict(2) == 2 and alloc.num_free == 8  # the GPU copy goes, the host copy stays
    m = rc.match(A + [0])
    assert m.blocks == [] and [n.host_value for n in m.host_nodes] == [[0, 1]]
    loaded, event = rc.load(m.host_nodes)
    assert loaded == m.host_nodes and event is None and host.copies[-1][:2] == ("to_gpu", [0, 1])
    m2 = rc.match(A + [0])
    assert m2.blocks == host.copies[-1][2] and not m2.host_nodes and alloc.num_free == 6
    assert rc.num_hit_tokens == 8 and rc.num_host_hit_tokens == 8


def test_eviction_writes_back_a_node_without_host_copy():
    alloc, host, rc = _tiered()
    _, blocks = _cached(alloc, rc, A)
    assert rc.evict(2) == 2
    assert host.copies == [("to_host", blocks, [0, 1])]
    assert rc.match(A + [0]).host_nodes


def test_full_host_tier():
    alloc, host, rc = _tiered(pages=2)
    na, _ = _cached(alloc, rc, A)
    nb, _ = _cached(alloc, rc, B)
    rc.write_through(na)
    rc.match(B + [0])  # B is now the more recent one
    assert rc.write_through(nb) == 2  # fresh KV may push out the LRU host copy (A's; A stays in L1)
    assert na.host_value is None and na.value is not None and nb.host_value == [0, 1]
    # evicting A must not push B's copy out to write A back: A (LRU) is dropped, B moves to L2
    assert rc.evict(4) == 4 and alloc.num_free == 8
    assert rc.match(A + [0]).blocks == [] and not rc.match(A + [0]).host_nodes
    assert [n.host_value for n in rc.match(B + [0]).host_nodes] == [[0, 1]]


def test_host_eviction_lru_removes_l2_leaves_and_keeps_l1_copies():
    alloc, host, rc = _tiered(pages=4)
    na, _ = _cached(alloc, rc, A)
    nb, _ = _cached(alloc, rc, B)
    rc.write_through(na)
    rc.write_through(nb)
    rc.evict(2)  # A (older) moves to L2 only
    assert na.value is None and nb.value is not None
    rc.match(A + [0])  # A becomes the more recent one
    assert rc.evict_host(2) == 2  # B's host copy goes (B keeps its GPU copy)
    assert nb.host_value is None and nb.value is not None and na.host_value is not None
    assert rc.evict_host(2) == 2  # then A, an L2-only leaf, leaves the tree
    assert not rc.match(A + [0]).host_nodes and rc.num_host_pages == 0


def test_insert_adopts_an_l2_only_node():
    alloc, host, rc = _tiered()
    node, _ = _cached(alloc, rc, A)
    rc.write_through(node)
    rc.evict(2)
    fresh = alloc.allocate(2)  # a request recomputed A instead of loading it
    _, canonical = rc.insert(A, fresh)
    assert canonical == fresh and node.value == fresh and node.host_value == [0, 1]
    assert rc.match(A + [0]).blocks == fresh


def test_split_keeps_both_tiers_aligned():
    alloc, host, rc = _tiered()
    node, _ = _cached(alloc, rc, A)
    rc.write_through(node)
    rc.evict(2)
    m = rc.match(A[:4] + [99, 98, 97, 96, 0])  # diverges after the first block: split
    assert [n.host_value for n in m.host_nodes] == [[0]]
    m = rc.match(A + [0])
    assert [n.host_value for n in m.host_nodes] == [[0], [1]]
