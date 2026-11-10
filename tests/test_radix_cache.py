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
