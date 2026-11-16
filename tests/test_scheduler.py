"""Scheduler unit tests without a model: fake sampled tokens drive the state machine."""

from __future__ import annotations

from tandem.cache.block_allocator import BlockAllocator
from tandem.engine.request import Request
from tandem.sampling_params import SamplingParams
from tandem.scheduler.scheduler import Scheduler


def _sched(num_blocks=64, budget=8, max_seqs=8, policy="prefill_first"):
    return Scheduler(BlockAllocator(num_blocks), block_size=4, max_num_seqs=max_seqs,
                     max_num_batched_tokens=budget, max_model_len=1024, policy=policy)


def _req(rid, n_prompt, max_tokens=4):
    return Request(rid, list(range(1, n_prompt + 1)), SamplingParams(max_tokens=max_tokens, ignore_eos=True))


def _step(s: Scheduler):
    out = s.schedule()
    sampled = {}
    for sr in out.scheduled:
        r = sr.request
        if r.num_computed_tokens + sr.num_tokens >= r.num_tokens_with_spec:
            sampled[r.request_id] = [7]
    s.update_from_output(out, sampled)
    return out


def test_chunked_prefill_respects_budget():
    s = _sched(budget=8)
    r = _req("a", 20)
    s.add_request(r)
    sizes = [_step(s).scheduled[0].num_tokens for _ in range(3)]
    assert sizes == [8, 8, 4]
    assert r.num_output_tokens == 1 and r.is_decoding


def test_decode_first_mixes_decodes_and_new_prefill():
    s = _sched(budget=8, policy="decode_first")
    a = _req("a", 4, max_tokens=10)
    s.add_request(a)
    _step(s)  # a: prefill done, now decoding
    s.add_request(_req("b", 30))
    out = _step(s)
    assert [x.request.request_id for x in out.scheduled] == ["a", "b"]
    assert out.scheduled[0].num_tokens == 1 and out.scheduled[1].num_tokens == 7


def test_prefill_first_prefills_new_request_alone():
    s = _sched(budget=8)  # default policy
    a = _req("a", 4, max_tokens=10)
    s.add_request(a)
    _step(s)  # a: prefill done, now decoding
    s.add_request(_req("b", 12))
    assert [(x.request.request_id, x.num_tokens) for x in _step(s).scheduled] == [("b", 8)]  # no decode for a
    assert [(x.request.request_id, x.num_tokens) for x in _step(s).scheduled] == [("b", 4)]  # b still prefilling
    assert sorted(x.request.request_id for x in _step(s).scheduled) == ["a", "b"]  # both decode


def test_preemption_and_completion_free_all_blocks():
    s = _sched(num_blocks=6, budget=64)
    reqs = [_req(str(i), 8, max_tokens=6) for i in range(3)]
    for r in reqs:
        s.add_request(r)
    for _ in range(100):
        if not s.has_unfinished():
            break
        _step(s)
    assert all(r.finished and r.num_output_tokens == 6 for r in reqs)
    assert s.num_preemptions > 0
    assert s.allocator.num_free == 6


def test_max_tokens_and_eos_stop():
    s = _sched()
    r = Request("x", [1, 2, 3], SamplingParams(max_tokens=50), eos_token_ids=[7])
    s.add_request(r)
    _step(s)
    assert r.finished and r.finish_reason == "stop" and r.output_token_ids == [7]


def _cached_scheduler(budget=8):
    """Scheduler with a radix cache in which tokens 1..24 (6 blocks of 4) are cached."""
    from tandem.cache.radix_cache import RadixCache

    alloc = BlockAllocator(64)
    s = Scheduler(alloc, block_size=4, max_num_seqs=8, max_num_batched_tokens=budget, max_model_len=1024,
                  radix_cache=RadixCache(4, alloc))
    s.add_request(_req("warm", 24, max_tokens=1))
    while s.has_unfinished():
        _step(s)
    return s


def test_short_prefill_goes_before_running_long_prefill():
    """A request whose prompt is cached except for 2 tokens gets a step of its own ahead of the
    next chunk of a long uncached prompt, which continues in the following step."""
    s = _cached_scheduler()
    s.add_request(Request("long", list(range(100, 140)), SamplingParams(max_tokens=2, ignore_eos=True)))
    assert [(x.request.request_id, x.num_tokens) for x in _step(s).scheduled] == [("long", 8)]
    short = Request("short", list(range(1, 25)) + [50, 51], SamplingParams(max_tokens=2, ignore_eos=True))
    s.add_request(short)
    assert [(x.request.request_id, x.num_tokens) for x in _step(s).scheduled] == [("short", 2)]
    assert short.num_cached_tokens == 24
    assert [(x.request.request_id, x.num_tokens) for x in _step(s).scheduled] == [("long", 8)]


def test_short_prefills_shortest_first_up_to_the_step_cap():
    s = _cached_scheduler(budget=64)
    a = Request("a", list(range(1, 25)) + list(range(60, 70)), SamplingParams(max_tokens=2, ignore_eos=True))
    b = Request("b", list(range(1, 25)) + [80, 81, 82], SamplingParams(max_tokens=2, ignore_eos=True))
    s.add_request(a)
    s.add_request(b)
    s.SHORT_STEP_TOKENS = 3  # b (3 tokens left) fills the step
    assert [(x.request.request_id, x.num_tokens) for x in _step(s).scheduled] == [("b", 3)]
    assert [(x.request.request_id, x.num_tokens) for x in _step(s).scheduled] == [("a", 10)]


def test_short_request_preempts_long_prefill_when_it_does_not_fit():
    """GPU blocks are taken by a long prompt being prefilled; a request whose prompt is cached
    (partly in host memory) except for 2 tokens needs 3 blocks: the long prefill is preempted
    (swapped out to host memory) and the short request is admitted in the same step; both finish."""
    from tandem.cache.radix_cache import RadixCache
    from test_radix_cache import FakeHost

    alloc = BlockAllocator(12)
    s = Scheduler(alloc, block_size=4, max_num_seqs=8, max_num_batched_tokens=8, max_model_len=1024,
                  radix_cache=RadixCache(4, alloc, FakeHost(32)))
    s.SHORT_PREFILL = 4
    s.add_request(_req("warm", 24, max_tokens=1))  # tokens 1..24 cached, written to host memory
    while s.has_unfinished():
        _step(s)
    long = Request("long", list(range(100, 140)), SamplingParams(max_tokens=2, ignore_eos=True))
    s.add_request(long)
    for _ in range(4):  # 32 of 40 tokens: 8 blocks; part of warm's GPU copies evicted, kept in host memory
        _step(s)
    assert long.num_computed_tokens == 32 and alloc.num_free == 0
    short = Request("short", list(range(1, 25)) + [50, 51], SamplingParams(max_tokens=2, ignore_eos=True))
    s.add_request(short)
    out = _step(s)
    assert [(x.request.request_id, x.num_tokens) for x in out.scheduled] == [("short", 2)]
    assert out.preempted == [long] and short.num_cached_tokens == 24
    while s.has_unfinished():
        _step(s)
    assert long.num_output_tokens == 2 and short.num_output_tokens == 2
    assert alloc.num_free + s.radix.num_cached_blocks == alloc.num_blocks
