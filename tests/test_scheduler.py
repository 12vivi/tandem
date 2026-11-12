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
