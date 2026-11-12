"""Overlap scheduling: step N is launched before step N-1's tokens reach the host.

Outputs must equal the synchronous engine's, including requests that end while their next
step is already in flight (stop tokens, length, aborts), and every KV block must come back.
"""

from __future__ import annotations

from conftest import make_engine, prompts
from tandem.sampling_params import SamplingParams


def _blocks_accounted(eng) -> bool:
    cached = eng.radix.num_cached_blocks if eng.radix is not None else 0
    return eng.allocator.num_free + cached == eng.allocator.num_blocks


def test_stop_tokens_and_lengths_match_sync(tiny_model):
    tm = tiny_model("qwen3")
    ps = prompts(12, [5, 9, 30, 17, 3])
    greedy = [tm.greedy(p, 14) for p in ps]
    # stop at the first occurrence of the 5th greedy token; the last request runs to max_tokens
    params = [SamplingParams(max_tokens=14, temperature=0.0, ignore_eos=True, stop_token_ids=[g[4]]) for g in greedy[:-1]]
    params.append(SamplingParams(max_tokens=14, temperature=0.0, ignore_eos=True))
    sync = make_engine(tm.path, overlap_schedule=False).generate(ps, params)
    eng = make_engine(tm.path, max_num_batched_tokens=16)
    ovl = eng.generate(ps, params)
    for i, (s, o, g) in enumerate(zip(sync, ovl, greedy)):
        expect = g if i == len(ps) - 1 else g[: g.index(g[4]) + 1]
        assert o.output_token_ids == s.output_token_ids == expect
        assert o.finish_reason == s.finish_reason == ("length" if i == len(ps) - 1 else "stop")
    assert not eng.has_unfinished_requests() and eng.inflight is None
    assert _blocks_accounted(eng)


def test_abort_while_in_flight(tiny_model):
    tm = tiny_model("llama")
    eng = make_engine(tm.path)
    ps = prompts(13, [7, 11, 6])
    sp = SamplingParams(max_tokens=20, temperature=0.0, ignore_eos=True)
    ids = [eng.add_request(p, sp) for p in ps]
    finished = {}
    for _ in range(4):
        eng.step()
    eng.abort_request(ids[1])  # its next token may already be on the way
    after_abort = []
    while eng.has_unfinished_requests():
        for out in eng.step():
            if out.request_id == ids[1]:
                after_abort.append(out)
            if out.finished:
                finished[out.request_id] = out
    assert after_abort == []  # the in-flight token of an aborted request is dropped silently
    for i in (0, 2):
        assert finished[ids[i]].output_token_ids == tm.greedy(ps[i], 20)
    assert _blocks_accounted(eng)


def test_preemption_with_tokens_in_flight(tiny_model):
    tm = tiny_model("qwen2")
    # 12 blocks x 4 tokens cannot hold four requests of 10 + 14 tokens at once
    eng = make_engine(tm.path, num_kv_blocks=12, max_num_batched_tokens=16)
    ps = prompts(14, [10, 10, 10, 10])
    outs = eng.generate(ps, SamplingParams(max_tokens=14, temperature=0.0, ignore_eos=True))
    assert eng.scheduler.num_preemptions > 0
    for p, out in zip(ps, outs):
        assert out.output_token_ids == tm.greedy(p, 14)
    assert _blocks_accounted(eng)


def test_prefill_first_policy(tiny_model):
    """SGLang-style scheduling: steps that prefill carry no decodes; outputs unchanged."""
    tm = tiny_model("qwen3")
    eng = make_engine(tm.path, max_num_batched_tokens=16, schedule_policy="prefill_first")
    ps = prompts(15, [9, 30, 5, 21])
    sp = SamplingParams(max_tokens=10, temperature=0.0, ignore_eos=True)
    ids = [eng.add_request(p, sp) for p in ps[:2]]
    finished = {}
    late = False
    while eng.has_unfinished_requests():
        if not late and eng.num_steps == 3:  # two more requests arrive while the first two run
            ids += [eng.add_request(p, sp) for p in ps[2:]]
            late = True
        sched_waiting = bool(eng.scheduler.waiting)
        for out in eng.step():
            if out.finished:
                finished[out.request_id] = out
        if sched_waiting and eng.inflight is not None:
            rows = eng.inflight.sched.scheduled
            assert all(s.request.num_computed_tokens < s.request.num_prompt_tokens or s.num_tokens > 1
                       or s.request.num_output_tokens <= 1 for s in rows)
    for rid, p in zip(ids, ps):
        assert finished[rid].output_token_ids == tm.greedy(p, 10)
    assert _blocks_accounted(eng)
