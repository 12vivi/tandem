"""Host-memory KV tier (L2) end to end on the CPU engine: prefixes evicted from the GPU pool
stay in host memory, are loaded back on a hit, and the outputs do not change.

The CPU KV cache makes every host copy synchronous (tandem.cache.host_pool falls back to
per-layer tensor copies), so these tests cover the tree, scheduler and copy layout; the
asynchronous CUDA path is in test_gpu_fastpath.py.
"""

from __future__ import annotations

import torch

from conftest import make_engine, prompts
from tandem.sampling_params import SamplingParams

GREEDY = SamplingParams(max_tokens=6, temperature=0.0, ignore_eos=True)


def _run_one_by_one(eng, ps):
    return [eng.generate([p], GREEDY)[0] for p in ps]


def test_prefix_evicted_to_host_is_reused(tiny_model):
    tm = tiny_model("qwen3")
    shared = prompts(1, [40])[0]
    others = prompts(2, [30] * 5)  # 5 x (30 + 6) tokens: more than the 24-block (96-token) GPU pool
    ps = [shared + [7], *others, shared + [9, 9]]
    ref = _run_one_by_one(make_engine(tm.path, enable_prefix_caching=False), ps)

    eng = make_engine(tm.path, num_kv_blocks=24, host_kv_gb=0.002)  # ~950 host pages of 4 tokens
    assert eng.host_pool is not None and eng.host_pool.num_pages > 100
    out = _run_one_by_one(eng, ps)
    for a, b in zip(ref, out):
        assert a.output_token_ids == b.output_token_ids
    # the shared 40-token prefix (10 blocks) was pushed out of the GPU pool and came back from host memory
    assert out[-1].metrics["cached_tokens"] == 40
    assert eng.radix.num_host_hit_tokens >= 40
    assert eng.radix.num_host_pages > 0

    # without the host tier the same sequence recomputes the prefix
    gpu_only = make_engine(tm.path, num_kv_blocks=24)
    assert _run_one_by_one(gpu_only, ps)[-1].metrics["cached_tokens"] < 40


def test_preemption_with_host_tier_keeps_outputs(tiny_model):
    """Concurrent requests overflow a tiny GPU pool: preempted requests resume from host copies."""
    tm = tiny_model("qwen3")
    ps = prompts(3, [25, 31, 18, 40, 22, 35])
    sp = SamplingParams(max_tokens=12, temperature=0.0, ignore_eos=True)
    ref = make_engine(tm.path, enable_prefix_caching=False).generate(ps, sp)
    eng = make_engine(tm.path, num_kv_blocks=20, host_kv_gb=0.002)
    out = eng.generate(ps, sp)
    assert eng.scheduler.num_preemptions > 0
    for a, b in zip(ref, out):
        assert a.output_token_ids == b.output_token_ids
    assert eng.allocator.num_free + eng.radix.num_cached_blocks == eng.allocator.num_blocks
    assert not eng.scheduler.loading


def test_host_pool_page_layout_round_trip(tiny_model):
    """to_host then to_gpu into other blocks reproduces every layer's K and V."""
    tm = tiny_model("qwen3")
    eng = make_engine(tm.path, num_kv_blocks=16, host_kv_gb=0.001)
    kv, pool = eng.kv_cache, eng.host_pool
    torch.manual_seed(0)
    for t in kv.k + kv.v:
        t.normal_()
    before = [t.clone() for t in kv.k + kv.v]
    pool.to_host([1, 2, 5], [7, 0, 3])
    pool.to_gpu([7, 0, 3], [10, 11, 12])
    bs = kv.block_size
    for t, b in zip(kv.k + kv.v, before):
        for src, dst in ((1, 10), (2, 11), (5, 12)):
            assert torch.equal(t[dst * bs : (dst + 1) * bs], b[src * bs : (src + 1) * bs])
