"""Numerics vs Hugging Face Transformers for every supported architecture."""

from __future__ import annotations

import pytest
import torch

from conftest import ARCHS, make_engine, prompts
from tandem.executor.forward_context import ForwardContext, build_attention_metadata
from tandem.layers.attention import TorchAttentionBackend
from tandem.sampling_params import SamplingParams
from tandem.utils import cdiv

GREEDY = dict(temperature=0.0, ignore_eos=True)


@pytest.mark.parametrize("arch", ARCHS)
def test_prefill_logits_match_hf(tiny_model, arch):
    tm = tiny_model(arch)
    eng = make_engine(tm.path)
    prompt = prompts(0, [23])[0]
    blocks = eng.allocator.allocate(cdiv(len(prompt), eng.config.block_size))
    positions, meta = build_attention_metadata([(blocks, 0, len(prompt))], eng.config.block_size, eng.device)
    ctx = ForwardContext(meta, eng.kv_cache, TorchAttentionBackend(), eng.moe_executor)
    with torch.inference_mode():
        ours = eng.model.compute_logits(eng.model(torch.tensor(prompt), positions, ctx))
    ref = tm.logits(prompt)
    torch.testing.assert_close(ours.double(), ref, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("arch", ARCHS)
def test_batched_chunked_greedy_matches_hf(tiny_model, arch):
    """Several requests of different lengths, small token budget (chunked prefill), paged KV."""
    tm = tiny_model(arch)
    eng = make_engine(tm.path, max_num_batched_tokens=8)
    ps = prompts(1, [5, 13, 21])
    outs = eng.generate(ps, SamplingParams(max_tokens=10, **GREEDY))
    for p, out in zip(ps, outs):
        assert out.output_token_ids == tm.greedy(p, 10)
        assert out.finish_reason == "length"


def test_prefix_cache_hit_and_correctness(tiny_model):
    tm = tiny_model("qwen3")
    eng = make_engine(tm.path)
    base = prompts(2, [20])[0]
    eng.generate([base], SamplingParams(max_tokens=4, **GREEDY))
    follow = base + prompts(3, [6])[0]
    out = eng.generate([follow], SamplingParams(max_tokens=6, **GREEDY))[0]
    assert out.metrics["cached_tokens"] >= 16  # whole blocks of the shared 20-token prefix
    assert out.output_token_ids == tm.greedy(follow, 6)
    assert eng.radix.hit_rate > 0


def test_preemption_recovers_exact_outputs(tiny_model):
    tm = tiny_model("llama")
    # 12 blocks x 4 tokens cannot hold four requests of 10 + 14 tokens at once
    eng = make_engine(tm.path, num_kv_blocks=12, max_num_batched_tokens=16)
    ps = prompts(4, [10, 10, 10, 10])
    outs = eng.generate(ps, SamplingParams(max_tokens=14, **GREEDY))
    assert eng.scheduler.num_preemptions > 0
    for p, out in zip(ps, outs):
        assert out.output_token_ids == tm.greedy(p, 14)
    assert eng.allocator.num_free + eng.radix.num_cached_blocks == eng.allocator.num_blocks


def test_kv_blocks_fully_released(tiny_model):
    tm = tiny_model("qwen2")
    eng = make_engine(tm.path, enable_prefix_caching=False)
    eng.generate(prompts(5, [7, 9, 30]), SamplingParams(max_tokens=5, **GREEDY))
    assert eng.allocator.num_free == eng.allocator.num_blocks
