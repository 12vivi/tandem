from __future__ import annotations

import pytest
import torch

from conftest import make_engine, prompts
from tandem.config import SpeculativeConfig
from tandem.layers.sampler import filtered_probs, verify_draft
from tandem.sampling_params import SamplingParams
from tandem.spec.ngram import NgramProposer


def test_ngram_proposes_continuation_of_latest_match():
    p = NgramProposer(n_min=2, n_max=3)
    assert p.propose([1, 2, 3, 4, 9, 1, 2, 3], k=3) == [4, 9, 1]
    assert p.propose([5, 6, 7, 8], k=3) == []


def test_ngram_spec_greedy_identical_to_plain_greedy(tiny_model):
    tm = tiny_model("qwen3")
    pattern = prompts(7, [6])[0]
    ps = [pattern * 4, prompts(8, [11])[0]]
    sp = SamplingParams(max_tokens=16, temperature=0.0, ignore_eos=True)
    plain = make_engine(tm.path).generate(ps, sp)
    spec = make_engine(tm.path, speculative=SpeculativeConfig(method="ngram", num_speculative_tokens=4)).generate(ps, sp)
    for a, b, p in zip(plain, spec, ps):
        assert a.output_token_ids == b.output_token_ids == tm.greedy(p, 16)


class _OracleProposer:
    """Proposes the true greedy continuation with the last draft corrupted, so every
    step exercises acceptance, rejection and KV rollback of the rejected position."""

    def __init__(self, tm):
        self.tm = tm

    def propose(self, token_ids, k):
        if k <= 0:
            return []
        drafts = self.tm.greedy(token_ids, k)
        if k >= 2:
            drafts[-1] = (drafts[-1] + 1) % 256
        return drafts


@pytest.mark.parametrize("arch", ["llama", "qwen3_moe"])
def test_verification_accepts_rejects_and_rolls_back(tiny_model, arch):
    tm = tiny_model(arch)
    eng = make_engine(tm.path, speculative=SpeculativeConfig(method="ngram", num_speculative_tokens=4))
    eng.proposer = _OracleProposer(tm)
    ps = prompts(11, [9, 14])
    outs = eng.generate(ps, SamplingParams(max_tokens=17, temperature=0.0, ignore_eos=True))
    for p, out in zip(ps, outs):
        assert out.output_token_ids == tm.greedy(p, 17)
        assert out.metrics["spec_acceptance"] == pytest.approx(0.75, abs=0.1)
    assert eng.num_steps <= 8  # 1 prefill step + ~4 tokens per decode step (17 steps without drafts)


def test_verify_draft_preserves_target_distribution():
    torch.manual_seed(0)
    logits = torch.tensor([[2.0, 1.0, 0.5, -1.0], [0.0, 0.0, 0.0, 0.0]])
    target = torch.softmax(logits[0], dim=-1)
    sp = SamplingParams(temperature=1.0)
    gen = torch.Generator().manual_seed(1)
    n = 20000
    counts = torch.zeros(4)
    for _ in range(n):
        counts[verify_draft(logits, [2], sp, gen)[0]] += 1
    assert torch.allclose(counts / n, target, atol=0.015)


def test_filters_top_k_top_p_min_p():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    p = filtered_probs(logits, [SamplingParams(top_k=2)])
    assert torch.allclose(p, torch.tensor([[0.625, 0.375, 0.0, 0.0]]))
    p = filtered_probs(logits, [SamplingParams(top_p=0.7)])
    assert torch.allclose(p, torch.tensor([[0.625, 0.375, 0.0, 0.0]]))
    p = filtered_probs(logits, [SamplingParams(min_p=0.5)])
    assert torch.allclose(p, torch.tensor([[0.625, 0.375, 0.0, 0.0]]))


def test_seeded_sampling_is_reproducible(tiny_model):
    tm = tiny_model("llama")
    sp = SamplingParams(max_tokens=12, temperature=0.8, top_p=0.9, seed=42, ignore_eos=True)
    p = prompts(9, [9])
    a = make_engine(tm.path).generate(p, sp)[0].output_token_ids
    b = make_engine(tm.path).generate(p, sp)[0].output_token_ids
    assert a == b
