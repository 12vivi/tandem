"""Shared fixtures: tiny random-weight HF models saved as real checkpoints.

Every architecture is built with Hugging Face Transformers, saved with ``save_pretrained``
and loaded by tandem through the normal safetensors path, so the tests cover config
parsing, weight loading and numerics against the reference implementation.
"""

from __future__ import annotations

import pytest
import torch

ARCHS = ["llama", "qwen2", "qwen3", "qwen3_moe"]

_COMMON = dict(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=512,
    rms_norm_eps=1e-6,
)


def _hf_config(arch: str):
    from transformers import (
        LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM,
        Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM,
    )

    if arch == "llama":
        scaling = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                   "high_freq_factor": 4.0, "original_max_position_embeddings": 64}
        return LlamaConfig(**_COMMON, rope_theta=10000.0, rope_scaling=scaling, tie_word_embeddings=False), LlamaForCausalLM
    if arch == "qwen2":
        return Qwen2Config(**_COMMON, tie_word_embeddings=True), Qwen2ForCausalLM
    if arch == "qwen3":
        return Qwen3Config(**_COMMON, head_dim=16, tie_word_embeddings=True), Qwen3ForCausalLM
    if arch == "qwen3_moe":
        return Qwen3MoeConfig(**_COMMON, head_dim=16, moe_intermediate_size=32, num_experts=8,
                              num_experts_per_tok=2, norm_topk_prob=True, decoder_sparse_step=1,
                              mlp_only_layers=[]), Qwen3MoeForCausalLM
    raise ValueError(arch)


class TinyModel:
    def __init__(self, path: str, hf_model) -> None:
        self.path = path
        self.hf = hf_model  # float64 reference

    @torch.no_grad()
    def logits(self, token_ids: list[int]) -> torch.Tensor:
        return self.hf(torch.tensor([token_ids])).logits[0]

    @torch.no_grad()
    def greedy(self, prompt: list[int], n: int) -> list[int]:
        ids = list(prompt)
        for _ in range(n):
            ids.append(int(self.logits(ids)[-1].argmax()))
        return ids[len(prompt):]


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    cache: dict[str, TinyModel] = {}

    def make(arch: str) -> TinyModel:
        if arch not in cache:
            torch.manual_seed(1234)
            cfg, cls = _hf_config(arch)
            model = cls(cfg)
            with torch.no_grad():  # well-separated logits: no greedy near-ties
                for name, p in model.named_parameters():
                    if name.endswith("norm.weight"):
                        p.copy_(1.0 + 0.1 * torch.randn_like(p))
                    else:
                        p.normal_(0.0, 0.1)
            path = tmp_path_factory.mktemp(arch)
            model.save_pretrained(str(path))
            ref = model.to(torch.float64).eval()
            if hasattr(ref, "set_experts_implementation") and ref._can_set_experts_implementation():
                ref.set_experts_implementation("eager")  # torch._grouped_mm has no float64 kernel
            cache[arch] = TinyModel(str(path), ref)
        return cache[arch]

    return make


def make_engine(path: str, **overrides):
    from tandem.config import EngineConfig
    from tandem.engine.llm_engine import LLMEngine

    kw = dict(model=path, device="cpu", dtype="float64", block_size=4, num_kv_blocks=256,
              max_num_batched_tokens=64, max_model_len=256)
    tokenizer = overrides.pop("tokenizer", None)
    kw.update(overrides)
    return LLMEngine(EngineConfig(**kw), tokenizer=tokenizer, load_tokenizer=False)


def prompts(seed: int, lengths: list[int], vocab: int = 256) -> list[list[int]]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(1, vocab, (n,), generator=g).tolist() for n in lengths]
