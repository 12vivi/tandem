"""Optional check against a real checkpoint (skipped unless TANDEM_TEST_MODEL is set).

    TANDEM_TEST_MODEL=models/Qwen3-0.6B pytest tests/test_real_model.py -q
"""

from __future__ import annotations

import os

import pytest
import torch

MODEL = os.environ.get("TANDEM_TEST_MODEL")
pytestmark = pytest.mark.skipif(not MODEL, reason="set TANDEM_TEST_MODEL to a local HF model directory")

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "请用一句话介绍你自己。",
]


def test_greedy_matches_transformers():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tandem.config import EngineConfig
    from tandem.engine.llm_engine import LLMEngine
    from tandem.sampling_params import SamplingParams

    n = 24
    tok = AutoTokenizer.from_pretrained(MODEL)
    eng = LLMEngine(EngineConfig(model=MODEL, device="cpu", dtype="float32", kv_cache_memory_gb=0.5), tokenizer=tok)
    outs = eng.generate(PROMPTS, SamplingParams(max_tokens=n, temperature=0.0, ignore_eos=True))

    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    for prompt, out in zip(PROMPTS, outs):
        cur = tok(prompt, return_tensors="pt").input_ids
        ref, past = [], None
        with torch.no_grad():  # plain greedy loop with HF's own KV cache, no EOS handling
            for _ in range(n):
                o = hf(cur, past_key_values=past, use_cache=True)
                past = o.past_key_values
                ref.append(int(o.logits[0, -1].argmax()))
                cur = torch.tensor([[ref[-1]]])
        assert out.output_token_ids == ref, (prompt, tok.decode(out.output_token_ids), tok.decode(ref))
