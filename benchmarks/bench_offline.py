"""Offline throughput benchmark: synthetic token prompts straight into LLMEngine.

Example:
    python benchmarks/bench_offline.py --model models/Qwen3-0.6B --num-prompts 16 \
        --input-len 128 --output-len 64

For serving-level comparisons against vLLM / SGLang use `tandem serve` plus a common load
generator (see benchmarks/README.md); this script measures the engine alone.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch

from tandem.config import EngineConfig, HybridConfig, SpeculativeConfig
from tandem.engine.llm_engine import LLMEngine
from tandem.sampling_params import SamplingParams


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-prompts", type=int, default=16)
    ap.add_argument("--input-len", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048)
    ap.add_argument("--kv-cache-gb", type=float, default=1.0)
    ap.add_argument("--cpu-threads", type=int, default=0)
    ap.add_argument("--spec", choices=["none", "ngram"], default="none")
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--gpu-expert-slots", type=int, default=None)
    ap.add_argument("--hw-profile", default="probe")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None, help="append the result as a JSON line to this file")
    args = ap.parse_args()

    cfg = EngineConfig(
        model=args.model, device=args.device, dtype=args.dtype, max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens, kv_cache_memory_gb=args.kv_cache_gb,
        cpu_threads=args.cpu_threads, seed=args.seed,
        speculative=SpeculativeConfig(method=args.spec),
        hybrid=HybridConfig(enabled=args.hybrid, gpu_expert_slots=args.gpu_expert_slots, hw_profile=args.hw_profile),
    )
    engine = LLMEngine(cfg, load_tokenizer=False)
    vocab = min(engine.model_config.vocab_size, 100_000)  # stay clear of special tokens
    g = torch.Generator().manual_seed(args.seed)
    prompts = [torch.randint(10, vocab, (args.input_len,), generator=g).tolist() for _ in range(args.num_prompts)]
    sp = SamplingParams(max_tokens=args.output_len, temperature=0.0, ignore_eos=True)

    engine.generate([prompts[0][:16]], SamplingParams(max_tokens=4, ignore_eos=True))  # warm-up
    t0 = time.perf_counter()
    outs = engine.generate(prompts, sp)
    elapsed = time.perf_counter() - t0

    n_out = sum(len(o.output_token_ids) for o in outs)
    ttft = [o.metrics["ttft"] for o in outs]
    tpot = [(o.metrics["latency"] - o.metrics["ttft"]) / max(1, len(o.output_token_ids) - 1) for o in outs]
    result = {
        "model": os.path.basename(args.model.rstrip("/\\")),
        "device": str(engine.device),
        "dtype": str(engine.dtype).replace("torch.", ""),
        "num_prompts": args.num_prompts,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "elapsed_s": round(elapsed, 3),
        "output_tok_per_s": round(n_out / elapsed, 2),
        "total_tok_per_s": round((n_out + args.num_prompts * args.input_len) / elapsed, 2),
        "mean_ttft_ms": round(statistics.mean(ttft) * 1e3, 1),
        "mean_tpot_ms": round(statistics.mean(tpot) * 1e3, 1),
        "engine": engine.stats(),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
