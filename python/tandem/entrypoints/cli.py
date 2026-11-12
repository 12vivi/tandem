"""Command line: ``tandem serve | generate | plan | probe``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time


def _engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="local model directory (config.json + safetensors + tokenizer)")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--max-num-batched-tokens", type=int, default=2048)
    p.add_argument("--schedule-policy", choices=["prefill_first", "decode_first"], default="prefill_first")
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--kv-cache-gb", type=float, default=1.0)
    p.add_argument("--no-prefix-caching", action="store_true")
    p.add_argument("--cpu-threads", type=int, default=0)
    p.add_argument("--attention-backend", choices=["auto", "torch", "flashinfer"], default="auto",
                   help="auto: FlashInfer kernels on CUDA (bf16 / fp16) when installed, else the PyTorch reference")
    p.add_argument("--no-cuda-graph", action="store_true", help="run decode steps eagerly")
    p.add_argument("--cuda-graph-max-bs", type=int, default=64)
    p.add_argument("--no-overlap", action="store_true", help="wait for each step's tokens before scheduling the next")
    p.add_argument("--no-fused-prologue", action="store_true",
                   help="unfused QK-norm / RoPE / KV write instead of one Triton kernel (FlashInfer backend)")
    p.add_argument("--spec", choices=["none", "ngram"], default="none", help="speculative decoding method")
    p.add_argument("--num-spec-tokens", type=int, default=4)
    p.add_argument("--hybrid", action="store_true", help="three-path MoE expert execution")
    p.add_argument("--gpu-expert-slots", type=int, default=None)
    p.add_argument("--vram-budget-gb", type=float, default=None)
    p.add_argument("--hw-profile", default="probe", help="preset name, JSON file, or 'probe'")


def _make_engine(args):
    from tandem.engine.llm_engine import LLMEngine

    return LLMEngine(_engine_config(args))


def _engine_config(args):
    from tandem.config import EngineConfig, HybridConfig, SpeculativeConfig

    return EngineConfig(
        model=args.model,
        device=args.device,
        dtype=args.dtype,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        schedule_policy=args.schedule_policy,
        max_model_len=args.max_model_len,
        block_size=args.block_size,
        kv_cache_memory_gb=args.kv_cache_gb,
        enable_prefix_caching=not args.no_prefix_caching,
        cpu_threads=args.cpu_threads,
        attention_backend=args.attention_backend,
        cuda_graph=not args.no_cuda_graph,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
        overlap_schedule=not args.no_overlap,
        fused_prologue=not args.no_fused_prologue,
        speculative=SpeculativeConfig(method=args.spec, num_speculative_tokens=args.num_spec_tokens),
        hybrid=HybridConfig(enabled=args.hybrid, gpu_expert_slots=args.gpu_expert_slots,
                            vram_budget_gb=args.vram_budget_gb, hw_profile=args.hw_profile),
    )


def cmd_serve(args) -> None:
    import uvicorn

    from tandem.entrypoints.openai_server import build_app

    name = args.served_model_name or args.model.rstrip("/\\").replace("\\", "/").split("/")[-1]
    if args.no_engine_process:  # engine thread inside the API process
        from tandem.engine.async_engine import AsyncEngine

        engine = _make_engine(args)
        async_engine, tokenizer = AsyncEngine(engine), engine.tokenizer
    else:  # engine core in its own process (default)
        from transformers import AutoTokenizer

        from tandem.engine.mp_engine import MPAsyncEngine

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        async_engine = MPAsyncEngine(_engine_config(args), tokenizer)
    app = build_app(async_engine, tokenizer, name, args.default_max_tokens)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def cmd_generate(args) -> None:
    from tandem.sampling_params import SamplingParams

    engine = _make_engine(args)
    prompt = args.prompt
    if args.chat:
        prompt = engine.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p, seed=args.seed)
    t0 = time.perf_counter()
    out = engine.generate([prompt], sp)[0]
    dt = time.perf_counter() - t0
    print(out.text)
    n = len(out.output_token_ids)
    print(f"\n[{n} tokens in {dt:.2f}s, {n / dt:.2f} tok/s, ttft {out.metrics.get('ttft', 0):.3f}s, finish={out.finish_reason}]", file=sys.stderr)
    print(json.dumps(engine.stats(), ensure_ascii=False), file=sys.stderr)


def cmd_plan(args) -> None:
    from tandem.model_config import ModelConfig
    from tandem.planner.hardware import load_profile
    from tandem.planner.model_presets import MODEL_PRESETS
    from tandem.planner.placement import make_plan

    if args.model in MODEL_PRESETS:
        cfg = ModelConfig.from_dict(MODEL_PRESETS[args.model])
    else:
        cfg = ModelConfig.from_pretrained(args.model)
    hw = load_profile(args.hw_profile)
    plan = make_plan(cfg, hw, bytes_per_param=args.weight_bits / 8, vram_budget_gb=args.vram_budget_gb, kv_gb=args.kv_cache_gb)
    print(f"model: {args.model} ({cfg.architecture})   hardware: {hw.name}   weights: {args.weight_bits}-bit")
    print(plan.explain())


def cmd_probe(args) -> None:
    from tandem.planner.hardware import probe

    print(probe(quick=not args.full).to_json())


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="tandem", description="single-GPU + CPU heterogeneous LLM inference engine")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve", help="OpenAI-compatible HTTP server")
    _engine_args(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--served-model-name", default=None)
    p.add_argument("--default-max-tokens", type=int, default=256)
    p.add_argument("--no-engine-process", action="store_true",
                   help="run the engine in a thread of the API process instead of its own process")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("generate", help="one-off generation")
    _engine_args(p)
    p.add_argument("--prompt", required=True)
    p.add_argument("--chat", action="store_true", help="wrap the prompt with the chat template")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(fn=cmd_generate)

    p = sub.add_parser("plan", help="placement plan for a model on a hardware profile")
    p.add_argument("--model", required=True, help="model directory or preset (Qwen3-8B, Qwen3-30B-A3B, Qwen3-235B-A22B, Llama-3.3-70B)")
    p.add_argument("--hw-profile", default="probe")
    p.add_argument("--weight-bits", type=float, default=16)
    p.add_argument("--vram-budget-gb", type=float, default=None)
    p.add_argument("--kv-cache-gb", type=float, default=2.0)
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("probe", help="measure this machine's hardware profile")
    p.add_argument("--full", action="store_true")
    p.set_defaults(fn=cmd_probe)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
