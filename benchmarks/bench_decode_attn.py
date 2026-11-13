"""Microbenchmark: FlashInfer paged decode attention at small batch (design doc P2, A1).

At batch 1-8 and 0.5-1K context, a Qwen3-0.6B / 1.7B decode step spends about 12 us per layer
in attention (P1 profile: decode kernel + split-KV merge), for only ~3-4 MB of KV. This compares
the ways FlashInfer can run it, each in CUDA-graph mode as in tandem's decode graphs:
  cuda-core           BatchDecode, split-KV chosen by FlashInfer (tandem's current path, GQA < 4)
  cuda-core-nosplit   same, disable_split_kv=True
  tensor-core         BatchDecode with use_tensor_cores=True (prefill kernel), default split
  tensor-core-sNNN    tensor-core with fixed_split_size=NNN tokens
Each graph runs the 28 layers' attention back to back over distinct KV caches (no L2 reuse).

  python benchmarks/bench_decode_attn.py --save benchmarks/results/p2-attn.jsonl
"""

from __future__ import annotations

import argparse
import json

import flashinfer
import torch


def run_case(nqo: int, nkv: int, hd: int, bs: int, ctx: int, variant: str, layers: int = 28, page: int = 16) -> float:
    dev = torch.device("cuda")
    dt = torch.bfloat16
    pages_per = (ctx + page - 1) // page
    num_pages = bs * pages_per
    caches = [(torch.randn(num_pages, page, nkv, hd, device=dev, dtype=dt),
               torch.randn(num_pages, page, nkv, hd, device=dev, dtype=dt)) for _ in range(layers)]
    q = torch.randn(bs, nqo, hd, device=dev, dtype=dt)
    indptr = torch.arange(0, bs + 1, dtype=torch.int32) * pages_per
    indices = torch.arange(num_pages, dtype=torch.int32)
    last = torch.full((bs,), ctx - (pages_per - 1) * page, dtype=torch.int32)
    ws = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    tc = variant.startswith("tensor-core")
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        ws, "NHD", use_cuda_graph=True, use_tensor_cores=tc,
        paged_kv_indptr_buffer=torch.zeros(bs + 1, dtype=torch.int32, device=dev),
        paged_kv_indices_buffer=torch.zeros(num_pages, dtype=torch.int32, device=dev),
        paged_kv_last_page_len_buffer=torch.zeros(bs, dtype=torch.int32, device=dev),
    )
    kw = {}
    if variant.endswith("nosplit"):
        kw["disable_split_kv"] = True
    if variant.startswith("tensor-core-s"):
        kw["fixed_split_size"] = int(variant.rsplit("s", 1)[1])
    wrapper.plan(indptr, indices.to(dev), last, nqo, nkv, hd, page, q_data_type=dt, kv_data_type=dt,
                 sm_scale=hd ** -0.5, **kw)
    outs = [torch.empty_like(q) for _ in range(layers)]

    def step() -> None:
        for (k, v), o in zip(caches, outs):
            wrapper.run(q, (k, v), out=o)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    graph.replay()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    reps = 50
    t0.record()
    for _ in range(reps):
        graph.replay()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1e3 / (reps * layers)  # us per layer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default="16,8,128", help="q heads, kv heads, head dim (Qwen3-0.6B / 1.7B)")
    ap.add_argument("--batches", default="1,4,8")
    ap.add_argument("--contexts", default="512,768,1024")
    ap.add_argument("--variants", default="cuda-core,cuda-core-nosplit,tensor-core,tensor-core-s64,"
                                          "tensor-core-s128,tensor-core-s256,tensor-core-s512")
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    nqo, nkv, hd = (int(x) for x in a.heads.split(","))
    for bs in (int(x) for x in a.batches.split(",")):
        for ctx in (int(x) for x in a.contexts.split(",")):
            kv_mb = bs * ctx * nkv * hd * 2 * 2 / 1e6
            rec = {"heads": [nqo, nkv, hd], "batch": bs, "context": ctx, "kv_mb_per_layer": round(kv_mb, 2)}
            for v in a.variants.split(","):
                try:
                    us = run_case(nqo, nkv, hd, bs, ctx, v)
                    rec[v] = round(us, 2)
                except Exception as exc:  # noqa: BLE001 - some variants are unsupported for some shapes
                    rec[v] = f"error: {str(exc).splitlines()[0][:80]}"
                torch.cuda.empty_cache()
            best = min((k for k in rec if isinstance(rec[k], float) and k not in ("kv_mb_per_layer",)), key=rec.get)
            rec["best"] = best
            rec["best_gbps"] = round(kv_mb / rec[best] * 1e3, 1)  # MB per us -> GB/s
            print(json.dumps(rec), flush=True)
            if a.save:
                with open(a.save, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
