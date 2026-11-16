"""Microbenchmark: KV page copies between the GPU cache and the pinned host pool (P3, L2).

Qwen3-8B shapes: 36 layers, 8 KV heads x 128, BF16, 16-token pages (2.36 MB per page with
every layer). Measures HostKVPool's two paths for batches of scattered pages:
  * loads (to_gpu): the page-copy kernel, one launch per layer, for each --load-ctas;
  * writes (to_host): gather into the GPU staging buffer + copy-engine copies;
the bandwidth of each, against one cudaMemcpyAsync of the same number of contiguous bytes (the
PCIe ceiling; P0 measured 12.9 / 13.2 GB/s), and the interference: how much slower a stream of
BF16 GEMMs (4096^3) on the compute stream runs while a 1024-page copy runs beside it (each
against a GEMM-only run measured right before, as the GPU clock drifts).

  python benchmarks/bench_kv_io.py --save benchmarks/results/p3-kv-io.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import time

import torch

from tandem.cache.host_pool import HostKVPool
from tandem.cache.kv_cache import KVCache


def _time(fn, reps: int = 5) -> float:
    fn()  # warm-up
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def _gemm_ms(a: torch.Tensor, b: torch.Tensor, n: int, copy=None) -> float:
    """Milliseconds for n GEMMs on the compute stream, with ``copy()`` started just before."""
    torch.cuda.synchronize()
    if copy is not None:
        copy()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(n):
        torch.mm(a, b)
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1)


def _slowdown(ga, gb, copy) -> float:
    alone = min(_gemm_ms(ga, gb, 20) for _ in range(2))
    beside = _gemm_ms(ga, gb, 20, copy)
    return round(beside / alone - 1, 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=1024)
    ap.add_argument("--batches", default="16,256,1024")
    ap.add_argument("--load-ctas", default="2,4,8,16,32")
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    dev = torch.device("cuda")
    kv = KVCache(36, a.blocks + 1, 16, 8, 128, torch.bfloat16, dev)
    pool = HostKVPool(kv, a.blocks)
    assert pool.ext is not None, "page-copy kernels did not build"
    rng = random.Random(0)
    ga, gb = (torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16) for _ in range(2))
    for _ in range(5):  # warm the clocks
        _gemm_ms(ga, gb, 20)
    blocks_all, pages_all = rng.sample(range(a.blocks), a.blocks), rng.sample(range(a.blocks), a.blocks)
    batches = [int(x) for x in a.batches.split(",")]
    recs = []
    for ctas in (int(x) for x in a.load_ctas.split(",")):
        pool.load_ctas = ctas
        rec = {"path": "load", "ctas": ctas}
        for n in batches:
            nbytes = n * pool.bytes_per_page
            rec[f"gbps_{n}"] = round(nbytes / _time(lambda: pool.to_gpu(pages_all[:n], blocks_all[:n]).synchronize()) / 1e9, 2)
        rec["gemm_slowdown"] = _slowdown(ga, gb, lambda: pool.to_gpu(pages_all, blocks_all))
        recs.append(rec)
    rec = {"path": "write (staged + copy engine)"}
    for n in batches:
        nbytes = n * pool.bytes_per_page
        rec[f"gbps_{n}"] = round(nbytes / _time(lambda: pool.to_host(blocks_all[:n], pages_all[:n]).synchronize()) / 1e9, 2)
    rec["gemm_slowdown"] = _slowdown(ga, gb, lambda: pool.to_host(blocks_all, pages_all))
    recs.append(rec)
    host = torch.empty(a.blocks * pool.bytes_per_page, dtype=torch.uint8, pin_memory=True)
    gpu = torch.empty_like(host, device=dev)
    recs.append({"path": "cudaMemcpyAsync, contiguous", "mb": round(host.numel() / 1e6, 1),
                 "h2d_gbps": round(host.numel() / _time(lambda: gpu.copy_(host, non_blocking=True)) / 1e9, 2),
                 "d2h_gbps": round(host.numel() / _time(lambda: host.copy_(gpu, non_blocking=True)) / 1e9, 2),
                 "gemm_slowdown_d2h": _slowdown(ga, gb, lambda: host.copy_(gpu, non_blocking=True))})
    for rec in recs:
        print(json.dumps(rec), flush=True)
        if a.save:
            with open(a.save, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
