"""Microbenchmark: KV page copies between the GPU cache and the pinned host pool (P3, L2).

Qwen3-8B shapes: 36 layers, 8 KV heads x 128, BF16, 16-token pages (2.36 MB per page with
every layer). Times HostKVPool.to_host / to_gpu (the page-copy kernel reading or writing mapped
pinned memory) for batches of scattered pages, against one cudaMemcpyAsync of the same number
of contiguous bytes (the PCIe ceiling; P0 measured 12.9 GB/s H2D, 13.2 GB/s D2H).

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=1024)
    ap.add_argument("--batches", default="4,16,64,256,1024")
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    dev = torch.device("cuda")
    kv = KVCache(36, a.blocks + 1, 16, 8, 128, torch.bfloat16, dev)
    pool = HostKVPool(kv, a.blocks)
    assert pool.ext is not None, "page-copy kernels did not build"
    rng = random.Random(0)
    for n in (int(x) for x in a.batches.split(",")):
        blocks, pages = rng.sample(range(a.blocks), n), rng.sample(range(a.blocks), n)
        nbytes = n * pool.bytes_per_page
        d2h = _time(lambda: pool.to_host(blocks, pages).synchronize())
        h2d = _time(lambda: pool.to_gpu(pages, blocks).synchronize())
        host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        gpu = torch.empty(nbytes, dtype=torch.uint8, device=dev)
        memcpy_h2d = _time(lambda: gpu.copy_(host, non_blocking=True))
        memcpy_d2h = _time(lambda: host.copy_(gpu, non_blocking=True))
        rec = {"pages": n, "mb": round(nbytes / 1e6, 1),
               "to_host_gbps": round(nbytes / d2h / 1e9, 2), "to_gpu_gbps": round(nbytes / h2d / 1e9, 2),
               "memcpy_d2h_gbps": round(nbytes / memcpy_d2h / 1e9, 2), "memcpy_h2d_gbps": round(nbytes / memcpy_h2d / 1e9, 2),
               "to_host_ms": round(d2h * 1e3, 2), "to_gpu_ms": round(h2d * 1e3, 2)}
        print(json.dumps(rec), flush=True)
        if a.save:
            with open(a.save, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        del host, gpu


if __name__ == "__main__":
    main()
