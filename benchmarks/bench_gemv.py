"""Microbenchmark: small-batch GEMV, Triton (tandem.layers.triton_gemv) vs cuBLAS (torch.mm),
on the projection shapes of Qwen3-0.6B / 1.7B / 8B at M = 1, 2, 4, 8 (design doc P2, A1 / A2).

Each measurement replays a CUDA graph of repeated calls that cycles through enough weight
copies (>= 96 MB) to defeat the 6 MB L2, so every call streams its weight from DRAM as in a
decode step. Reports microseconds per call and effective bandwidth (weight bytes / time), and
the best Triton config per (shape, M).

  python benchmarks/bench_gemv.py --models 0.6b,1.7b --save benchmarks/results/p2-gemv.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time

import torch

from tandem.layers.triton_gemv import gemv

# (N, K) of y = x @ w^T for each projection (qkv and gate_up are the merged weights)
SHAPES = {
    "0.6b": {"qkv": (4096, 1024), "o": (1024, 2048), "gate_up": (6144, 1024), "down": (1024, 3072),
             "lm_head": (151936, 1024)},
    "1.7b": {"qkv": (4096, 2048), "o": (2048, 2048), "gate_up": (12288, 2048), "down": (2048, 6144),
             "lm_head": (151936, 2048)},
    "8b": {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (24576, 4096), "down": (4096, 12288)},
}


def time_us(calls, iters: int = 10, reps: int = 5) -> float:
    """Microseconds per call: CUDA graph of ``iters`` rounds over ``calls``, replayed ``reps`` times."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # warm-up (Triton compiles here, outside capture)
        for c in calls:
            c()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            for c in calls:
                c()
    graph.replay()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(reps):
        graph.replay()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1e3 / (reps * iters * len(calls))


def configs(m: int, k: int, sweep: bool):
    """(block_n, block_k, num_warps, num_stages, defer) candidates."""
    if not sweep:
        return [(2, 1024, 2, 3, False), (2, 512, 2, 3, True), (4, 256, 4, 3, True)]
    out = []
    bm = 1 << (m - 1).bit_length() if m > 1 else 1
    for bn, bk, nw, ns, defer in itertools.product((1, 2, 4, 8), (256, 512, 1024), (1, 2, 4), (2, 3),
                                                   (False, True)):
        per_thread = bm * bn * bk / (nw * 32)  # values of the [BM, BN, BK] product per thread
        if k % bk or per_thread > (128 if defer else 256):  # skip register-spilling tiles
            continue
        out.append((bn, bk, nw, ns, defer))
    if m >= 2:  # tensor-core variant (x padded to 16 rows)
        for bn, bk, nw, ns in itertools.product((16, 32, 64), (64, 128, 256), (1, 2, 4), (2, 3, 4)):
            if k % bk == 0:
                out.append((bn, bk, nw, ns, 2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="0.6b,1.7b")
    ap.add_argument("--ms", default="1,2,4,8")
    ap.add_argument("--projs", default="", help="subset, e.g. qkv,o (default: all)")
    ap.add_argument("--sweep", action="store_true", help="search block / warp / stage configs")
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    dev = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(0)
    for model in a.models.split(","):
        for proj, (n, k) in SHAPES[model].items():
            if a.projs and proj not in a.projs.split(","):
                continue
            nbytes = n * k * 2
            copies = max(2, math.ceil(96e6 / nbytes))
            ws = [torch.randn(n, k, device=dev, dtype=dtype) * 0.02 for _ in range(copies)]
            for m in (int(x) for x in a.ms.split(",")):
                x = torch.randn(m, k, device=dev, dtype=dtype)
                ys = [torch.empty(m, n, device=dev, dtype=dtype) for _ in range(copies)]
                ref = torch.mm(x.float(), ws[0].float().t())
                cublas = time_us([lambda w=w, y=y: torch.mm(x, w.t(), out=y) for w, y in zip(ws, ys)])
                best = None
                t_start = time.time()
                for cfg in configs(m, k, a.sweep):
                    bn, bk, nw, ns, defer = cfg
                    try:
                        out = gemv(x, ws[0], bn, bk, nw, ns, defer=defer)
                        err = (out.float() - ref).abs().max().item()
                        if err > 0.02 * ref.abs().max().item() + 1e-3:
                            raise RuntimeError(f"wrong result: max err {err}")
                        us = time_us([lambda w=w, y=y: gemv(x, w, bn, bk, nw, ns, out=y, defer=defer)
                                      for w, y in zip(ws, ys)])
                    except Exception as exc:  # noqa: BLE001 - report and skip configs Triton rejects
                        print(f"  skip {cfg}: {str(exc).splitlines()[0][:100]}")
                        continue
                    if best is None or us < best[0]:
                        best = (us, cfg)
                rec = {"model": model, "proj": proj, "n": n, "k": k, "m": m, "weight_mb": round(nbytes / 1e6, 2),
                       "cublas_us": round(cublas, 2), "cublas_gbps": round(nbytes / cublas / 1e3, 1),
                       "triton_us": round(best[0], 2), "triton_gbps": round(nbytes / best[0] / 1e3, 1),
                       "speedup": round(cublas / best[0], 3), "config": dict(zip(("block_n", "block_k", "num_warps", "num_stages",
                                                                                  "variant"), best[1])),
                       "search_s": round(time.time() - t_start, 1)}
                print(json.dumps(rec), flush=True)
                if a.save:
                    with open(a.save, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec) + "\n")
            del ws
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
