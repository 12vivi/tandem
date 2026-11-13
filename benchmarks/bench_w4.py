"""Microbenchmark: W4A16 small-batch GEMV (tandem.layers.awq.w4_gemv) on Qwen3-8B AWQ shapes (A2).

Each measurement replays a CUDA graph of repeated calls that cycles through enough weight copies
(>= 96 MB) to defeat the L2, like a decode step. Reports microseconds per call and the effective
bandwidth (4-bit weights + float16 scales and zero points), and the best config per (shape, M):
CUDA-core variant for M <= 2 candidates, tensor-core variant for all M.

  python benchmarks/bench_w4.py --save benchmarks/results/p2-w4.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
import math

import torch

from bench_gemv import time_us  # noqa: E402  (same directory)
from tandem.layers.awq import w4_gemv

# (N, K) of the fused projections of Qwen3-8B (AWQ, group 128)
SHAPES = {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (24576, 4096), "down": (4096, 12288)}


def configs(m: int):
    out = []
    if m <= 2:
        for bn, bk, nw, ns in itertools.product((1, 2, 4, 8), (64, 128, 256, 512, 1024), (1, 2, 4), (2, 3, 4)):
            if m * bn * bk / (nw * 32) <= 256:
                out.append((bn, bk, nw, ns, False))
    for bn, bk, nw, ns in itertools.product((16, 32, 64), (64, 128, 256), (2, 4, 8), (2, 3, 4)):
        out.append((bn, bk, nw, ns, True))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="1,2,4,8")
    ap.add_argument("--projs", default="")
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    dev, group = torch.device("cuda"), 128
    torch.manual_seed(0)
    for proj, (n, k) in SHAPES.items():
        if a.projs and proj not in a.projs.split(","):
            continue
        nbytes = n * k // 2 + 2 * n * (k // group) * 2
        copies = max(2, math.ceil(96e6 / nbytes))
        ws = [(torch.randint(-2**31, 2**31 - 1, (n, k // 8), dtype=torch.int32, device=dev),
               (torch.rand(n, k // group, device=dev) * 0.01).half(),
               torch.randint(0, 16, (n, k // group), device=dev).half()) for _ in range(copies)]
        for m in (int(v) for v in a.ms.split(",")):
            x = torch.randn(m, k, device=dev, dtype=torch.float16)
            ys = [torch.empty(m, n, device=dev, dtype=torch.float16) for _ in range(copies)]
            best = None
            for cfg in configs(m):
                bn, bk, nw, ns, mma = cfg
                try:
                    us = time_us([lambda w=w, y=y: w4_gemv(x, w[0], w[1], w[2], group, bn, bk, nw, ns, mma, out=y)
                                  for w, y in zip(ws, ys)])
                except Exception as exc:  # noqa: BLE001
                    print(f"  skip {cfg}: {str(exc).splitlines()[0][:100]}")
                    continue
                if best is None or us < best[0]:
                    best = (us, cfg)
            rec = {"proj": proj, "n": n, "k": k, "m": m, "weight_mb": round(nbytes / 1e6, 2),
                   "us": round(best[0], 2), "gbps": round(nbytes / best[0] / 1e3, 1),
                   "config": dict(zip(("block_n", "block_k", "num_warps", "num_stages", "mma"), best[1]))}
            print(json.dumps(rec), flush=True)
            if a.save:
                with open(a.save, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
        del ws
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
