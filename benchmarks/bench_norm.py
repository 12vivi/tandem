"""Microbenchmark: fused residual add + RMSNorm at decode batch sizes, FlashInfer vs Triton
(tandem.layers.triton_ops.fused_add_rmsnorm).

Each decode step runs this twice per layer (72 times for Qwen3-8B). The P2 profile of
Qwen3-8B-AWQ showed FlashInfer's kernel at 2.37 us per call at batch 1 but 3.07 us at batch 8.
Timed in a CUDA graph over 8 distinct (x, residual) pairs, as in a decode graph; also checks
that both give the same result.

  python benchmarks/bench_norm.py --save benchmarks/results/p2-norm.jsonl
"""

from __future__ import annotations

import argparse
import json

import flashinfer
import torch

from bench_gemv import time_us  # noqa: E402  (same directory)
from tandem.layers.triton_ops import fused_add_rmsnorm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", default="1024,2048,4096")
    ap.add_argument("--ms", default="1,2,4,8,16,32,64")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    dt, dev, eps = getattr(torch, a.dtype), torch.device("cuda"), 1e-6
    for h in (int(v) for v in a.hidden.split(",")):
        w = (1.0 + 0.1 * torch.randn(h, device=dev)).to(dt)
        for m in (int(v) for v in a.ms.split(",")):
            x0, r0 = torch.randn(m, h, device=dev, dtype=dt), torch.randn(m, h, device=dev, dtype=dt)
            xa, ra, xb, rb = x0.clone(), r0.clone(), x0.clone(), r0.clone()
            flashinfer.norm.fused_add_rmsnorm(xa, ra, w, eps)
            fused_add_rmsnorm(xb, rb, w, eps)
            same = torch.equal(ra, rb)
            err = (xa.float() - xb.float()).abs().max().item()
            pairs = [(torch.randn(m, h, device=dev, dtype=dt), torch.randn(m, h, device=dev, dtype=dt)) for _ in range(8)]
            rec = {"hidden": h, "m": m, "dtype": a.dtype, "residual_equal": same, "out_max_abs_diff": err,
                   "flashinfer_us": round(time_us([lambda p=p: flashinfer.norm.fused_add_rmsnorm(p[0], p[1], w, eps)
                                                   for p in pairs]), 2)}
            for nw in (2, 4, 8, 16):
                rec[f"triton_w{nw}_us"] = round(time_us([lambda p=p: fused_add_rmsnorm(p[0], p[1], w, eps, num_warps=nw)
                                                         for p in pairs]), 2)
            print(json.dumps(rec), flush=True)
            if a.save:
                with open(a.save, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
