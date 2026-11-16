"""Microbenchmark: W4A16 GEMM for M > 16 (prefill, decode batches over 16) on Qwen3-8B AWQ shapes.

Paths per (projection, M):
  unfused  w4_dequant (Triton, float16 copy of the weights) + cuBLAS (+ FlashInfer SwiGLU for
           gate_up_silu): what W4Linear did for M > 16
  dense    cuBLAS alone on float16 weights: the unfused path if dequantization were free
  cuda     w4_gemm in tandem/csrc/w4_gemv.cu (dequantized in registers), best config
Each timing replays a CUDA graph cycling through >= 96 MB of weight copies (no L2 reuse across
calls, like a forward pass). Every config's output is checked against the unfused path.

  python benchmarks/bench_w4_gemm.py --save benchmarks/results/p2-w4g.jsonl
"""

from __future__ import annotations

import argparse
import json
import math

import torch
import torch.nn.functional as F

from bench_gemv import time_us  # noqa: E402  (same directory)
from tandem.layers import w4_cuda
from tandem.layers.awq import silu_and_mul, w4_dequant

# (N, K) of the fused projections of Qwen3-8B (AWQ, group 128); gate_up_silu: with the SwiGLU
SHAPES = {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (24576, 4096), "gate_up_silu": (24576, 4096),
          "down": (4096, 12288)}


# (MT, NT, WARPS_N, WARPS_K, STAGES): BM = 16 MT tokens x BN = 8 NT WARPS_N rows per CTA
CUDA_GEMM = ((4, 4, 8, 1, 3), (4, 4, 4, 1, 4), (4, 4, 4, 1, 2), (4, 4, 4, 2, 2), (4, 2, 8, 1, 3),
             (2, 4, 8, 1, 4), (2, 4, 4, 2, 3), (2, 4, 4, 1, 3), (2, 2, 4, 2, 4), (2, 2, 8, 1, 4))
SMS = 82  # RTX 3090


def cuda_configs(m: int, n: int, silu: bool, only: set | None = None, split_set: tuple = (0, 1, 2, 3, 4, 6, 8)):
    """CUDA_GEMM x split-K factors (M <= 1024, at most 6 CTAs per SM, within the workspace)."""
    for mt, nt, wn, wk, st in CUDA_GEMM:
        if only and f"{mt}{nt}{wn}{wk}{st}" not in only:
            continue
        bm, bn = 16 * mt, 8 * nt * wn
        tiles = -(-m // bm) * -(-n // bn)
        for splits in split_set:  # 0: split only the last, partial wave of tiles
            if splits > 1 and (m > 1024 or tiles * splits > 6 * SMS or tiles >= 2 * SMS * 3
                               or tiles * splits * bm * bn > w4_cuda.WORKSPACE_FLOATS):
                continue
            yield mt, nt, wn, wk, st, splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="17,24,32,48,64,96,128,160,192,256,320,384,512,768,1024,1536,2048")
    ap.add_argument("--projs", default="")
    ap.add_argument("--configs", default="", help='e.g. "44412,24413": only these (MT NT WARPS_N WARPS_K STAGES)')
    ap.add_argument("--splits", default="0,1,2,3,4,6,8")
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    ext = w4_cuda.load()
    assert ext is not None, "W4A16 CUDA kernels did not build"
    dev, group = torch.device("cuda"), 128
    torch.manual_seed(0)
    for proj, (n, k) in SHAPES.items():
        if a.projs and proj not in a.projs.split(","):
            continue
        silu = proj.endswith("_silu")
        nbytes = n * k // 2 + 2 * n * (k // group) * 2
        copies = max(2, math.ceil(96e6 / nbytes))
        ws = [(torch.randint(-2**31, 2**31 - 1, (n, k // 8), dtype=torch.int32, device=dev),
               (torch.rand(n, k // group, device=dev) * 0.01 + 0.001).half(),
               torch.randint(0, 16, (n, k // group), device=dev).half()) for _ in range(copies)]
        dense = [torch.randn(n, k, device=dev, dtype=torch.float16) * 0.01
                 for _ in range(max(2, math.ceil(96e6 / (n * k * 2))))]
        for m in (int(v) for v in a.ms.split(",")):
            x = torch.randn(m, k, device=dev, dtype=torch.float16)
            flop = 2 * m * n * k

            def unfused(w):
                y = F.linear(x, w4_dequant(w[0], w[1], w[2], group, deinterleave=silu))
                return silu_and_mul(y) if silu else y

            ref = unfused(ws[0]).float()
            scale = ref.abs().max().clamp_min(1e-6)
            worst = 0.0
            best = None
            per_cfg = {}
            buf, locks = w4_cuda.workspace(dev)
            only = set(a.configs.split(",")) if a.configs else None
            for cfg in cuda_configs(m, n, silu, only, tuple(int(v) for v in a.splits.split(","))):
                got = ext.w4_gemm(x, *ws[0], buf, locks, *cfg, int(silu)).float()
                worst = max(worst, ((got - ref).abs().max() / scale).item())
                us = time_us([lambda w=w: ext.w4_gemm(x, w[0], w[1], w[2], buf, locks, *cfg, int(silu)) for w in ws])
                per_cfg["".join(map(str, cfg))] = round(us, 1)
                if best is None or us < best[0]:
                    best = (us, cfg)
            rec = {"proj": proj, "n": n, "k": k, "m": m,
                   "unfused_us": round(time_us([lambda w=w: unfused(w) for w in ws]), 1),
                   "dense_us": round(time_us([lambda w=w: F.linear(x, w) for w in dense]), 1),
                   "cuda_us": round(best[0], 1), "cuda_cfg": list(best[1]), "max_rel_err": round(worst, 5)}
            rec["cuda_tflops"] = round(flop / rec["cuda_us"] / 1e6, 1)
            rec["dense_tflops"] = round(flop / rec["dense_us"] / 1e6, 1)
            rec["speedup"] = round(rec["unfused_us"] / rec["cuda_us"], 3)
            rec["cuda_all"] = per_cfg
            print(json.dumps(rec), flush=True)
            if a.save:
                with open(a.save, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
        del ws, dense
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
