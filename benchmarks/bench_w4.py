"""Microbenchmark: W4A16 decode kernels (tandem/csrc/w4_gemv.cu) on Qwen3-8B AWQ shapes (A2).

Each measurement replays a CUDA graph of repeated calls that cycles through enough weight copies
(>= 96 MB) to defeat the L2, like a decode step. Reports microseconds per call, the effective
bandwidth (4-bit weights + float16 scales and zero points), the best config of each kernel
(w4_gemv: CUDA cores, M <= 4; w4_mma: tensor cores) and, for scale, cuBLAS on float16 weights.

  python benchmarks/bench_w4.py --save benchmarks/results/p2-w4.jsonl
"""

from __future__ import annotations

import argparse
import json
import math

import torch

from bench_gemv import time_us  # noqa: E402  (same directory)
from tandem.layers import w4_cuda

# (N, K) of the fused projections of Qwen3-8B (AWQ, group 128)
SHAPES = {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (24576, 4096), "down": (4096, 12288)}
GEMV = ((1, 4), (2, 2), (2, 4), (2, 8), (4, 2), (4, 4), (4, 8), (8, 2), (8, 4))  # (rows, warps)
MMA = ((1, 2, 1), (1, 4, 1), (2, 2, 1), (2, 4, 1), (4, 1, 1), (4, 2, 1), (4, 4, 1), (8, 1, 1), (8, 2, 1),
       (2, 2, 2), (4, 1, 2), (4, 2, 2), (8, 1, 2), (8, 2, 2))  # (warps_k, unroll, row tiles per warp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="1,2,4,8,16")
    ap.add_argument("--projs", default="")
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--act", action="store_true", help="gate_up: also time the fused SwiGLU epilogue")
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    ext = w4_cuda.load()
    assert ext is not None, "W4A16 CUDA kernels did not build"
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
        dense = None
        if not a.no_dense:
            dense = [torch.randn(n, k, device=dev, dtype=torch.float16) * 0.01
                     for _ in range(max(2, math.ceil(96e6 / (n * k * 2))))]
        for m in (int(v) for v in a.ms.split(",")):
            x = torch.randn(m, k, device=dev, dtype=torch.float16)
            res = {}
            for kind, cfgs in (("gemv", GEMV if m <= 4 else ()), ("mma", MMA)):
                for cfg in cfgs:
                    if kind == "gemv":
                        calls = [lambda w=w: ext.w4_gemv(x, w[0], w[1], w[2], *cfg) for w in ws]
                    else:
                        calls = [lambda w=w: ext.w4_mma(x, w[0], w[1], w[2], *cfg, 0) for w in ws]
                    us = time_us(calls)
                    if kind not in res or us < res[kind][0]:
                        res[kind] = (us, cfg)
            best_kind = min(res, key=lambda kd: res[kd][0])
            us = res[best_kind][0]
            rec = {"proj": proj, "n": n, "k": k, "m": m, "weight_mb": round(nbytes / 1e6, 2),
                   "us": round(us, 2), "gbps": round(nbytes / us / 1e3, 1),
                   "kernel": [best_kind, *res[best_kind][1]],
                   **{f"{kd}_us": round(v[0], 2) for kd, v in res.items()},
                   **{f"{kd}_cfg": list(v[1]) for kd, v in res.items()}}
            if a.act and proj == "gate_up":
                # SwiGLU in the tensor-core kernel's epilogue vs the best kernel + FlashInfer's activation
                import flashinfer

                best_act = None
                for cfg in MMA:
                    t = time_us([lambda w=w: ext.w4_mma(x, w[0], w[1], w[2], *cfg, 1) for w in ws])
                    if best_act is None or t < best_act[0]:
                        best_act = (t, cfg)
                kd, cfg = best_kind, res[best_kind][1]
                fn = ext.w4_gemv if kd == "gemv" else (lambda *args: ext.w4_mma(*args, 0))
                rec["unfused_us"] = round(time_us(
                    [lambda w=w: flashinfer.activation.silu_and_mul(fn(x, w[0], w[1], w[2], *cfg)) for w in ws]), 2)
                rec["fused_us"], rec["fused_cfg"] = round(best_act[0], 2), list(best_act[1])
            if dense is not None:
                rec["dense_us"] = round(time_us([lambda w=w: torch.mm(x, w.t()) for w in dense]), 2)
            print(json.dumps(rec), flush=True)
            if a.save:
                with open(a.save, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
        del ws, dense
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
