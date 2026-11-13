"""Microbenchmark: GEMV with the norm / activation fused into its input (triton_gemv.gemv_prologue)
vs the separate FlashInfer kernel followed by the tuned small-batch GEMV (design doc P2, A1).

Per decoder layer of Qwen3 three small kernels feed the projections:
  add_rmsnorm -> qkv_proj      add_rmsnorm -> gate_up_proj      silu_and_mul -> down_proj
Both sides run in a CUDA graph cycling through weight copies (>= 96 MB, no L2 reuse); the fused
side searches a few launch configs. Reports us per pair and the saving.

  python benchmarks/bench_gemv_fused.py --save benchmarks/results/p2-fused.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
import math

import flashinfer.activation
import flashinfer.norm
import torch

from tandem.layers.triton_gemv import CONFIGS, gemv, gemv_prologue

from bench_gemv import SHAPES, time_us  # noqa: E402  (same directory)

PAIRS = (("add_rmsnorm", "qkv"), ("add_rmsnorm", "gate_up"), ("silu_mul", "down"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="0.6b,1.7b")
    ap.add_argument("--ms", default="1,2,4")
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    dev, dt = torch.device("cuda"), torch.bfloat16
    table = CONFIGS.get(torch.cuda.get_device_name(dev), {})
    torch.manual_seed(0)
    for model in a.models.split(","):
        hidden = SHAPES[model]["qkv"][1]
        for mode, proj in PAIRS:
            n, k = SHAPES[model][proj]
            copies = max(2, math.ceil(96e6 / (n * k * 2)))
            ws = [torch.randn(n, k, device=dev, dtype=dt) * 0.02 for _ in range(copies)]
            g = torch.rand(k, device=dev, dtype=dt) + 0.5
            for m in (int(v) for v in a.ms.split(",")):
                cfg = table.get((n, k, m))
                if cfg is None:
                    continue
                bn, bk, nw, ns, var = cfg
                if mode == "silu_mul":
                    h = torch.randn(m, 2 * k, device=dev, dtype=dt)
                    act = torch.empty(m, k, device=dev, dtype=dt)

                    def unfused(w, h=h, act=act):
                        flashinfer.activation.silu_and_mul(h, out=act)
                        gemv(act, w, bn, bk, nw, ns, defer=var)
                    ref = (torch.nn.functional.silu(h[:, :k].float()) * h[:, k:].float()).to(dt).float() @ ws[0].float().t()
                    fused_args = dict(x=h)
                else:
                    x = torch.randn(m, hidden, device=dev, dtype=dt)
                    r = torch.randn(m, hidden, device=dev, dtype=dt)
                    xs, rs = x.clone(), r.clone()

                    def unfused(w, xs=xs, rs=rs):
                        flashinfer.norm.fused_add_rmsnorm(xs, rs, g, 1e-6)
                        gemv(xs, w, bn, bk, nw, ns, defer=var)
                    v = x.float() + r.float()
                    normed = (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6) * g.float()).to(dt).float()
                    ref = normed @ ws[0].float().t()
                    fused_args = dict(x=x, residual=r, norm_weight=g, eps=1e-6)
                base = time_us([lambda w=w: unfused(w) for w in ws])
                best = None
                for fbn, fbk, fnw, fns in itertools.product((1, 2, 4, 8), (256, 512, 1024), (1, 2, 4), (2, 3)):
                    if k % fbk or max(1, 1 << (m - 1).bit_length()) * fbn * fbk / (fnw * 32) > 256:
                        continue
                    try:
                        out = gemv_prologue(mode, w=ws[0], block_n=fbn, block_k=fbk, num_warps=fnw, num_stages=fns,
                                            **fused_args)
                        out = out[0] if isinstance(out, tuple) else out
                        err = (out.float() - ref).abs().max().item()
                        if err > 0.03 * ref.abs().max().item() + 1e-2:
                            raise RuntimeError(f"wrong result: max err {err}")
                        us = time_us([lambda w=w: gemv_prologue(mode, w=w, block_n=fbn, block_k=fbk, num_warps=fnw,
                                                                num_stages=fns, **fused_args) for w in ws])
                    except Exception as exc:  # noqa: BLE001
                        print(f"  skip {(fbn, fbk, fnw, fns)}: {str(exc).splitlines()[0][:100]}")
                        continue
                    if best is None or us < best[0]:
                        best = (us, (fbn, fbk, fnw, fns))
                rec = {"model": model, "pair": f"{mode}+{proj}", "n": n, "k": k, "m": m,
                       "unfused_us": round(base, 2), "fused_us": round(best[0], 2), "saving_us": round(base - best[0], 2),
                       "fused_config": dict(zip(("block_n", "block_k", "num_warps", "num_stages"), best[1]))}
                print(json.dumps(rec), flush=True)
                if a.save:
                    with open(a.save, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec) + "\n")
            del ws
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
