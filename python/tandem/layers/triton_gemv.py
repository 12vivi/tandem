"""Small-batch GEMV for decode (design doc §4.3, P2): y[M, N] = x[M, K] @ w[N, K]^T, M <= 8.

At batch 1-8 a decode step reads every weight once per step, so the projections are bound by
memory bandwidth, not FLOPs. cuBLAS reaches about 590-730 GB/s on Qwen3-0.6B's projection
shapes on the RTX 3090 (P1 profile), well below what a streaming read gets (lm_head: 877 GB/s).

The kernel streams BLOCK_N rows of ``w`` per program and reduces each against all M rows of
``x`` on CUDA cores: x[M, 1, BK] * w[1, BN, BK] summed over K in float32. No tensor cores (M is
too small to fill an MMA tile); every weight byte is read exactly once, x stays in L1/L2.
Config per (N, K, M) comes from ``benchmarks/bench_gemv.py``.
"""

from __future__ import annotations

import functools
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn


@triton.jit
def _gemv_kernel(
    x_ptr, w_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_wn, stride_ym,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, DEFER: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :]
    if DEFER:  # per-thread partial sums over K, one cross-thread reduction at the end
        acc3 = tl.zeros((BLOCK_M, BLOCK_N, BLOCK_K), dtype=tl.float32)
        for _ in range(0, K, BLOCK_K):  # K % BLOCK_K == 0 (checked on the host)
            w = tl.load(w_ptrs, mask=mask_n[:, None], other=0.0, eviction_policy="evict_first").to(tl.float32)
            x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0, eviction_policy="evict_last").to(tl.float32)
            acc3 += x[:, None, :] * w[None, :, :]
            x_ptrs += BLOCK_K
            w_ptrs += BLOCK_K
        acc = tl.sum(acc3, axis=2)
    else:  # reduce every BLOCK_K chunk (fewer registers)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in range(0, K, BLOCK_K):
            w = tl.load(w_ptrs, mask=mask_n[:, None], other=0.0, eviction_policy="evict_first").to(tl.float32)
            x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0, eviction_policy="evict_last").to(tl.float32)
            acc += tl.sum(x[:, None, :] * w[None, :, :], axis=2)
            x_ptrs += BLOCK_K
            w_ptrs += BLOCK_K
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :]
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])



@triton.jit
def _gemv_mma_kernel(
    x_ptr, w_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_wn, stride_ym,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Tensor-core variant for M = 5-16: x padded to 16 rows, one MMA tile per BLOCK_K chunk."""
    BLOCK_M: tl.constexpr = 16
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0, eviction_policy="evict_last")
        w = tl.load(w_ptrs, mask=mask_n[:, None], other=0.0, eviction_policy="evict_first")
        acc = tl.dot(x, tl.trans(w), acc)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :]
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])



@triton.jit
def _gemv_prologue_kernel(
    x_ptr, r_ptr, g_ptr, w_ptr, y_ptr, r_out_ptr,
    M, N, K,
    stride_xm, stride_rm, stride_wn, stride_ym,
    eps,
    PROLOGUE: tl.constexpr,  # 1: silu(x[:, :K]) * x[:, K:]   2: rmsnorm(x + r) * g, r_out = x + r   3: rmsnorm(x) * g
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """GEMV whose input vector is computed on the fly from the previous op's output, so the
    activation / norm kernel before the projection disappears. Every program recomputes the
    (cheap, L2-resident) input; program 0 also writes the updated residual (PROLOGUE 2)."""
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    rstd = tl.zeros((BLOCK_M,), dtype=tl.float32)
    if PROLOGUE >= 2:  # pass 1: rms of (x + r) over the whole row
        ss = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            v = tl.load(x_ptr + offs_m[:, None] * stride_xm + k0 + offs_k[None, :], mask=mask_m[:, None], other=0.0)
            v = v.to(tl.float32)
            if PROLOGUE == 2:
                v += tl.load(r_ptr + offs_m[:, None] * stride_rm + k0 + offs_k[None, :], mask=mask_m[:, None],
                             other=0.0).to(tl.float32)
            ss += tl.sum(v * v, axis=1)
        rstd = tl.rsqrt(ss / K + eps)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :]
    for k0 in range(0, K, BLOCK_K):
        xk = x_ptr + offs_m[:, None] * stride_xm + k0 + offs_k[None, :]
        if PROLOGUE == 1:
            gate = tl.load(xk, mask=mask_m[:, None], other=0.0).to(tl.float32)
            up = tl.load(xk + K, mask=mask_m[:, None], other=0.0).to(tl.float32)
            a = gate / (1.0 + tl.exp(-gate)) * up
        else:
            v = tl.load(xk, mask=mask_m[:, None], other=0.0).to(tl.float32)
            if PROLOGUE == 2:
                v += tl.load(r_ptr + offs_m[:, None] * stride_rm + k0 + offs_k[None, :], mask=mask_m[:, None],
                             other=0.0).to(tl.float32)
                if pid == 0:
                    tl.store(r_out_ptr + offs_m[:, None] * stride_rm + k0 + offs_k[None, :],
                             v.to(r_out_ptr.dtype.element_ty), mask=mask_m[:, None])
            gw = tl.load(g_ptr + k0 + offs_k).to(tl.float32)
            a = v * rstd[:, None] * gw[None, :]
        # round the input like the unfused path (activation / norm write bf16 before the GEMV)
        a = a.to(x_ptr.dtype.element_ty).to(tl.float32)
        w = tl.load(w_ptrs, mask=mask_n[:, None], other=0.0, eviction_policy="evict_first").to(tl.float32)
        acc += tl.sum(a[:, None, :] * w[None, :, :], axis=2)
        w_ptrs += BLOCK_K
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :]
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


def gemv_prologue(
    mode: str,
    x: torch.Tensor,
    w: torch.Tensor,
    residual: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    eps: float = 1e-6,
    block_n: int = 2,
    block_k: int = 512,
    num_warps: int = 1,
    num_stages: int = 3,
):
    """Fused input + GEMV for M <= 8.
    mode "silu_mul": y = (silu(x[:, :K]) * x[:, K:]) @ w^T            -> y
    mode "add_rmsnorm": r' = x + residual; y = rmsnorm(r') * g @ w^T   -> (y, r')
    mode "rmsnorm": y = rmsnorm(x) * g @ w^T                           -> y"""
    m = x.shape[0]
    n, k = w.shape
    prologue = {"silu_mul": 1, "add_rmsnorm": 2, "rmsnorm": 3}[mode]
    if m > 8 or k % block_k or x.stride(1) != 1 or w.stride(1) != 1 or x.shape[1] != (2 * k if prologue == 1 else k):
        raise ValueError(f"gemv_prologue({mode}): unsupported shapes x {tuple(x.shape)} w {tuple(w.shape)}")
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    r_out = torch.empty_like(x) if prologue == 2 else y
    r = residual if prologue == 2 else x
    g = norm_weight if prologue >= 2 else x
    _gemv_prologue_kernel[(triton.cdiv(n, block_n),)](
        x, r, g, w, y, r_out, m, n, k, x.stride(0), r.stride(0), w.stride(0), y.stride(0), eps,
        PROLOGUE=prologue, BLOCK_M=max(1, triton.next_power_of_2(m)), BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=num_warps, num_stages=num_stages,
    )
    return (y, r_out) if prologue == 2 else y


def gemv(
    x: torch.Tensor,
    w: torch.Tensor,
    block_n: int = 8,
    block_k: int = 256,
    num_warps: int = 4,
    num_stages: int = 3,
    out: torch.Tensor | None = None,
    defer: int = 0,
) -> torch.Tensor:
    """x [M, K] (M <= 8), w [N, K] row-major, same dtype -> [M, N] in x.dtype.
    ``defer`` picks the variant: 0 reduce every K chunk, 1 deferred reduction, 2 tensor cores."""
    m, k = x.shape
    n, k2 = w.shape
    if k != k2 or m > 8 or k % block_k or x.stride(1) != 1 or w.stride(1) != 1:
        raise ValueError(f"gemv: unsupported shapes x {tuple(x.shape)} w {tuple(w.shape)} block_k {block_k}")
    y = out if out is not None else torch.empty((m, n), dtype=x.dtype, device=x.device)
    if int(defer) == 2:
        _gemv_mma_kernel[(triton.cdiv(n, block_n),)](
            x, w, y, m, n, k, x.stride(0), w.stride(0), y.stride(0),
            BLOCK_N=block_n, BLOCK_K=block_k, num_warps=num_warps, num_stages=num_stages,
        )
        return y
    block_m = max(1, triton.next_power_of_2(m))
    _gemv_kernel[(triton.cdiv(n, block_n),)](
        x, w, y, m, n, k, x.stride(0), w.stride(0), y.stride(0),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, DEFER=bool(defer), num_warps=num_warps,
        num_stages=num_stages,
    )
    return y


# ---------------------------------------------------------------------------- dispatch
# Config per GPU and (N, K, M bucket): (block_n, block_k, num_warps, num_stages, defer). Only
# shapes where the Triton kernel beat cuBLAS by >= 3% in benchmarks/bench_gemv.py are listed;
# everything else keeps cuBLAS. M buckets are the CUDA-graph decode batch sizes 1, 2, 4, 8.
Config = tuple[int, int, int, int, int]  # ..., variant (gemv's defer)
# Measured 2026-11-13 (benchmarks/results/p2-gemv*.jsonl), best of all sweeps per shape. Variant
# 0 / 1: CUDA-core kernel (reduce per chunk / deferred), 2: tensor-core kernel (x padded to 16
# rows), which wins at M = 4-8 and on some M = 2 shapes.
CONFIGS: dict[str, dict[tuple[int, int, int], Config]] = {
    "NVIDIA GeForce RTX 3090": {
        (1024, 2048, 1): (1, 1024, 1, 3, 0),  # 0.6b o: x1.14, 675 GB/s
        (1024, 2048, 2): (1, 1024, 1, 3, 0),  # 0.6b o: x1.50, 657 GB/s
        (1024, 2048, 4): (16, 128, 2, 4, 2),  # 0.6b o: x1.43, 626 GB/s
        (1024, 2048, 8): (16, 128, 2, 4, 2),  # 0.6b o: x1.52, 624 GB/s
        (1024, 3072, 1): (1, 1024, 1, 3, 0),  # 0.6b down: x1.21, 737 GB/s
        (1024, 3072, 2): (1, 512, 1, 3, 1),  # 0.6b down: x1.34, 708 GB/s
        (1024, 3072, 4): (16, 128, 2, 4, 2),  # 0.6b down: x1.32, 695 GB/s
        (1024, 3072, 8): (16, 128, 2, 4, 2),  # 0.6b down: x1.32, 691 GB/s
        (2048, 2048, 1): (2, 512, 1, 3, 0),  # 1.7b o: x1.13, 768 GB/s
        (2048, 2048, 2): (2, 512, 1, 3, 0),  # 1.7b o: x1.43, 756 GB/s
        (2048, 2048, 4): (16, 64, 2, 4, 2),  # 1.7b o: x1.38, 743 GB/s
        (2048, 2048, 8): (16, 64, 2, 4, 2),  # 1.7b o: x1.39, 740 GB/s
        (2048, 6144, 1): (1, 1024, 2, 3, 1),  # 1.7b down: x1.20, 837 GB/s
        (2048, 6144, 2): (2, 512, 1, 3, 1),  # 1.7b down: x1.39, 818 GB/s
        (2048, 6144, 4): (16, 256, 2, 4, 2),  # 1.7b down: x1.34, 798 GB/s
        (2048, 6144, 8): (16, 128, 2, 4, 2),  # 1.7b down: x1.35, 793 GB/s
        (4096, 1024, 1): (4, 256, 1, 2, 0),  # 0.6b qkv: x1.33, 758 GB/s
        (4096, 1024, 2): (4, 256, 1, 3, 1),  # 0.6b qkv: x1.24, 750 GB/s
        (4096, 1024, 4): (32, 64, 2, 3, 2),  # 0.6b qkv: x1.15, 745 GB/s
        (4096, 1024, 8): (32, 64, 4, 4, 2),  # 0.6b qkv: x1.19, 723 GB/s
        (4096, 2048, 1): (2, 512, 1, 3, 1),  # 1.7b qkv: x1.17, 819 GB/s
        (4096, 2048, 2): (16, 64, 2, 4, 2),  # 1.7b qkv: x1.13, 808 GB/s
        (4096, 2048, 4): (16, 64, 2, 3, 2),  # 1.7b qkv: x1.12, 805 GB/s
        (4096, 2048, 8): (16, 64, 2, 4, 2),  # 1.7b qkv: x1.13, 802 GB/s
        (6144, 1024, 1): (1, 512, 1, 3, 0),  # 0.6b gate_up: x1.09, 807 GB/s
        (6144, 1024, 2): (1, 512, 1, 3, 1),  # 0.6b gate_up: x1.10, 796 GB/s
        (6144, 1024, 4): (16, 64, 2, 3, 2),  # 0.6b gate_up: x1.09, 788 GB/s
        (6144, 1024, 8): (32, 64, 4, 3, 2),  # 0.6b gate_up: x1.08, 778 GB/s
        (12288, 2048, 1): (1, 1024, 1, 3, 0),  # 1.7b gate_up: x1.03, 864 GB/s
        (12288, 2048, 2): (16, 128, 2, 2, 2),  # 1.7b gate_up: x1.03, 860 GB/s
        (12288, 2048, 4): (16, 128, 2, 2, 2),  # 1.7b gate_up: x1.03, 856 GB/s
        (12288, 2048, 8): (16, 128, 2, 2, 2),  # 1.7b gate_up: x1.04, 854 GB/s
        (151936, 1024, 1): (2, 512, 2, 2, 0),  # 0.6b lm_head: x1.03, 889 GB/s
        (151936, 1024, 2): (2, 512, 2, 2, 0),  # 0.6b lm_head: x1.04, 887 GB/s
        (151936, 1024, 4): (4, 1024, 2, 2, 0),  # 0.6b lm_head: x1.04, 882 GB/s
        (151936, 2048, 1): (2, 1024, 4, 2, 0),  # 1.7b lm_head: x1.03, 891 GB/s
        (151936, 2048, 2): (2, 1024, 4, 2, 0),  # 1.7b lm_head: x1.03, 889 GB/s
    },
}

# decoder-layer projections that run per decode step (the merged qkv / gate_up weights)
PROJECTIONS = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")


def m_bucket(m: int) -> int:
    return 1 if m <= 1 else 2 if m == 2 else 4 if m <= 4 else 8


def table_for(device: torch.device, model: nn.Module) -> dict[tuple[int, int, int], Config]:
    """This GPU's tuned table; ``TANDEM_GEMV_FORCE=1`` (or 2, which also fuses the norms and the
    activation, see fused_table_for) sends every projection through the kernel (tests)."""
    if os.environ.get("TANDEM_GEMV_FORCE") in ("1", "2"):
        return default_table(model)
    return CONFIGS.get(torch.cuda.get_device_name(device), {})


def linear(x: torch.Tensor, w: torch.Tensor, table: dict[tuple[int, int, int], Config]) -> torch.Tensor:
    """F.linear(x, w) without bias, through the small-batch kernel where ``table`` says it wins."""
    if x.dim() == 2 and x.shape[0] <= 8 and x.is_contiguous() and x.dtype == w.dtype:
        cfg = table.get((w.shape[0], w.shape[1], m_bucket(x.shape[0])))
        if cfg is not None:
            bn, bk, nw, ns, defer = cfg
            return gemv(x, w, bn, bk, nw, ns, defer=defer)
    return F.linear(x, w)


def _forward(module: nn.Linear, table: dict, x: torch.Tensor) -> torch.Tensor:
    return linear(x, module.weight, table)


def install(model: nn.Module, table: dict[tuple[int, int, int], Config]) -> int:
    """Route the decoder projections of ``model`` through ``linear``; returns how many."""
    if not table:
        return 0
    n = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and mod.bias is None and name.rsplit(".", 1)[-1] in PROJECTIONS:
            mod.forward = functools.partial(_forward, mod, table)  # instance attribute shadows Linear.forward
            n += 1
    model.gemv_table = table  # compute_logits routes lm_head through ``linear`` too
    return n


# Fused norm / activation + GEMV (gemv_prologue), per GPU and (mode, N, K, M bucket):
# (block_n, block_k, num_warps, num_stages). A decoder layer takes the fused path at batch M
# only when all three of its pairs have an entry (benchmarks/bench_gemv_fused.py).
FusedConfig = tuple[int, int, int, int]
# Measured 2026-11-13 (benchmarks/results/p2-fused.jsonl): the fused kernel only won at M = 1, by
# 0.4-0.9 us per pair in the microbenchmark (at M >= 2 recomputing the norm in every program costs
# more than the separate kernel it replaces). In the decode graph the gain vanished: Qwen3-0.6B
# decode-1 dropped 84 kernels per step but the fused GEMVs got slower by about as much (2124 vs
# 2123 us per step, A1 TPOT 2.04 vs 2.01 ms). So no shape is enabled; TANDEM_GEMV_FORCE=2 still
# exercises the path in tests. The entries that won in isolation, for reference:
#   ("add_rmsnorm", 4096, 1024, 1): (1, 1024, 2, 2)   ("add_rmsnorm", 6144, 1024, 1): (1, 1024, 2, 3)
#   ("silu_mul", 1024, 3072, 1): (1, 1024, 1, 2)      ("add_rmsnorm", 4096, 2048, 1): (2, 1024, 4, 2)
#   ("silu_mul", 2048, 6144, 1): (2, 512, 2, 2)
FUSED_CONFIGS: dict[str, dict[tuple[str, int, int, int], FusedConfig]] = {}


class FusedLayer:
    """One decoder layer with its small kernels folded into the projections where the table says
    it pays: add_rmsnorm -> qkv_proj, add_rmsnorm -> gate_up_proj, silu_and_mul -> down_proj.
    Pairs without an entry run as before (separate FlashInfer kernel + projection)."""

    def __init__(self, layer: nn.Module, table: dict[tuple[str, int, int, int], FusedConfig]) -> None:
        self.wqkv = layer.self_attn.qkv_proj.weight
        self.wgu = layer.mlp.gate_up_proj.weight
        self.wdown = layer.mlp.down_proj.weight
        self.table = table

    def _cfg(self, mode: str, w: torch.Tensor, m: int) -> FusedConfig | None:
        return self.table.get((mode, w.shape[0], w.shape[1], m_bucket(m)))

    def applies(self, m: int) -> bool:
        return m <= 8 and any(self._cfg(mode, w, m) is not None for mode, w in
                              (("add_rmsnorm", self.wqkv), ("add_rmsnorm", self.wgu), ("silu_mul", self.wdown)))

    @staticmethod
    def _run(cfg: FusedConfig, mode: str, x: torch.Tensor, w: torch.Tensor, **kw):
        bn, bk, nw, ns = cfg
        return gemv_prologue(mode, x, w, block_n=bn, block_k=bk, num_warps=nw, num_stages=ns, **kw)

    def forward(self, layer: nn.Module, hidden: torch.Tensor, residual: torch.Tensor | None, rope, ctx):
        ops = ctx.ops
        m = hidden.shape[0]
        n1, n2 = layer.input_layernorm, layer.post_attention_layernorm
        cfg = self._cfg("add_rmsnorm", self.wqkv, m)
        if cfg is None:
            if residual is None:
                residual = hidden
                hidden = ops.rms_norm(hidden, n1.weight, n1.eps)
            else:
                hidden, residual = ops.fused_add_rms_norm(hidden, residual, n1.weight, n1.eps)
            hidden = layer.self_attn(hidden, rope, ctx)
        else:
            if residual is None:  # first layer: plain norm, the residual stream starts at the embedding
                qkv = self._run(cfg, "rmsnorm", hidden, self.wqkv, norm_weight=n1.weight, eps=n1.eps)
                residual = hidden
            else:
                qkv, residual = self._run(cfg, "add_rmsnorm", hidden, self.wqkv, residual=residual,
                                          norm_weight=n1.weight, eps=n1.eps)
            hidden = layer.self_attn(None, rope, ctx, qkv=qkv)
        cfg = self._cfg("add_rmsnorm", self.wgu, m)
        if cfg is None:
            hidden, residual = ops.fused_add_rms_norm(hidden, residual, n2.weight, n2.eps)
            gate_up = layer.mlp.gate_up_proj(hidden)
        else:
            gate_up, residual = self._run(cfg, "add_rmsnorm", hidden, self.wgu, residual=residual,
                                          norm_weight=n2.weight, eps=n2.eps)
        cfg = self._cfg("silu_mul", self.wdown, m)
        if cfg is None:
            return layer.mlp.down_proj(ops.silu_and_mul(gate_up)), residual
        return self._run(cfg, "silu_mul", gate_up, self.wdown), residual


def install_fused(model: nn.Module, table: dict[tuple[str, int, int, int], FusedConfig]) -> int:
    """Attach a FusedLayer to every dense decoder layer; returns how many."""
    if not table:
        return 0
    n = 0
    for mod in model.modules():
        if hasattr(mod, "self_attn") and hasattr(mod, "mlp") and hasattr(getattr(mod, "mlp"), "gate_up_proj"):
            layer = FusedLayer(mod, table)
            if any(layer.applies(m) for m in (1, 2, 4, 8)):  # only layers whose shapes have an entry
                mod.fused_gemv = layer
                n += 1
    return n


def fused_table_for(device: torch.device, model: nn.Module) -> dict[tuple[str, int, int, int], FusedConfig]:
    """``TANDEM_GEMV_FORCE=2`` fuses every dense layer with a default config (tests)."""
    if os.environ.get("TANDEM_GEMV_FORCE") == "2":
        table = {}
        for mod in model.modules():
            if hasattr(mod, "self_attn") and hasattr(getattr(mod, "mlp", None), "gate_up_proj"):
                for mode, w in (("add_rmsnorm", mod.self_attn.qkv_proj.weight),
                                ("add_rmsnorm", mod.mlp.gate_up_proj.weight), ("silu_mul", mod.mlp.down_proj.weight)):
                    bk = 256 if w.shape[1] % 256 == 0 else 128
                    for m in (1, 2, 4, 8):
                        table[(mode, w.shape[0], w.shape[1], m)] = (2, bk, 2, 3)
        return table
    return FUSED_CONFIGS.get(torch.cuda.get_device_name(device), {})


def default_table(model: nn.Module) -> dict[tuple[int, int, int], Config]:
    """A table that sends every projection shape of ``model`` through the kernel (tests)."""
    table = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name.rsplit(".", 1)[-1] in PROJECTIONS:
            n, k = mod.weight.shape
            bk = 256 if k % 256 == 0 else 128
            for m in (1, 2, 4, 8):
                table[(n, k, m)] = (2, bk, 2, 3, m == 1)
    return table
