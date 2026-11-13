"""4-bit AWQ weights (W4A16, design doc P2 / A2): load-time repack + Triton kernels.

AutoAWQ "gemm" checkpoints store each linear layer as
    qweight [K, N/8] int32   8 4-bit weights per int32 along the output dim, AWQ nibble order
    qzeros  [K/G, N/8] int32 zero points, packed the same way
    scales  [K/G, N]         one scale per group of G inputs (G = 128)
and w[k, n] = (q[k, n] - z[k // G, n]) * s[k // G, n].

That layout is bad for small-batch decode: one output column's weights are 4 bits wide and K
rows apart, so a GEMV reads tiny strided pieces or has to split K and reduce. At load time the
weights are repacked once, output-row-major like tandem's BF16 GEMV:
    packed [N, K/8] int32  8 consecutive k of one output row per int32, in AWQ's nibble order
    scales [N, K/G]        float16
    zeros  [N, K/G]        float16 (the integer zero point, exact)
so each output row streams K/2 contiguous bytes. Marlin likewise converts AWQ checkpoints into
its own layout at load time.

Kernels:
  w4_gemv      M <= 8: dequantize on the fly; CUDA-core FMA (M <= 2) or tensor cores (M >= 4)
  w4_dequant   [N, K] float16 for the large-M (prefill) path, followed by cuBLAS
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn

# AWQ packs output columns j = 0..7 of an int32 in nibble order [0, 2, 4, 6, 1, 3, 5, 7]: column j
# lives in nibble REVERSE[j]
AWQ_REVERSE_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


def unpack_awq(packed: torch.Tensor) -> torch.Tensor:
    """[R, C] int32 in AWQ order -> [R, 8 C] int32 values 0..15 (column c*8 + j)."""
    shifts = torch.tensor([4 * i for i in AWQ_REVERSE_ORDER], dtype=torch.int32, device=packed.device)
    return ((packed[:, :, None] >> shifts) & 0xF).reshape(packed.shape[0], -1)


def repack_awq(qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor, group: int,
               dtype: torch.dtype = torch.float16) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """AWQ (qweight [K, N/8], qzeros [K/G, N/8], scales [K/G, N]) -> tandem layout
    (packed [N, K/8] int32, scales [N, K/G], zeros [N, K/G]), on the tensors' device."""
    q = unpack_awq(qweight).t().contiguous()  # [N, K]
    n, k = q.shape
    if k % 8 or k % group or scales.shape != (k // group, n):
        raise ValueError(f"repack_awq: qweight {tuple(qweight.shape)} scales {tuple(scales.shape)} group {group}")
    q = q.view(n, k // 8, 8).to(torch.int64)
    packed = torch.zeros(n, k // 8, dtype=torch.int64, device=q.device)
    for j in range(8):  # k offset j -> nibble REVERSE[j], AWQ's order (LOP3 dequant yields k pairs)
        packed |= q[:, :, j] << (4 * AWQ_REVERSE_ORDER[j])
    packed = packed.to(torch.int32)  # wraps for top nibbles >= 8: same bit pattern
    zeros = unpack_awq(qzeros).t().contiguous().to(dtype)  # [N, K/G]
    return packed, scales.t().contiguous().to(dtype), zeros


@triton.jit
def _unpack_tile(wp, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, UNPACK: tl.constexpr):
    """[BN, BK/8] int32 -> [BN, BK] int32 nibbles (k offset j of each int32 in nibble REVERSE[j]).
    UNPACK 0: replicate each int32 8x with tl.interleave; 1: [BN, BK/8, 8] shifts + reshape."""
    if UNPACK == 0:
        w = tl.interleave(wp, wp)
        w = tl.interleave(w, w)
        w = tl.interleave(w, w)  # each int32 repeated 8 times along k
        j = tl.arange(0, BLOCK_K) % 8
        shifts = (j // 2 + 4 * (j % 2)) * 4  # k offset j sits in nibble REVERSE[j] (AWQ order)
        return (w >> shifts[None, :]) & 0xF
    else:
        j = tl.arange(0, 8)
        shifts = (j // 2 + 4 * (j % 2)) * 4
        q = (wp[:, :, None] >> shifts[None, None, :]) & 0xF
        return tl.reshape(q, (BLOCK_N, BLOCK_K))


@triton.jit
def _nibble_to_float(q):
    """Exact float of a 0..15 nibble, offset by 2^23: OR it into the mantissa of 2^23 (one integer
    op and a bitcast instead of an int-to-float conversion, which runs at a quarter of the FMA rate
    on Ampere); the caller subtracts 2^23 together with the zero point."""
    return (q | 0x4B000000).to(tl.float32, bitcast=True)


@triton.jit
def _w4_gemv_kernel(
    x_ptr, w_ptr, s_ptr, z_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_wn, stride_sn, stride_ym,
    GROUP: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    MMA: tl.constexpr, UNPACK: tl.constexpr, DEFER: tl.constexpr, F16DQ: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_kp = tl.arange(0, BLOCK_K // 8)
    mask_m = offs_m < M
    mask_n = offs_n < N
    NG: tl.constexpr = BLOCK_K // GROUP if BLOCK_K > GROUP else 1  # groups per chunk
    offs_g = tl.arange(0, NG)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_M, BLOCK_N, BLOCK_K) if DEFER else (1, 1, 1), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):  # BLOCK_K divides GROUP or is a multiple of it (checked on the host)
        wp = tl.load(w_ptr + offs_n[:, None] * stride_wn + (k0 // 8 + offs_kp)[None, :], mask=mask_n[:, None],
                     other=0, eviction_policy="evict_first")
        nib = _unpack_tile(wp, BLOCK_N, BLOCK_K, UNPACK)
        if F16DQ:  # float16: OR the nibble into 1024.0 (0x6400) -> 1024 + q exactly
            q = (nib.to(tl.int16) | 0x6400).to(tl.float16, bitcast=True)
        else:
            q = _nibble_to_float(nib)  # [BN, BK], 2^23 + q
        if F16DQ:
            g = k0 // GROUP
            s = tl.load(s_ptr + offs_n * stride_sn + g, mask=mask_n, other=0.0).to(tl.float16)
            z = tl.load(z_ptr + offs_n * stride_sn + g, mask=mask_n, other=0.0).to(tl.float16) + 1024.0
            w = (q - z[:, None].to(tl.float16)) * s[:, None]
        elif NG == 1:
            g = k0 // GROUP
            s = tl.load(s_ptr + offs_n * stride_sn + g, mask=mask_n, other=0.0).to(tl.float32)
            z = tl.load(z_ptr + offs_n * stride_sn + g, mask=mask_n, other=0.0).to(tl.float32) + 8388608.0
            w = (q - z[:, None]) * s[:, None]  # [BN, BK]
        else:  # one scale / zero per GROUP columns of the chunk
            sg_off = offs_n[:, None] * stride_sn + (k0 // GROUP + offs_g)[None, :]
            s = tl.load(s_ptr + sg_off, mask=mask_n[:, None], other=0.0).to(tl.float32)  # [BN, NG]
            z = tl.load(z_ptr + sg_off, mask=mask_n[:, None], other=0.0).to(tl.float32) + 8388608.0
            s = tl.reshape(tl.broadcast_to(s[:, :, None], (BLOCK_N, NG, GROUP)), (BLOCK_N, BLOCK_K))
            z = tl.reshape(tl.broadcast_to(z[:, :, None], (BLOCK_N, NG, GROUP)), (BLOCK_N, BLOCK_K))
            w = (q - z) * s
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :], mask=mask_m[:, None], other=0.0,
                    eviction_policy="evict_last")
        if MMA:  # x padded to BLOCK_M = 16 rows
            acc = tl.dot(x, tl.trans(w.to(x.dtype)), acc)
        elif DEFER:
            acc3 += x.to(tl.float32)[:, None, :] * w.to(tl.float32)[None, :, :]
        else:
            acc += tl.sum(x.to(tl.float32)[:, None, :] * w.to(tl.float32)[None, :, :], axis=2)
    if DEFER:
        acc = tl.sum(acc3, axis=2)
    tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :], acc.to(y_ptr.dtype.element_ty),
             mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _w4_dequant_kernel(
    w_ptr, s_ptr, z_ptr, out_ptr,
    N, K,
    stride_wn, stride_sn, stride_on,
    GROUP: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k0 = pid_k * BLOCK_K
    offs_k = k0 + tl.arange(0, BLOCK_K)
    offs_kp = k0 // 8 + tl.arange(0, BLOCK_K // 8)
    mask_n = offs_n < N
    wp = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_kp[None, :], mask=mask_n[:, None], other=0)
    q = _unpack_tile(wp, BLOCK_N, BLOCK_K, 0).to(tl.float32)
    g = offs_k // GROUP
    s = tl.load(s_ptr + offs_n[:, None] * stride_sn + g[None, :], mask=mask_n[:, None], other=0.0).to(tl.float32)
    z = tl.load(z_ptr + offs_n[:, None] * stride_sn + g[None, :], mask=mask_n[:, None], other=0.0).to(tl.float32)
    w = (q - z) * s
    tl.store(out_ptr + offs_n[:, None] * stride_on + offs_k[None, :], w.to(out_ptr.dtype.element_ty),
             mask=mask_n[:, None])


def w4_gemv(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group: int,
            block_n: int = 2, block_k: int = 128, num_warps: int = 2, num_stages: int = 3, mma: bool = False,
            out: torch.Tensor | None = None, unpack: int = 0, defer: bool = False,
            f16dq: bool = False) -> torch.Tensor:
    """x [M, K] (M <= 8) @ dequant(packed)^T -> [M, N] in x.dtype."""
    m, k = x.shape
    n = packed.shape[0]
    if (m > (16 if mma else 8) or packed.shape[1] * 8 != k or (group % block_k and block_k % group) or block_k % 8
            or k % block_k or x.stride(1) != 1):
        raise ValueError(f"w4_gemv: x {tuple(x.shape)} packed {tuple(packed.shape)} group {group} block_k {block_k}")
    y = out if out is not None else torch.empty((m, n), dtype=x.dtype, device=x.device)
    block_m = 16 if mma else max(1, triton.next_power_of_2(m))
    _w4_gemv_kernel[(triton.cdiv(n, block_n),)](
        x, packed, scales, zeros, y, m, n, k, x.stride(0), packed.stride(0), scales.stride(0), y.stride(0),
        GROUP=group, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, MMA=mma, UNPACK=unpack,
        DEFER=defer and not mma, F16DQ=f16dq and block_k <= group,
        num_warps=num_warps, num_stages=num_stages,
    )
    return y


def w4_dequant(packed: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group: int,
               dtype: torch.dtype = torch.float16, block_n: int = 32, block_k: int = 128) -> torch.Tensor:
    """-> dense [N, K] (nn.Linear layout)."""
    n, k = packed.shape[0], packed.shape[1] * 8
    out = torch.empty((n, k), dtype=dtype, device=packed.device)
    _w4_dequant_kernel[(triton.cdiv(n, block_n), k // block_k)](
        packed, scales, zeros, out, n, k, packed.stride(0), scales.stride(0), out.stride(0),
        GROUP=group, BLOCK_N=block_n, BLOCK_K=block_k, num_warps=4,
    )
    return out


# Config per GPU and (N, K, M bucket): (block_n, block_k, num_warps, num_stages, mma), from
# benchmarks/bench_w4.py. Shapes without an entry use DEFAULT_CONFIG.
# CUDA kernel (tandem/csrc/w4_gemv.cu) per GPU and (N, K, M): (rows per warp, warps per block)
# Measured 2026-11-13 on Qwen3-8B AWQ shapes (fused qkv / o / gate_up / down): 749-860 GB/s at M = 1,
# 648-795 at M = 2, 539-767 at M = 4, 324-465 at M = 8 (CUDA-core FMAs run out at M = 8).
W4_CUDA_CONFIGS: dict[str, dict[tuple[int, int, int], tuple[int, int]]] = {
    "NVIDIA GeForce RTX 3090": {
        (6144, 4096, 1): (2, 8), (6144, 4096, 2): (4, 4), (6144, 4096, 4): (4, 4), (6144, 4096, 8): (4, 4),
        (4096, 4096, 1): (2, 2), (4096, 4096, 2): (2, 2), (4096, 4096, 4): (4, 4), (4096, 4096, 8): (4, 4),
        (24576, 4096, 1): (2, 2), (24576, 4096, 2): (2, 2), (24576, 4096, 4): (4, 4), (24576, 4096, 8): (4, 4),
        (4096, 12288, 1): (2, 2), (4096, 12288, 2): (2, 4), (4096, 12288, 4): (4, 4), (4096, 12288, 8): (4, 4),
    },
}
W4Config = tuple[int, int, int, int, bool]
W4_CONFIGS: dict[str, dict[tuple[int, int, int], W4Config]] = {}
DEFAULT_CONFIG: dict[int, W4Config] = {1: (2, 128, 2, 3, False), 2: (2, 128, 2, 3, False),
                                       4: (16, 128, 4, 3, True), 8: (16, 128, 4, 3, True),
                                       16: (16, 128, 4, 3, True)}


class W4Linear(nn.Module):
    """Linear layer with AWQ 4-bit weights (no bias). Built with the checkpoint's AWQ tensors as
    buffers (qweight / qzeros / scales, filled by the loader, fused along the output dimension for
    qkv / gate_up); ``finalize`` repacks them on the GPU into tandem's layout."""

    def __init__(self, in_features: int, out_features: int, group: int) -> None:
        super().__init__()
        if in_features % group or out_features % 8:
            raise ValueError(f"W4Linear: {in_features} x {out_features} with group {group}")
        self.in_features, self.out_features, self.group = in_features, out_features, group
        self.awq_layout = True
        self.register_buffer("qweight", torch.empty(in_features, out_features // 8, dtype=torch.int32))
        self.register_buffer("qzeros", torch.empty(in_features // group, out_features // 8, dtype=torch.int32))
        self.register_buffer("scales", torch.empty(in_features // group, out_features, dtype=torch.float16))
        self.table: dict[tuple[int, int, int], W4Config] = {}
        self.cuda_table: dict[tuple[int, int, int], tuple[int, int]] = {}

    @torch.no_grad()
    def finalize(self, device_name: str | None = None) -> None:
        packed, scales, zeros = repack_awq(self.qweight, self.qzeros, self.scales, self.group, self.scales.dtype)
        del self._buffers["qweight"], self._buffers["qzeros"]
        self.register_buffer("packed", packed)
        self.register_buffer("scales", scales)
        self.register_buffer("zeros", zeros)
        self.awq_layout = False
        self.table = W4_CONFIGS.get(device_name, {}) if device_name else {}
        self.cuda_table = W4_CUDA_CONFIGS.get(device_name, {}) if device_name else {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = x.shape[0]
        if x.dim() == 2 and m <= 8 and x.is_contiguous() and x.dtype == torch.float16:
            from tandem.layers import w4_cuda

            ext = w4_cuda.load()
            if ext is not None:
                rows, warps = self.cuda_table.get((self.out_features, self.in_features, m), (2, 4))
                return ext.w4_gemv(x, self.packed, self.scales, self.zeros, rows, warps)
        if x.dim() == 2 and m <= 16 and x.is_contiguous():
            mb = 1 if m <= 1 else 2 if m == 2 else 4 if m <= 4 else 8 if m <= 8 else 16
            bn, bk, nw, ns, mma = self.table.get((self.out_features, self.in_features, mb), DEFAULT_CONFIG[mb])
            return w4_gemv(x, self.packed, self.scales, self.zeros, self.group, bn, bk, nw, ns, mma)
        w = w4_dequant(self.packed, self.scales, self.zeros, self.group, x.dtype)
        return F.linear(x, w)
