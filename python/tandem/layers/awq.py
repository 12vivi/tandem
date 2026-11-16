"""4-bit AWQ weights (W4A16, design doc P2 / A2): load-time repack, decode kernels, prefill path.

AutoAWQ "gemm" checkpoints store each linear layer as
    qweight [K, N/8] int32   8 4-bit weights per int32 along the output dim, AWQ nibble order
    qzeros  [K/G, N/8] int32 zero points, packed the same way
    scales  [K/G, N]         one scale per group of G = 128 inputs
and w[k, n] = (q[k, n] - z[k // G, n]) * s[k // G, n].

That layout is bad for small-batch decode: one output's weights are 4 bits wide and K rows apart.
At load time the weights are repacked once, output-row-major (Marlin likewise converts AWQ
checkpoints into its own layout):
    packed [N, K/8] int32, viewed as [N][K/G][4 j][4 cc]: int32 (j, cc) of row n and group g holds
           the 8 consecutive weights k = 128 g + 32 cc + 8 j + 2 q + e (q = 0..3, e = 0..1) in
           nibble q + 4 e
    scales [N, K/G] float16, zeros [N, K/G] float16 (the integer zero point, exact)
The nibble order is the one the LOP3 dequantization (FasterTransformer / Marlin) turns into half2
(w_2q, w_2q+1) with no int-to-float conversion. A lane loads 16 bytes (j, cc = 0..3) = one quant
group of one row, and the 4 lanes j = 0..3 hold 32 consecutive k for each cc: for the tensor-core
kernel that is the A-fragment layout of mma.m16n8k16 up to a k permutation that x shares (see
tandem/csrc/w4_gemv.cu).

Kernels (tandem/csrc/w4_gemv.cu, built at load time): w4_gemv (M <= 4, CUDA cores), w4_mma
(M <= 16, tensor cores) and w4_gemm (M > 16: prefill and large decode batches; tensor cores,
x through a cp.async pipeline, weights dequantized in registers, optional split-K). Where the
benchmark table says so, and on machines without nvcc: dequantize to float16 (Triton) + cuBLAS.
"""

from __future__ import annotations

import os

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


def pack_rows(q: torch.Tensor) -> torch.Tensor:
    """[N, K] values 0..15 -> tandem layout [N, K/8] int32 (see the module docstring)."""
    n, k = q.shape
    v = q.to(torch.int64).view(n, k // 128, 4, 4, 4, 2)  # k = 128 g + 32 cc + 8 j + 2 q + e
    v = v.permute(0, 1, 3, 2, 4, 5)  # -> [n, g, j, cc, q, e]
    shifts = 4 * (torch.arange(4, device=q.device)[:, None] + 4 * torch.arange(2, device=q.device)[None, :])
    packed = (v << shifts).sum(dim=(-1, -2))  # nibbles do not overlap: sum == or
    return packed.reshape(n, k // 8).to(torch.int32)  # wraps for top nibbles >= 8: same bit pattern


def repack_awq(qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor, group: int,
               dtype: torch.dtype = torch.float16) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """AWQ (qweight [K, N/8], qzeros [K/G, N/8], scales [K/G, N]) -> tandem layout
    (packed [N, K/8] int32, scales [N, K/G], zeros [N, K/G]), on the tensors' device."""
    q = unpack_awq(qweight).t().contiguous()  # [N, K]
    n, k = q.shape
    if group != 128 or k % 128 or scales.shape != (k // group, n):
        raise ValueError(f"repack_awq: qweight {tuple(qweight.shape)} scales {tuple(scales.shape)} group {group}")
    zeros = unpack_awq(qzeros).t().contiguous().to(dtype)  # [N, K/G]
    return pack_rows(q), scales.t().contiguous().to(dtype), zeros


def interleave_gate_up(t: torch.Tensor) -> torch.Tensor:
    """Rows [gate (I); up (I)] -> tiles of 16 rows [gate 8i .. 8i+7, up 8i .. 8i+7], the order in
    which the tensor-core kernel's SwiGLU epilogue finds gate and up of a feature in one lane."""
    n = t.shape[0]
    return t.view(2, n // 16, 8, *t.shape[1:]).transpose(0, 1).reshape(t.shape)


def silu_and_mul(y: torch.Tensor) -> torch.Tensor:
    """[M, 2I] = [gate | up] -> silu(gate) * up: FlashInfer's kernel (as ops.silu_and_mul on the
    GPU path), torch without it."""
    try:
        from flashinfer.activation import silu_and_mul as fi_silu_and_mul
    except ImportError:
        gate, up = y.chunk(2, dim=-1)
        return F.silu(gate) * up
    return fi_silu_and_mul(y)


@triton.jit
def _w4_dequant_kernel(
    w_ptr, s_ptr, z_ptr, out_ptr,
    N, K,
    stride_wn, stride_sn, stride_on,
    BLOCK_N: tl.constexpr, DEINTERLEAVE: tl.constexpr,
):
    """One quant group (128 k) of BLOCK_N rows per program -> float16 [N, K]. DEINTERLEAVE: the
    rows are in interleave_gate_up order; write each back to its [gate | up] position."""
    pid_n = tl.program_id(0)
    g = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    cc = tl.arange(0, 4)
    j = tl.arange(0, 4)
    # int32 (j, cc) holds k = 32 cc + 8 j + o of the group: read it as [BN, cc, j] so that the
    # unpacked [BN, cc, j, o] is in natural k order
    wp = tl.load(w_ptr + offs_n[:, None, None] * stride_wn + g * 16 + 4 * j[None, None, :] + cc[None, :, None],
                 mask=mask_n[:, None, None], other=0)
    o = tl.arange(0, 8)  # o = 2 q + e lives in nibble q + 4 e
    shift = 4 * (o // 2 + 4 * (o % 2))
    q = tl.reshape((wp[:, :, :, None] >> shift[None, None, None, :]) & 0xF, (BLOCK_N, 128))
    s = tl.load(s_ptr + offs_n * stride_sn + g, mask=mask_n, other=0.0).to(tl.float32)
    z = tl.load(z_ptr + offs_n * stride_sn + g, mask=mask_n, other=0.0).to(tl.float32)
    w = (q.to(tl.float32) - z[:, None]) * s[:, None]
    kk = g * 128 + tl.arange(0, 128)
    dest = offs_n
    if DEINTERLEAVE:
        r = offs_n % 16
        dest = tl.where(r < 8, (offs_n // 16) * 8 + r, N // 2 + (offs_n // 16) * 8 + r - 8)
    tl.store(out_ptr + dest[:, None] * stride_on + kk[None, :], w.to(out_ptr.dtype.element_ty), mask=mask_n[:, None])


def w4_dequant(packed: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group: int = 128,
               dtype: torch.dtype = torch.float16, block_n: int = 32, deinterleave: bool = False) -> torch.Tensor:
    """-> dense [N, K] (nn.Linear layout); ``deinterleave``: undo interleave_gate_up on the rows."""
    n, k = packed.shape[0], packed.shape[1] * 8
    if deinterleave and (n % 16 or block_n % 16):
        raise ValueError("w4_dequant: deinterleave needs N and block_n multiples of 16")
    out = torch.empty((n, k), dtype=dtype, device=packed.device)
    _w4_dequant_kernel[(triton.cdiv(n, block_n), k // 128)](
        packed, scales, zeros, out, n, k, packed.stride(0), scales.stride(0), out.stride(0),
        BLOCK_N=block_n, DEINTERLEAVE=deinterleave, num_warps=4,
    )
    return out


# Decode kernel per GPU and (N, K, M), the fastest of benchmarks/bench_w4.py:
#   ("gemv", rows per warp, warps per block)                        CUDA cores, M <= 4
#   ("mma", warps per block (split over K), quant groups in flight per lane, row tiles per warp)
# Shapes without an entry use DEFAULT_KERNEL.
W4Kernel = tuple
DEFAULT_KERNEL: W4Kernel = ("mma", 4, 4, 1)
W4_KERNELS: dict[str, dict[tuple[int, int, int], W4Kernel]] = {
    "NVIDIA GeForce RTX 3090": {  # Qwen3-8B-AWQ, benchmarks/results/p2-w4d.jsonl
        (6144, 4096, 1): ("gemv", 1, 4), (6144, 4096, 2): ("mma", 4, 1, 1), (6144, 4096, 4): ("mma", 4, 1, 1),
        (6144, 4096, 8): ("mma", 2, 2, 1), (6144, 4096, 16): ("mma", 4, 1, 1),
        (4096, 4096, 1): ("gemv", 2, 2), (4096, 4096, 2): ("mma", 4, 1, 1), (4096, 4096, 4): ("mma", 4, 1, 1),
        (4096, 4096, 8): ("mma", 4, 1, 1), (4096, 4096, 16): ("mma", 4, 1, 1),
        (24576, 4096, 1): ("gemv", 1, 4), (24576, 4096, 2): ("mma", 4, 4, 1), (24576, 4096, 4): ("mma", 4, 4, 1),
        (24576, 4096, 8): ("mma", 4, 4, 1), (24576, 4096, 16): ("mma", 2, 2, 2),
        (4096, 12288, 1): ("gemv", 2, 2), (4096, 12288, 2): ("mma", 4, 2, 1), (4096, 12288, 4): ("mma", 4, 4, 1),
        (4096, 12288, 8): ("mma", 4, 4, 1), (4096, 12288, 16): ("mma", 4, 1, 1),
        # gate_up with the SwiGLU epilogue (p2-w4e.jsonl); M = 1-8: DEFAULT_KERNEL
        (24576, 4096, 16, "silu"): ("mma", 4, 1, 1),
    },
}


# GEMM for M > 16 (prefill, decode batches over 16) per GPU and (N, K) / (N, K, "silu"):
# ((m_max, config), ...) by increasing m_max, the fastest of benchmarks/bench_w4_gemm.py; M past
# the last m_max uses the last entry. config = (MT, NT, WARPS_N, WARPS_K, STAGES, SPLITS) of w4_gemm
# (tensor cores, weights dequantized in registers; SPLITS: split-K of every tile, or 0: of the
# last, partial wave of tiles only), or None: dequantize to float16 + cuBLAS.
# Shapes without an entry use DEFAULT_GEMM. TANDEM_W4_GEMM=0: always dequantize + cuBLAS.
W4Gemm = tuple
DEFAULT_GEMM: tuple[tuple[int, W4Gemm | None], ...] = ((32, (2, 4, 4, 2, 3, 0)), (1 << 30, (4, 4, 4, 1, 2, 0)))
W4_GEMM_KERNELS: dict[str, dict[tuple, tuple[tuple[int, W4Gemm | None], ...]]] = {
    # Qwen3-8B-AWQ, benchmarks/results/p2-w4g.jsonl (M = 17 .. 96) and p2-w4g2.jsonl (M = 128 ..
    # 2048, with the last-wave split). Each shape keeps one choice for large M, where neighbouring
    # M points flip within +-3% (cuBLAS's own tile choices): the GEMM, except down past M = 1024
    # (dequantize + cuBLAS 2-10% faster there)
    "NVIDIA GeForce RTX 3090": {
        (6144, 4096): ((32, (2, 4, 4, 1, 3, 3)), (64, (4, 4, 4, 1, 2, 3)), (96, (2, 4, 4, 1, 3, 1)),  # qkv
                       (160, (2, 4, 4, 1, 3, 0)), (1 << 30, (4, 4, 4, 1, 2, 0))),
        (4096, 4096): ((17, (2, 4, 4, 2, 3, 2)), (32, (2, 4, 4, 1, 3, 2)), (48, (4, 2, 8, 1, 3, 2)),  # o
                       (64, (4, 4, 4, 1, 2, 2)), (96, (2, 4, 4, 1, 3, 4)), (128, (4, 4, 4, 1, 2, 0)),
                       (192, (2, 4, 4, 1, 3, 0)), (1 << 30, (4, 4, 4, 1, 2, 0))),
        (24576, 4096): ((32, (2, 4, 4, 1, 3, 2)), (64, (4, 4, 4, 1, 2, 2)), (96, (2, 4, 4, 1, 3, 1)),  # gate_up
                        (128, (4, 4, 4, 1, 2, 0)), (160, (2, 4, 4, 1, 3, 0)), (1 << 30, (4, 4, 4, 1, 2, 0))),
        (24576, 4096, "silu"): ((32, (2, 4, 4, 1, 3, 2)), (64, (4, 4, 4, 1, 2, 2)), (96, (2, 4, 4, 1, 3, 1)),
                                (128, (4, 4, 4, 1, 2, 0)), (160, (2, 4, 4, 1, 3, 0)), (1 << 30, (4, 4, 4, 1, 2, 0))),
        (4096, 12288): ((17, (2, 2, 4, 2, 4, 6)), (24, (2, 4, 4, 1, 3, 4)), (32, (2, 2, 4, 2, 4, 1)),  # down
                        (64, (2, 4, 4, 1, 3, 6)), (96, (2, 4, 4, 1, 3, 4)), (128, (4, 4, 4, 1, 2, 0)),
                        (160, (2, 4, 4, 1, 3, 0)), (1024, (4, 4, 4, 1, 2, 0)), (1 << 30, None)),
    },
}
FUSED_GEMM = os.environ.get("TANDEM_W4_GEMM", "1") != "0"


def pick_gemm(table: tuple[tuple[int, W4Gemm | None], ...], m: int) -> W4Gemm | None:
    for m_max, cfg in table:
        if m <= m_max:
            return cfg
    return table[-1][1]


def _cuda_ok(x: torch.Tensor) -> bool:
    return (x.dim() == 2 and x.shape[0] <= 16 and x.dtype == torch.float16 and x.is_contiguous()
            and x.data_ptr() % 16 == 0)


def _gemm_ok(x: torch.Tensor) -> bool:
    return (FUSED_GEMM and x.is_cuda and x.dim() == 2 and x.dtype == torch.float16 and x.is_contiguous()
            and x.data_ptr() % 16 == 0)


class W4Linear(nn.Module):
    """Linear layer with AWQ 4-bit weights (no bias). Built with the checkpoint's AWQ tensors as
    buffers (qweight / qzeros / scales, filled by the loader, fused along the output dimension for
    qkv / gate_up); ``finalize`` repacks them on the GPU into tandem's layout.

    ``enable_silu_mul`` (gate_up of a SwiGLU MLP, before ``finalize``): the layer returns
    silu(gate) * up, [M, out_features / 2], instead of [gate | up]; the tensor-core kernels apply
    it in their epilogue (rows reordered by interleave_gate_up)."""

    def __init__(self, in_features: int, out_features: int, group: int) -> None:
        super().__init__()
        if group != 128 or in_features % group or out_features % 8:
            raise ValueError(f"W4Linear: {in_features} x {out_features} with group {group} (group 128 only)")
        self.in_features, self.out_features, self.group = in_features, out_features, group
        self.awq_layout = True
        self.silu_mul = False
        self.register_buffer("qweight", torch.empty(in_features, out_features // 8, dtype=torch.int32))
        self.register_buffer("qzeros", torch.empty(in_features // group, out_features // 8, dtype=torch.int32))
        self.register_buffer("scales", torch.empty(in_features // group, out_features, dtype=torch.float16))
        self.kernels: dict[tuple, W4Kernel] = {}
        self.gemm_kernels: dict[tuple, tuple[tuple[int, W4Gemm | None], ...]] = {}

    def enable_silu_mul(self) -> bool:
        if not self.awq_layout or self.out_features % 16:
            return False
        self.silu_mul = True
        return True

    @torch.no_grad()
    def finalize(self, device_name: str | None = None) -> None:
        packed, scales, zeros = repack_awq(self.qweight, self.qzeros, self.scales, self.group, torch.float16)
        if self.silu_mul:
            packed, scales, zeros = (interleave_gate_up(t) for t in (packed, scales, zeros))
        del self._buffers["qweight"], self._buffers["qzeros"]
        self.register_buffer("packed", packed)
        self.register_buffer("scales", scales)
        self.register_buffer("zeros", zeros)
        self.awq_layout = False
        self.kernels = W4_KERNELS.get(device_name, {}) if device_name else {}
        self.gemm_kernels = W4_GEMM_KERNELS.get(device_name, {}) if device_name else {}
        from tandem.layers import w4_cuda

        # build now, not inside CUDA graph capture (cached after the first layer); same for the
        # GEMM's split-K workspace
        if w4_cuda.load() is not None and packed.is_cuda:
            w4_cuda.workspace(packed.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _cuda_ok(x):
            from tandem.layers import w4_cuda

            ext = w4_cuda.load()
            if ext is not None:
                m = x.shape[0]
                key = (self.out_features, self.in_features, m)
                if self.silu_mul:  # tensor-core kernel only (the epilogue needs a 16-row tile per lane group)
                    kind, *cfg = self.kernels.get((*key, "silu")) or self.kernels.get(key, DEFAULT_KERNEL)
                    if kind != "mma":
                        _, *cfg = DEFAULT_KERNEL
                    return ext.w4_mma(x, self.packed, self.scales, self.zeros, *cfg, 1)
                kind, *cfg = self.kernels.get(key, DEFAULT_KERNEL)
                if kind == "gemv" and m <= 4:
                    return ext.w4_gemv(x, self.packed, self.scales, self.zeros, *cfg)
                if kind != "mma":
                    _, *cfg = DEFAULT_KERNEL
                return ext.w4_mma(x, self.packed, self.scales, self.zeros, *cfg, 0)
        elif _gemm_ok(x):  # M > 16: the GEMM, unless the table prefers dequantize + cuBLAS
            from tandem.layers import w4_cuda

            ext = w4_cuda.load()
            if ext is not None:
                key = (self.out_features, self.in_features) + (("silu",) if self.silu_mul else ())
                cfg = pick_gemm(self.gemm_kernels.get(key, DEFAULT_GEMM), x.shape[0])
                if cfg is not None:
                    buf, locks = w4_cuda.workspace(x.device)
                    return ext.w4_gemm(x, self.packed, self.scales, self.zeros, buf, locks, *cfg, int(self.silu_mul))
        # no CUDA kernels, or the table's choice: dequantize (gate_up: back in [gate | up] row
        # order), cuBLAS, and the same activation kernel as the unfused path
        w = w4_dequant(self.packed, self.scales, self.zeros, self.group, x.dtype, deinterleave=self.silu_mul)
        y = F.linear(x, w)
        return silu_and_mul(y) if self.silu_mul else y
