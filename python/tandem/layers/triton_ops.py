"""Fused attention prologue (design doc §4.3): QK-norm + RoPE + KV-cache write in one kernel.

Reads the fused ``qkv_proj`` output ``[T, (nh + 2 * nkv) * hd]`` directly and, per (token, head):
  q heads  RMSNorm (Qwen3 QK-norm) -> RoPE (NeoX halves) -> q_out [T, nh, hd]
  k heads  RMSNorm -> RoPE -> k_cache[slot]
  v heads  -> v_cache[slot]
One launch replaces seven on the unfused FlashInfer path (two copies out of the strided qkv
view, two RMSNorms, RoPE, two index_copy). Math in float32 with float32 cos/sin tables, one
rounding at the end, like FlashInfer's own fused kernels.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _qkv_prologue_kernel(
    qkv_ptr, qkv_stride,
    q_out_ptr,
    k_cache_ptr, v_cache_ptr,
    slot_ptr, pos_ptr,
    q_w_ptr, k_w_ptr,
    cos_sin_ptr,
    eps,
    NH: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr, HAS_NORM: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    HALF: tl.constexpr = HD // 2
    offs = tl.arange(0, HALF)
    row = qkv_ptr + t * qkv_stride + h * HD
    x1 = tl.load(row + offs).to(tl.float32)
    x2 = tl.load(row + HALF + offs).to(tl.float32)
    slot = tl.load(slot_ptr + t)
    if h >= NH + NKV:  # v head: straight into the cache
        dst = v_cache_ptr + (slot * NKV + (h - NH - NKV)) * HD
        tl.store(dst + offs, x1.to(v_cache_ptr.dtype.element_ty))
        tl.store(dst + HALF + offs, x2.to(v_cache_ptr.dtype.element_ty))
    else:
        if HAS_NORM:
            rstd = tl.rsqrt((tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)) / HD + eps)
            if h < NH:
                w1 = tl.load(q_w_ptr + offs).to(tl.float32)
                w2 = tl.load(q_w_ptr + HALF + offs).to(tl.float32)
            else:
                w1 = tl.load(k_w_ptr + offs).to(tl.float32)
                w2 = tl.load(k_w_ptr + HALF + offs).to(tl.float32)
            x1 = x1 * rstd * w1
            x2 = x2 * rstd * w2
        pos = tl.load(pos_ptr + t)
        cos = tl.load(cos_sin_ptr + pos * HD + offs)
        sin = tl.load(cos_sin_ptr + pos * HD + HALF + offs)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        if h < NH:
            dst = q_out_ptr + (t * NH + h) * HD
            tl.store(dst + offs, o1.to(q_out_ptr.dtype.element_ty))
            tl.store(dst + HALF + offs, o2.to(q_out_ptr.dtype.element_ty))
        else:
            dst = k_cache_ptr + (slot * NKV + (h - NH)) * HD
            tl.store(dst + offs, o1.to(k_cache_ptr.dtype.element_ty))
            tl.store(dst + HALF + offs, o2.to(k_cache_ptr.dtype.element_ty))


def qkv_prologue(
    qkv: torch.Tensor,  # [T, (nh + 2 * nkv) * hd], row-contiguous
    positions: torch.Tensor,  # [T] int64
    slot_mapping: torch.Tensor,  # [T] int64
    k_cache: torch.Tensor,  # [slots, nkv, hd]
    v_cache: torch.Tensor,  # [slots, nkv, hd]
    cos_sin_cache: torch.Tensor,  # [max_pos, hd] float32: [cos | sin] of the half dims
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_norm_weight: torch.Tensor = None,
    k_norm_weight: torch.Tensor = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """-> q [T, nh, hd]; k and v are written into the caches at ``slot_mapping``."""
    t = qkv.shape[0]
    q = torch.empty((t, num_heads, head_dim), dtype=qkv.dtype, device=qkv.device)
    if t == 0:
        return q
    has_norm = q_norm_weight is not None
    dummy = cos_sin_cache  # any tensor: unused weight pointers when there is no QK-norm
    _qkv_prologue_kernel[(t, num_heads + 2 * num_kv_heads)](
        qkv, qkv.stride(0), q, k_cache, v_cache, slot_mapping, positions,
        q_norm_weight if has_norm else dummy, k_norm_weight if has_norm else dummy,
        cos_sin_cache, eps,
        NH=num_heads, NKV=num_kv_heads, HD=head_dim, HAS_NORM=has_norm,
        num_warps=1,
    )
    return q
