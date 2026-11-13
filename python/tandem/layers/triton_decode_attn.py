"""Small-batch paged decode attention (design doc P2, A1): split-KV flash decoding in Triton.

At batch 1-8 and 0.5-1K context a Qwen3-0.6B / 1.7B layer spends about 11 us in FlashInfer's
decode attention whatever the context length (benchmarks/bench_decode_attn.py): its planner
splits a 1K context into ~90 chunks of about one page, so the time goes to per-chunk fixed costs
and a 92-way merge, not to reading the 2-4 MB of KV.

Here each program takes one (request, KV head) pair and a contiguous run of pages, keeps the
GQA group's G query heads in registers, and runs an online softmax over its tokens (exp2, float32
accumulation). With NSPLIT > 1 a second small kernel merges the NSPLIT partial results per head.
NSPLIT is a launch constant (fixed per CUDA-graph bucket); pages are spread evenly over the
splits at run time, so the same graph serves any context length.

KV layout: tandem's KVCache, per layer ``[num_blocks * page, num_kv_heads, head_dim]``; the paged
layout is FlashInfer's (kv_indptr, kv_indices, kv_last_page_len), read from the decode wrapper's
buffers that plan() fills every step.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


@triton.jit
def _decode_partial_kernel(
    q_ptr, k_ptr, v_ptr,
    indptr_ptr, indices_ptr, last_len_ptr,
    o_ptr, part_o_ptr, part_ml_ptr, count_ptr,
    qk_scale,  # sm_scale * log2(e)
    stride_qb, stride_qh, stride_kt, stride_kh, stride_ob, stride_oh,
    HKV: tl.constexpr, G: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
    NSPLIT: tl.constexpr, PAGES_PER_ITER: tl.constexpr, FUSED_MERGE: tl.constexpr,
):
    bh = tl.program_id(0)
    split = tl.program_id(1)
    b = bh // HKV
    h = bh % HKV
    p0 = tl.load(indptr_ptr + b)
    p1 = tl.load(indptr_ptr + b + 1)
    npages = p1 - p0
    seq_len = (npages - 1) * PAGE + tl.load(last_len_ptr + b)
    per_split = tl.cdiv(npages, NSPLIT)
    ps = p0 + split * per_split
    pe = tl.minimum(ps + per_split, p1)

    offs_g = tl.arange(0, G)
    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + b * stride_qb + (h * G + offs_g)[:, None] * stride_qh + offs_d[None, :]).to(tl.float32)
    q = q * qk_scale
    m = tl.full((G,), float("-inf"), dtype=tl.float32)
    l = tl.zeros((G,), dtype=tl.float32)
    acc = tl.zeros((G, D), dtype=tl.float32)
    BLOCK_T: tl.constexpr = PAGE * PAGES_PER_ITER
    offs_t = tl.arange(0, BLOCK_T)
    for pg in range(ps, pe, PAGES_PER_ITER):
        page_pos = pg + offs_t // PAGE  # position in the request's page list
        page_ok = page_pos < pe
        page_id = tl.load(indices_ptr + page_pos, mask=page_ok, other=0)
        tok = (page_pos - p0) * PAGE + offs_t % PAGE  # token index in the sequence
        ok = page_ok & (tok < seq_len)
        slot = page_id.to(tl.int64) * PAGE + offs_t % PAGE
        kv_off = slot[:, None] * stride_kt + h * stride_kh + offs_d[None, :]
        k = tl.load(k_ptr + kv_off, mask=ok[:, None], other=0.0).to(tl.float32)  # [T, D]
        v = tl.load(v_ptr + kv_off, mask=ok[:, None], other=0.0).to(tl.float32)  # issued with K
        s = tl.sum(q[:, None, :] * k[None, :, :], axis=2)  # [G, T]
        s = tl.where(ok[None, :], s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, axis=1))  # finite: every iteration has >= 1 valid token
        alpha = tl.exp2(m - m_new)
        p = tl.exp2(s - m_new[:, None])
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * v[None, :, :], axis=1)
        m = m_new

    if NSPLIT == 1:
        o = acc / l[:, None]
        tl.store(o_ptr + b * stride_ob + (h * G + offs_g)[:, None] * stride_oh + offs_d[None, :],
                 o.to(o_ptr.dtype.element_ty))
    else:  # partial result of this split: acc [G, D], (m, l) [G]; an empty split leaves m = -inf, l = 0
        row = (bh * NSPLIT + split) * G + offs_g
        tl.store(part_o_ptr + row[:, None] * D + offs_d[None, :], acc)
        tl.store(part_ml_ptr + row * 2, m)
        tl.store(part_ml_ptr + row * 2 + 1, l)
        if FUSED_MERGE:
            # the last split of this (request, head) to finish merges all of them: all threads'
            # partial stores, a barrier, then one gpu-scope acq_rel atomic (the split-K semaphore
            # pattern of CUTLASS); partials are read back through L2 (.cg), never a stale L1
            tl.debug_barrier()
            done = tl.atomic_add(count_ptr + bh, 1, sem="acq_rel", scope="gpu")
            if done == NSPLIT - 1:
                offs_s = tl.arange(0, NSPLIT)
                rows = (bh * NSPLIT + offs_s[:, None]) * G + offs_g[None, :]  # [S, G]
                pm = tl.load(part_ml_ptr + rows * 2, cache_modifier=".cg")
                pl = tl.load(part_ml_ptr + rows * 2 + 1, cache_modifier=".cg")
                m_max = tl.max(pm, axis=0)
                w = tl.exp2(pm - m_max[None, :])
                denom = tl.sum(pl * w, axis=0)
                pacc = tl.load(part_o_ptr + rows[:, :, None] * D + offs_d[None, None, :], cache_modifier=".cg")
                o = tl.sum(pacc * w[:, :, None], axis=0) / denom[:, None]
                tl.store(o_ptr + b * stride_ob + (h * G + offs_g)[:, None] * stride_oh + offs_d[None, :],
                         o.to(o_ptr.dtype.element_ty))
                tl.store(count_ptr + bh, 0)  # ready for the next launch (CUDA-graph replay)


@triton.jit
def _decode_merge_kernel(
    part_o_ptr, part_ml_ptr, o_ptr,
    stride_ob, stride_oh,
    HKV: tl.constexpr, G: tl.constexpr, D: tl.constexpr, NSPLIT: tl.constexpr,
):
    bh = tl.program_id(0)
    b = bh // HKV
    h = bh % HKV
    offs_s = tl.arange(0, NSPLIT)
    offs_g = tl.arange(0, G)
    offs_d = tl.arange(0, D)
    rows = (bh * NSPLIT + offs_s[:, None]) * G + offs_g[None, :]  # [S, G]
    m = tl.load(part_ml_ptr + rows * 2)
    l = tl.load(part_ml_ptr + rows * 2 + 1)
    m_max = tl.max(m, axis=0)  # [G]
    w = tl.exp2(m - m_max[None, :])  # empty splits: exp2(-inf) = 0
    denom = tl.sum(l * w, axis=0)
    acc = tl.load(part_o_ptr + rows[:, :, None] * D + offs_d[None, None, :])  # [S, G, D]
    o = tl.sum(acc * w[:, :, None], axis=0) / denom[:, None]
    tl.store(o_ptr + b * stride_ob + (h * G + offs_g)[:, None] * stride_oh + offs_d[None, :],
             o.to(o_ptr.dtype.element_ty))


def decode_attention(
    q: torch.Tensor,  # [B, Hq, D]
    k_cache: torch.Tensor,  # [slots, Hkv, D]
    v_cache: torch.Tensor,
    kv_indptr: torch.Tensor,  # [B + 1] int32 (device)
    kv_indices: torch.Tensor,  # [pages] int32 (device)
    kv_last_page_len: torch.Tensor,  # [B] int32 (device)
    page_size: int,
    sm_scale: float,
    nsplit: int = 16,
    pages_per_iter: int = 1,
    num_warps: int = 2,
    num_stages: int = 2,
    out: torch.Tensor | None = None,
    fused_merge: bool = True,
) -> torch.Tensor:
    """Attention of one new token per request over its paged KV -> [B, Hq, D] in q.dtype.
    ``fused_merge``: the last split to finish merges in-kernel (one launch) instead of a second
    merge kernel."""
    bsz, hq, d = q.shape
    hkv = k_cache.shape[1]
    g = hq // hkv
    if hq % hkv or g & (g - 1) or d & (d - 1) or nsplit & (nsplit - 1) or q.stride(2) != 1 or k_cache.stride(2) != 1:
        raise ValueError(f"decode_attention: unsupported q {tuple(q.shape)} kv {tuple(k_cache.shape)} nsplit {nsplit}")
    o = out if out is not None else torch.empty_like(q)
    if nsplit > 1:
        part_o, part_ml, count = _workspace(q.device, bsz * hkv, nsplit, g, d)
    else:
        part_o = part_ml = count = o  # unused
    _decode_partial_kernel[(bsz * hkv, nsplit)](
        q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, o, part_o, part_ml, count,
        sm_scale * LOG2E,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), o.stride(0), o.stride(1),
        HKV=hkv, G=g, D=d, PAGE=page_size, NSPLIT=nsplit, PAGES_PER_ITER=pages_per_iter,
        FUSED_MERGE=fused_merge, num_warps=num_warps, num_stages=num_stages,
    )
    if nsplit > 1 and not fused_merge:
        _decode_merge_kernel[(bsz * hkv,)](
            part_o, part_ml, o, o.stride(0), o.stride(1), HKV=hkv, G=g, D=d, NSPLIT=nsplit, num_warps=4,
        )
    return o


_WORKSPACES: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _workspace(device: torch.device, heads: int, nsplit: int, g: int, d: int):
    """Partial results and per-(request, head) counters, allocated once per shape: the first
    (warm-up) call creates them outside CUDA-graph capture; counters start at 0 and the merging
    program puts them back to 0, so graph replays and consecutive layers can share them."""
    key = (device, heads, nsplit, g, d)
    ws = _WORKSPACES.get(key)
    if ws is None:
        rows = heads * nsplit * g
        ws = _WORKSPACES[key] = (torch.empty((rows, d), dtype=torch.float32, device=device),
                                 torch.empty((rows, 2), dtype=torch.float32, device=device),
                                 torch.zeros(heads, dtype=torch.int32, device=device))
    return ws
