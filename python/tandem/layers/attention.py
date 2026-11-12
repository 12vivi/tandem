from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from tandem.executor.forward_context import ForwardContext, build_attention_metadata
from tandem.layers.norm import RMSNorm
from tandem.model_config import ModelConfig


class TorchAttentionBackend:
    """Reference paged attention: write new K/V into slots, gather context, run SDPA.

    Device-agnostic (CPU and CUDA) and exact; it is the correctness baseline. The GPU fast
    path (``tandem.layers.flashinfer_attention.FlashInferAttentionBackend``) implements the
    same two methods.
    """

    name = "torch"

    def prepare(
        self, seqs: Sequence[tuple[Sequence[int], int, int]], block_size: int, device: torch.device
    ) -> tuple[torch.Tensor, object]:
        """seqs: (block_ids, start, num_new) per request -> (positions [T], metadata)."""
        return build_attention_metadata(seqs, block_size, device)

    def forward(
        self,
        layer_id: int,
        q: torch.Tensor,  # [T, nh, hd]
        k: torch.Tensor,  # [T, nkv, hd]
        v: torch.Tensor,  # [T, nkv, hd]
        ctx: ForwardContext,
        scale: float,
    ) -> torch.Tensor:
        meta = ctx.attn
        k_cache = ctx.kv_cache.k[layer_id]
        v_cache = ctx.kv_cache.v[layer_id]
        k_cache.index_copy_(0, meta.slot_mapping, k.to(k_cache.dtype))
        v_cache.index_copy_(0, meta.slot_mapping, v.to(v_cache.dtype))

        qp = q[meta.q_index]  # [B, Qmax, nh, hd]
        kp = k_cache[meta.kv_slots].to(q.dtype)  # [B, Lmax, nkv, hd]
        vp = v_cache[meta.kv_slots].to(q.dtype)
        groups = q.shape[1] // k.shape[1]
        if groups > 1:
            kp = kp.repeat_interleave(groups, dim=2)
            vp = vp.repeat_interleave(groups, dim=2)
        out = F.scaled_dot_product_attention(
            qp.transpose(1, 2), kp.transpose(1, 2), vp.transpose(1, 2), attn_mask=meta.attn_mask, scale=scale
        )  # [B, nh, Qmax, hd]
        return out.transpose(1, 2)[meta.q_valid]  # [T, nh, hd]


class Attention(nn.Module):
    """GQA attention with q, k and v projected by one fused GEMM (``qkv_proj``, rows ordered
    [q | k | v]); the loader fills it from the checkpoint's q_proj / k_proj / v_proj."""

    def __init__(self, cfg: ModelConfig, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scale = cfg.head_dim**-0.5
        h = cfg.hidden_size
        self.qkv_proj = nn.Linear(h, self.q_size + 2 * self.kv_size, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(self.q_size, h, bias=False)
        self.qk_norm = cfg.qk_norm
        if cfg.qk_norm:  # Qwen3: per-head RMSNorm on q and k before RoPE
            self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)

    def forward(self, hidden: torch.Tensor, rope: nn.Module, ctx: ForwardContext) -> torch.Tensor:
        t = hidden.shape[0]
        ops = ctx.ops
        qkv = self.qkv_proj(hidden)
        if getattr(ops, "fused_prologue", False) and hasattr(ctx.attn_backend, "attend"):
            # one kernel: QK-norm + RoPE -> q, and k / v straight into the KV cache
            q = ops.qkv_prologue(
                qkv, ctx.positions, ctx.attn.slot_mapping, ctx.kv_cache.k[self.layer_id], ctx.kv_cache.v[self.layer_id],
                rope.cos_sin_cache, self.num_heads, self.num_kv_heads, self.head_dim,
                self.q_norm.weight if self.qk_norm else None, self.k_norm.weight if self.qk_norm else None,
                self.q_norm.eps if self.qk_norm else 1e-6,
            )
            o = ctx.attn_backend.attend(self.layer_id, q, ctx)
            return self.o_proj(o.reshape(t, self.q_size))
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(t, self.num_heads, self.head_dim)
        k = k.view(t, self.num_kv_heads, self.head_dim)
        v = v.view(t, self.num_kv_heads, self.head_dim)
        if self.qk_norm:
            q = ops.rms_norm(q, self.q_norm.weight, self.q_norm.eps)
            k = ops.rms_norm(k, self.k_norm.weight, self.k_norm.eps)
        q, k = ops.apply_rope(ctx.positions, q, k, rope)
        o = ctx.attn_backend.forward(self.layer_id, q, k, v, ctx, self.scale)
        return self.o_proj(o.reshape(t, self.q_size))
