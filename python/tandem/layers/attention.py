from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tandem.executor.forward_context import ForwardContext
from tandem.layers.norm import RMSNorm
from tandem.layers.rotary import apply_rotary
from tandem.model_config import ModelConfig


class TorchAttentionBackend:
    """Reference paged attention: write new K/V into slots, gather context, run SDPA.

    Device-agnostic (CPU and CUDA) and exact; it is the correctness baseline. The GPU
    fast path (FlashInfer paged decode/prefill + CUDA Graph, design doc §4.4) replaces
    this class behind the same ``forward`` signature.
    """

    name = "torch"

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
    def __init__(self, cfg: ModelConfig, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.scale = cfg.head_dim**-0.5
        h = cfg.hidden_size
        self.q_proj = nn.Linear(h, self.num_heads * self.head_dim, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, h, bias=False)
        self.qk_norm = cfg.qk_norm
        if cfg.qk_norm:  # Qwen3: per-head RMSNorm on q and k before RoPE
            self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)

    def forward(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, ctx: ForwardContext) -> torch.Tensor:
        t = hidden.shape[0]
        q = self.q_proj(hidden).view(t, self.num_heads, self.head_dim)
        k = self.k_proj(hidden).view(t, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden).view(t, self.num_kv_heads, self.head_dim)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        o = ctx.attn_backend.forward(self.layer_id, q, k, v, ctx, self.scale)
        return self.o_proj(o.reshape(t, self.num_heads * self.head_dim))
