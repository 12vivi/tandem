from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


def _inv_freq(head_dim: int, theta: float, rope_scaling: Optional[dict]) -> torch.Tensor:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(torch.float32) / head_dim))
    if rope_scaling is None:
        return inv_freq
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
    if rope_type == "llama3":  # Llama 3.1 frequency-dependent scaling
        factor = rope_scaling["factor"]
        low_freq_factor = rope_scaling["low_freq_factor"]
        high_freq_factor = rope_scaling["high_freq_factor"]
        old_context_len = rope_scaling["original_max_position_embeddings"]
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        wavelen = 2 * math.pi / inv_freq
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed = (1 - smooth) * inv_freq_llama / factor + smooth * inv_freq_llama
        is_medium = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        return torch.where(is_medium, smoothed, inv_freq_llama)
    if rope_type == "linear":
        return inv_freq / rope_scaling["factor"]
    raise NotImplementedError(f"rope_scaling type {rope_type!r} is not supported yet")


class RotaryEmbedding(nn.Module):
    """Precomputed cos/sin tables (float32), applied with the HF ``rotate_half`` convention."""

    def __init__(self, head_dim: int, max_position: int, theta: float, rope_scaling: Optional[dict]) -> None:
        super().__init__()
        inv_freq = _inv_freq(head_dim, theta, rope_scaling)
        t = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[positions], self.sin_cached[positions]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [T, heads, head_dim]; cos/sin: [T, head_dim] (float32)."""
    cos = cos.unsqueeze(1).to(x.dtype)
    sin = sin.unsqueeze(1).to(x.dtype)
    return x * cos + _rotate_half(x) * sin
