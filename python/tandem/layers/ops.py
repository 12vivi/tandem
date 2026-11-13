"""Small per-token kernels behind one interface: RMSNorm, fused residual-add + RMSNorm,
SiLU-and-mul and RoPE.

``TorchOps`` is the reference: plain PyTorch, any device and dtype, numerically identical to
the Hugging Face implementations (RMSNorm in float32, RoPE with float32 tables). It is what
the CPU path and the numerics tests use. ``FlashInferOps`` swaps in FlashInfer's fused CUDA
kernels (the same ones SGLang runs, design doc §4.3); results differ from the reference only
by BF16 rounding.

The model gets its ops object from ``ForwardContext.ops``, so one process can hold engines
with different backends (the GPU equivalence tests do).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class TorchOps:
    name = "torch"

    @staticmethod
    def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """Normalises over the last dimension (any leading shape)."""
        dtype = x.dtype
        x32 = x.to(torch.float32)
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
        return weight * x32.to(dtype)

    @classmethod
    def fused_add_rms_norm(
        cls, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (rms_norm(x + residual), x + residual). Inputs may be reused by the caller."""
        residual = x + residual
        return cls.rms_norm(residual, weight, eps), residual

    @staticmethod
    def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
        """x: [T, 2I] = [gate | up] -> silu(gate) * up, [T, I]."""
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up

    @staticmethod
    def apply_rope(
        positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor, rope: "torch.nn.Module"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """q: [T, nh, hd], k: [T, nkv, hd]; rotate_half (NeoX) convention, float32 tables."""
        cos = rope.cos_cached[positions].unsqueeze(1)
        sin = rope.sin_cached[positions].unsqueeze(1)

        def rot(x: torch.Tensor) -> torch.Tensor:
            return x * cos.to(x.dtype) + _rotate_half(x) * sin.to(x.dtype)

        return rot(q), rot(k)


class FlashInferOps:
    """FlashInfer kernels (CUDA only). In-place where FlashInfer is; callers treat inputs
    as consumed."""

    name = "flashinfer"

    # residual add + RMSNorm through the Triton kernel up to this many rows (decode batches):
    # 1.2-1.7x faster than FlashInfer's in a CUDA graph at 1-64 rows, hidden 1024-4096
    # (benchmarks/bench_norm.py, p2-norm.jsonl, RTX 3090); larger batches (prefill) keep FlashInfer
    TRITON_NORM_MAX_ROWS = 64

    def __init__(self, fused_prologue: bool = False, small_batch_norm: bool = False) -> None:
        import flashinfer.activation
        import flashinfer.norm
        import flashinfer.rope

        self._norm = flashinfer.norm
        self._act = flashinfer.activation
        self._rope = flashinfer.rope
        self._triton_norm = None
        if small_batch_norm:
            try:
                from tandem.layers.triton_ops import fused_add_rmsnorm
            except ImportError:
                pass
            else:
                self._triton_norm = fused_add_rmsnorm
        # QK-norm + RoPE + KV write as one Triton kernel (tandem.layers.triton_ops)
        self.fused_prologue = False
        if fused_prologue:
            try:
                from tandem.layers.triton_ops import qkv_prologue
            except ImportError as exc:
                import logging

                logging.getLogger(__name__).warning("fused attention prologue unavailable (%s): unfused kernels", exc)
            else:
                self.qkv_prologue = qkv_prologue
                self.fused_prologue = True

    def rms_norm(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        shape = x.shape
        return self._norm.rmsnorm(x.reshape(-1, shape[-1]), weight, eps).view(shape)

    def fused_add_rms_norm(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # in place: residual <- x + residual, x <- rms_norm(residual)
        if (self._triton_norm is not None and x.dim() == 2 and x.shape[0] <= self.TRITON_NORM_MAX_ROWS
                and x.stride(-1) == 1 and residual.stride(-1) == 1):
            self._triton_norm(x, residual, weight, eps)
        else:
            self._norm.fused_add_rmsnorm(x, residual, weight, eps)
        return x, residual

    def silu_and_mul(self, x: torch.Tensor) -> torch.Tensor:
        return self._act.silu_and_mul(x)

    def apply_rope(
        self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor, rope: "torch.nn.Module"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = q.contiguous()
        k = k.contiguous()
        t, hd = q.shape[0], q.shape[-1]
        self._rope.apply_rope_with_cos_sin_cache_inplace(
            positions, q.view(t, -1), k.view(t, -1), hd, rope.cos_sin_cache, is_neox=True
        )
        return q, k


def make_ops(name: str, fused_prologue: bool = False, small_batch_norm: bool = False):
    if name == "torch":
        return TorchOps()
    if name == "flashinfer":
        return FlashInferOps(fused_prologue=fused_prologue, small_batch_norm=small_batch_norm)
    raise ValueError(f"unknown ops backend {name!r}")
