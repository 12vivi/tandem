from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn


class GatedMLP(nn.Module):
    """SwiGLU MLP: down(silu(gate(x)) * up(x)). Also used as a single MoE expert."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MergedGatedMLP(nn.Module):
    """Dense SwiGLU MLP with gate and up fused into one GEMM ([gate | up] rows), as vLLM and
    SGLang run it: one launch instead of two, and the same GEMM shape (so the same cuBLAS
    kernel) as theirs. The loader fills ``gate_up_proj`` from the checkpoint's separate
    ``gate_proj`` / ``up_proj`` tensors."""

    def __init__(self, hidden_size: int, intermediate_size: int, quant=None) -> None:
        super().__init__()
        from tandem.layers.linear import make_linear

        self.intermediate_size = intermediate_size
        self.gate_up_proj = make_linear(hidden_size, 2 * intermediate_size, False, quant)
        self.down_proj = make_linear(intermediate_size, hidden_size, False, quant)
        # AWQ: the gate_up kernel applies SwiGLU in its epilogue (tandem.layers.awq.W4Linear);
        # TANDEM_W4_FUSE_ACT=0 keeps the separate activation kernel (A/B comparisons)
        enable = getattr(self.gate_up_proj, "enable_silu_mul", None)
        self.fused_act = bool(enable is not None and os.environ.get("TANDEM_W4_FUSE_ACT", "1") != "0" and enable())

    def forward(self, x: torch.Tensor, ops) -> torch.Tensor:
        if self.fused_act:
            return self.down_proj(self.gate_up_proj(x))
        return self.down_proj(ops.silu_and_mul(self.gate_up_proj(x)))


def swiglu(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    """Functional SwiGLU on raw weight tensors (used for GPU-cached expert copies)."""
    return F.linear(F.silu(F.linear(x, w_gate)) * F.linear(x, w_up), w_down)
