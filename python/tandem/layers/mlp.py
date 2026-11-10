from __future__ import annotations

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


def swiglu(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    """Functional SwiGLU on raw weight tensors (used for GPU-cached expert copies)."""
    return F.linear(F.silu(F.linear(x, w_gate)) * F.linear(x, w_up), w_down)
