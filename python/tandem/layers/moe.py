"""Sparse MoE block (Qwen3-MoE style routing) with pluggable expert executors.

Routing math follows the Hugging Face Qwen3-MoE reference: float32 softmax over the router
logits, top-k, optional renormalisation, cast back to the activation dtype.

Where each expert runs is decided by the executor in ``ForwardContext.moe_executor``:
``LocalMoEExecutor`` (all experts on the model device) or
``tandem.hybrid.moe_executor.HybridMoEExecutor`` (GPU-resident / page-in / CPU).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tandem.executor.forward_context import ForwardContext
from tandem.layers.mlp import GatedMLP
from tandem.model_config import ModelConfig


class SparseMoeBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.num_experts = cfg.num_experts
        self.top_k = cfg.num_experts_per_tok
        self.norm_topk_prob = cfg.norm_topk_prob
        self.gate = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [GatedMLP(cfg.hidden_size, cfg.moe_intermediate_size) for _ in range(cfg.num_experts)]
        )

    def route(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(hidden)
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        weights, ids = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return ids, weights.to(hidden.dtype)

    def forward(self, hidden: torch.Tensor, ctx: ForwardContext) -> torch.Tensor:
        ids, weights = self.route(hidden)
        return ctx.moe_executor.run(self.layer_id, hidden, ids, weights, self)


class LocalMoEExecutor:
    """All experts live on the model's device; experts run in ascending id order."""

    def run(
        self,
        layer_id: int,
        hidden: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        block: SparseMoeBlock,
    ) -> torch.Tensor:
        out = torch.zeros_like(hidden)
        for e in torch.unique(ids).tolist():
            tok, slot = torch.where(ids == e)
            y = block.experts[e](hidden[tok]) * weights[tok, slot, None]
            out.index_add_(0, tok, y.to(out.dtype))
        return out

    def stats(self) -> dict:
        return {}
