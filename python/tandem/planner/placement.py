"""Placement planner (design doc §3.2 and §4.6 a).

VRAM priority: KV reservation -> attention / norms / embeddings / lm_head -> dense MLPs and
routers -> GPU expert slots (uniform per MoE layer) -> everything else stays on the CPU.
Dense models that do not fit keep their first layers on the GPU (CPU computes the rest at
small batch, or streams it at large batch).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from tandem.model_config import ModelConfig
from tandem.planner.cost_model import ExpertCostModel, stream_crossover_tokens
from tandem.planner.hardware import HardwareProfile

GiB = 2**30


@dataclass
class Footprint:
    """Parameter counts (not bytes)."""

    embed: int
    lm_head: int
    attn_per_layer: list[int]
    mlp_per_layer: list[int]  # dense MLP params (0 for MoE layers)
    router_per_layer: list[int]
    expert: int  # params of one expert
    moe_layers: int
    num_experts: int

    @property
    def total(self) -> int:
        return (self.embed + self.lm_head + sum(self.attn_per_layer) + sum(self.mlp_per_layer)
                + sum(self.router_per_layer) + self.expert * self.num_experts * self.moe_layers)

    @property
    def non_expert(self) -> int:
        return self.total - self.expert * self.num_experts * self.moe_layers


def footprint(cfg: ModelConfig) -> Footprint:
    h, hd = cfg.hidden_size, cfg.head_dim
    q, kv = cfg.num_attention_heads * hd, cfg.num_key_value_heads * hd
    attn = h * q + 2 * h * kv + q * h + 2 * h  # q/k/v/o + two layer norms
    if cfg.attention_bias:
        attn += q + 2 * kv
    if cfg.qk_norm:
        attn += 2 * hd
    attn_l, mlp_l, router_l = [], [], []
    for i in range(cfg.num_hidden_layers):
        attn_l.append(attn)
        if cfg.is_moe_layer(i):
            mlp_l.append(0)
            router_l.append(h * cfg.num_experts)
        else:
            mlp_l.append(3 * h * cfg.intermediate_size)
            router_l.append(0)
    embed = cfg.vocab_size * h
    return Footprint(
        embed=embed + h,  # + final norm
        lm_head=0 if cfg.tie_word_embeddings else cfg.vocab_size * h,
        attn_per_layer=attn_l,
        mlp_per_layer=mlp_l,
        router_per_layer=router_l,
        expert=3 * h * cfg.moe_intermediate_size if cfg.is_moe else 0,
        moe_layers=len(cfg.moe_layer_ids),
        num_experts=cfg.num_experts,
    )


@dataclass
class PlacementPlan:
    mode: str  # "gpu" | "hybrid-moe" | "hybrid-dense" | "cpu"
    model_gb: float
    vram_budget_gb: float
    kv_gb: float
    gpu_weights_gb: float
    cpu_weights_gb: float
    expert_slots_per_layer: Optional[int] = None
    num_experts: int = 0
    gpu_layers: Optional[int] = None
    num_layers: int = 0
    crossover_tokens: float = float("inf")
    notes: list[str] = field(default_factory=list)

    def explain(self) -> str:
        lines = [
            f"mode              : {self.mode}",
            f"model weights     : {self.model_gb:.1f} GB",
            f"VRAM budget       : {self.vram_budget_gb:.1f} GB (KV reserve {self.kv_gb:.1f} GB)",
            f"weights on GPU    : {self.gpu_weights_gb:.1f} GB",
            f"weights on CPU    : {self.cpu_weights_gb:.1f} GB",
        ]
        if self.expert_slots_per_layer is not None:
            lines.append(f"GPU expert slots  : {self.expert_slots_per_layer}/{self.num_experts} per MoE layer")
        if self.gpu_layers is not None:
            lines.append(f"GPU layers        : {self.gpu_layers}/{self.num_layers}")
        if self.crossover_tokens != float("inf"):
            lines.append(f"stream crossover  : T* ~= {self.crossover_tokens:.0f} tokens per expert/operator")
        lines += [f"note: {n}" for n in self.notes]
        return "\n".join(lines)


def make_plan(
    cfg: ModelConfig,
    hw: HardwareProfile,
    bytes_per_param: float = 2.0,
    vram_budget_gb: Optional[float] = None,
    kv_gb: float = 2.0,
    reserve_gb: float = 1.0,
) -> PlacementPlan:
    fp = footprint(cfg)
    b = bytes_per_param
    model_gb = fp.total * b / GiB
    crossover = stream_crossover_tokens(hw, b)
    if hw.gpu is None:
        return PlacementPlan("cpu", model_gb, 0.0, 0.0, 0.0, model_gb, crossover_tokens=float("inf"),
                             notes=["no GPU: everything runs on the CPU executor (C mode)"])

    budget = vram_budget_gb if vram_budget_gb is not None else hw.gpu.vram_gb * 0.92
    avail = budget - kv_gb - reserve_gb
    if fp.total * b / GiB <= avail:
        return PlacementPlan("gpu", model_gb, budget, kv_gb, model_gb, 0.0, crossover_tokens=crossover,
                             notes=["model fits in VRAM (G mode)"])

    if cfg.is_moe:
        dense_gb = fp.non_expert * b / GiB
        if dense_gb <= avail:
            per_slot_gb = fp.expert * fp.moe_layers * b / GiB  # one slot in every MoE layer
            slots = min(fp.num_experts, int((avail - dense_gb) // per_slot_gb))
            gpu_gb = dense_gb + slots * per_slot_gb
            plan = PlacementPlan("hybrid-moe", model_gb, budget, kv_gb, gpu_gb, model_gb - gpu_gb,
                                 expert_slots_per_layer=slots, num_experts=fp.num_experts,
                                 crossover_tokens=crossover)
            cost = ExpertCostModel(hw, fp.expert, b)
            plan.notes.append(
                f"decode: a missing expert costs {cost.t_cpu(1) * 1e3:.2f} ms on CPU vs "
                f"{cost.t_pagein(1) * 1e3:.2f} ms to page in (1 token)"
            )
            plan.notes.append("per step, missing experts with high token load are paged in, the rest run on CPU")
            return plan

    # dense split by layers (also the fallback when MoE non-expert weights alone do not fit)
    fixed = (fp.embed + fp.lm_head) * b / GiB
    per_layer = [(fp.attn_per_layer[i] + fp.mlp_per_layer[i] + fp.router_per_layer[i]
                  + (fp.expert * fp.num_experts if cfg.is_moe_layer(i) else 0)) * b / GiB
                 for i in range(cfg.num_hidden_layers)]
    used, n = fixed, 0
    for size in per_layer:
        if used + size > avail:
            break
        used += size
        n += 1
    return PlacementPlan("hybrid-dense", model_gb, budget, kv_gb, used, model_gb - used,
                         gpu_layers=n, num_layers=cfg.num_hidden_layers, crossover_tokens=crossover,
                         notes=["CPU computes the remaining layers at small batch; streams them when "
                                "tokens per step exceed T*",
                                "large draft trees amortise the CPU-side weight reads (SpecExec)"])
