"""Llama / Qwen2 / Qwen3 / Qwen3-MoE decoder.

Parameter names follow the Hugging Face checkpoints except for two fused projections that
the loader assembles: ``self_attn.qkv_proj`` (from q_proj / k_proj / v_proj) and, in dense
layers, ``mlp.gate_up_proj`` (from gate_proj / up_proj). MoE experts keep their own names
(``mlp.experts.E.gate_proj`` ...). Inputs are flattened over the batch (``[num_tokens]``),
as in vLLM/SGLang.

The residual stream is carried between layers (vLLM / SGLang style) so that every
"add residual, then normalise" pair is one fused kernel on the GPU path:
    layer i:  x, r = add_rms_norm(x, r)   # r <- x + r; x <- norm(r)   (plain norm for layer 0)
              x = attn(x)
              x, r = add_rms_norm(x, r)
              x = mlp(x)
    final:    x, _ = add_rms_norm(x, r)
This is the same arithmetic as the reference ``residual + sublayer(norm(x))`` sequence.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from tandem.executor.forward_context import ForwardContext
from tandem.layers.attention import Attention
from tandem.layers.mlp import MergedGatedMLP
from tandem.layers.moe import SparseMoeBlock
from tandem.layers.norm import RMSNorm
from tandem.layers.rotary import RotaryEmbedding
from tandem.model_config import ModelConfig


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_id: int) -> None:
        super().__init__()
        self.self_attn = Attention(cfg, layer_id)
        self.is_moe = cfg.is_moe_layer(layer_id)
        self.mlp: nn.Module = (
            SparseMoeBlock(cfg, layer_id) if self.is_moe else MergedGatedMLP(cfg.hidden_size, cfg.intermediate_size)
        )
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        residual: Optional[torch.Tensor],
        rope: nn.Module,
        ctx: ForwardContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ops = ctx.ops
        n1, n2 = self.input_layernorm, self.post_attention_layernorm
        if residual is None:
            residual = hidden
            hidden = ops.rms_norm(hidden, n1.weight, n1.eps)
        else:
            hidden, residual = ops.fused_add_rms_norm(hidden, residual, n1.weight, n1.eps)
        hidden = self.self_attn(hidden, rope, ctx)
        hidden, residual = ops.fused_add_rms_norm(hidden, residual, n2.weight, n2.eps)
        hidden = self.mlp(hidden, ctx) if self.is_moe else self.mlp(hidden, ops)
        return hidden, residual


class LlamaFamilyModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)


class LlamaFamilyForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig, max_position: int) -> None:
        super().__init__()
        self.config = cfg
        self.model = LlamaFamilyModel(cfg)
        self.rotary_emb = RotaryEmbedding(cfg.head_dim, max_position, cfg.rope_theta, cfg.rope_scaling)
        self.lm_head = None if cfg.tie_word_embeddings else nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, ctx: ForwardContext) -> torch.Tensor:
        ctx.positions = positions
        hidden = self.model.embed_tokens(input_ids)
        residual = None
        for layer in self.model.layers:
            hidden, residual = layer(hidden, residual, self.rotary_emb, ctx)
        norm = self.model.norm
        hidden, _ = ctx.ops.fused_add_rms_norm(hidden, residual, norm.weight, norm.eps)
        return hidden

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        weight = self.model.embed_tokens.weight if self.lm_head is None else self.lm_head.weight
        logits = torch.nn.functional.linear(hidden, weight)
        return logits if logits.dtype == torch.float64 else logits.float()

    def moe_blocks(self) -> dict[int, SparseMoeBlock]:
        return {i: layer.mlp for i, layer in enumerate(self.model.layers) if layer.is_moe}
