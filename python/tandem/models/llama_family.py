"""Llama / Qwen2 / Qwen3 / Qwen3-MoE decoder.

Parameter names mirror the Hugging Face checkpoints (``model.layers.N.self_attn.q_proj``,
``model.layers.N.mlp.experts.E.gate_proj`` ...), so safetensors load without renaming.
Inputs are flattened over the batch (``[num_tokens]``), as in vLLM/SGLang.
"""

from __future__ import annotations

import torch
from torch import nn

from tandem.executor.forward_context import ForwardContext
from tandem.layers.attention import Attention
from tandem.layers.mlp import GatedMLP
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
            SparseMoeBlock(cfg, layer_id) if self.is_moe else GatedMLP(cfg.hidden_size, cfg.intermediate_size)
        )
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, ctx: ForwardContext) -> torch.Tensor:
        residual = hidden
        hidden = residual + self.self_attn(self.input_layernorm(hidden), cos, sin, ctx)
        residual = hidden
        h = self.post_attention_layernorm(hidden)
        h = self.mlp(h, ctx) if self.is_moe else self.mlp(h)
        return residual + h


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
        hidden = self.model.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(positions)
        for layer in self.model.layers:
            hidden = layer(hidden, cos, sin, ctx)
        return self.model.norm(hidden)

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        weight = self.model.embed_tokens.weight if self.lm_head is None else self.lm_head.weight
        logits = torch.nn.functional.linear(hidden, weight)
        return logits if logits.dtype == torch.float64 else logits.float()

    def moe_blocks(self) -> dict[int, SparseMoeBlock]:
        return {i: layer.mlp for i, layer in enumerate(self.model.layers) if layer.is_moe}
