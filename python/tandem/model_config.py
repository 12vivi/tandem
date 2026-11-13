"""Model configuration parsed from a Hugging Face ``config.json``."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

# architecture -> model family implemented in tandem.models.llama_family
SUPPORTED_ARCHS = {
    "LlamaForCausalLM": "llama",
    "MistralForCausalLM": "llama",
    "Qwen2ForCausalLM": "qwen2",
    "Qwen3ForCausalLM": "qwen3",
    "Qwen3MoeForCausalLM": "qwen3_moe",
}
_MODEL_TYPE_TO_ARCH = {
    "llama": "LlamaForCausalLM",
    "mistral": "MistralForCausalLM",
    "qwen2": "Qwen2ForCausalLM",
    "qwen3": "Qwen3ForCausalLM",
    "qwen3_moe": "Qwen3MoeForCausalLM",
}


@dataclass
class QuantConfig:
    """Weight-only quantization of a checkpoint (``quantization_config`` in config.json)."""

    method: str  # "awq"
    bits: int
    group_size: int

    @classmethod
    def from_dict(cls, d: Optional[dict[str, Any]]) -> Optional["QuantConfig"]:
        if not d:
            return None
        method = d.get("quant_method")
        if (method == "awq" and d.get("bits") == 4 and d.get("zero_point", True)
                and d.get("version", "gemm").lower() == "gemm"):
            return cls("awq", 4, int(d.get("group_size", 128)))
        raise NotImplementedError(f"unsupported quantization_config: {d} (tandem reads 4-bit AWQ 'gemm' checkpoints)")


@dataclass
class ModelConfig:
    architecture: str
    family: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: Optional[dict]
    max_position_embeddings: int
    tie_word_embeddings: bool
    attention_bias: bool
    qk_norm: bool
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = False
    decoder_sparse_step: int = 1
    mlp_only_layers: list[int] = field(default_factory=list)
    eos_token_ids: list[int] = field(default_factory=list)
    torch_dtype: Optional[str] = None
    quantization: Optional[QuantConfig] = None

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    def is_moe_layer(self, layer_id: int) -> bool:
        return (
            self.num_experts > 0
            and layer_id not in self.mlp_only_layers
            and (layer_id + 1) % self.decoder_sparse_step == 0
        )

    @property
    def moe_layer_ids(self) -> list[int]:
        return [i for i in range(self.num_hidden_layers) if self.is_moe_layer(i)]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        arch = (d.get("architectures") or [None])[0] or _MODEL_TYPE_TO_ARCH.get(d.get("model_type", ""))
        if arch not in SUPPORTED_ARCHS:
            raise NotImplementedError(
                f"architecture {arch!r} is not supported yet (supported: {sorted(SUPPORTED_ARCHS)})"
            )
        family = SUPPORTED_ARCHS[arch]
        if d.get("use_sliding_window"):
            raise NotImplementedError("sliding-window attention is not supported yet")

        num_heads = d["num_attention_heads"]
        head_dim = d.get("head_dim") or d["hidden_size"] // num_heads

        # transformers >= 5 stores RoPE settings in `rope_parameters`; older configs use
        # `rope_theta` + `rope_scaling`.
        rope_params = d.get("rope_parameters")
        if isinstance(rope_params, dict) and rope_params:
            rope_theta = float(rope_params.get("rope_theta") or d.get("rope_theta") or 10000.0)
            scaling: Optional[dict] = dict(rope_params)
        else:
            rope_theta = float(d.get("rope_theta") or 10000.0)
            scaling = dict(d["rope_scaling"]) if d.get("rope_scaling") else None
        if scaling is not None and scaling.get("rope_type", scaling.get("type", "default")) == "default":
            scaling = None

        eos = d.get("eos_token_id")
        eos_ids = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))

        return cls(
            architecture=arch,
            family=family,
            vocab_size=d["vocab_size"],
            hidden_size=d["hidden_size"],
            num_hidden_layers=d["num_hidden_layers"],
            num_attention_heads=num_heads,
            num_key_value_heads=d.get("num_key_value_heads") or num_heads,
            head_dim=head_dim,
            intermediate_size=d.get("intermediate_size", 0),
            rms_norm_eps=float(d.get("rms_norm_eps", 1e-6)),
            rope_theta=rope_theta,
            rope_scaling=scaling,
            max_position_embeddings=d.get("max_position_embeddings", 4096),
            tie_word_embeddings=bool(d.get("tie_word_embeddings", False)),
            attention_bias=True if family == "qwen2" else bool(d.get("attention_bias", False)),
            qk_norm=family in ("qwen3", "qwen3_moe"),
            # Hub checkpoints use `num_experts`; transformers >= 5 writes `num_local_experts`
            num_experts=d.get("num_experts") or d.get("num_local_experts") or 0,
            num_experts_per_tok=d.get("num_experts_per_tok", 0) or 0,
            moe_intermediate_size=d.get("moe_intermediate_size", 0) or 0,
            norm_topk_prob=bool(d.get("norm_topk_prob", False)),
            decoder_sparse_step=d.get("decoder_sparse_step", 1) or 1,
            mlp_only_layers=list(d.get("mlp_only_layers") or []),
            eos_token_ids=eos_ids,
            torch_dtype=d.get("torch_dtype") or d.get("dtype"),
            quantization=QuantConfig.from_dict(d.get("quantization_config")),
        )

    @classmethod
    def from_pretrained(cls, path: str) -> "ModelConfig":
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            cfg = cls.from_dict(json.load(f))
        gen_path = os.path.join(path, "generation_config.json")
        if os.path.exists(gen_path):
            with open(gen_path, encoding="utf-8") as f:
                eos = json.load(f).get("eos_token_id")
            if eos is not None:
                extra = [eos] if isinstance(eos, int) else list(eos)
                cfg.eos_token_ids = sorted(set(cfg.eos_token_ids) | set(extra))
        return cfg
