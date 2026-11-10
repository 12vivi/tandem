from __future__ import annotations

from torch import nn

from tandem.model_config import ModelConfig


def build_model(cfg: ModelConfig, max_position: int) -> nn.Module:
    """Instantiate an uninitialised model for ``cfg`` (weights are loaded separately)."""
    if cfg.family in ("llama", "qwen2", "qwen3", "qwen3_moe"):
        from tandem.models.llama_family import LlamaFamilyForCausalLM

        return LlamaFamilyForCausalLM(cfg, max_position)
    raise NotImplementedError(f"model family {cfg.family!r}")
