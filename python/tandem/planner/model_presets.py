"""config.json subsets of public models, for planning without downloading anything.

Values mirror the Hugging Face configs; total parameter counts were cross-checked
(Qwen3-30B-A3B ~30.5B / ~3B active, Qwen3-235B-A22B ~235B / ~22B active,
Qwen3-8B ~8.2B, Llama-3.3-70B ~70.6B).
"""

from __future__ import annotations

_QWEN3_COMMON = {"vocab_size": 151936, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "head_dim": 128,
                 "max_position_embeddings": 40960, "tie_word_embeddings": False}

MODEL_PRESETS: dict[str, dict] = {
    "Qwen3-8B": {
        "architectures": ["Qwen3ForCausalLM"], **_QWEN3_COMMON,
        "hidden_size": 4096, "num_hidden_layers": 36, "num_attention_heads": 32,
        "num_key_value_heads": 8, "intermediate_size": 12288,
    },
    "Qwen3-30B-A3B": {
        "architectures": ["Qwen3MoeForCausalLM"], **_QWEN3_COMMON,
        "hidden_size": 2048, "num_hidden_layers": 48, "num_attention_heads": 32,
        "num_key_value_heads": 4, "intermediate_size": 6144, "moe_intermediate_size": 768,
        "num_experts": 128, "num_experts_per_tok": 8, "norm_topk_prob": True,
    },
    "Qwen3-235B-A22B": {
        "architectures": ["Qwen3MoeForCausalLM"], **_QWEN3_COMMON,
        "hidden_size": 4096, "num_hidden_layers": 94, "num_attention_heads": 64,
        "num_key_value_heads": 4, "intermediate_size": 12288, "moe_intermediate_size": 1536,
        "num_experts": 128, "num_experts_per_tok": 8, "norm_topk_prob": True,
    },
    "Llama-3.3-70B": {
        "architectures": ["LlamaForCausalLM"], "vocab_size": 128256, "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0, "max_position_embeddings": 131072, "tie_word_embeddings": False,
        "hidden_size": 8192, "num_hidden_layers": 80, "num_attention_heads": 64,
        "num_key_value_heads": 8, "intermediate_size": 28672,
        "rope_scaling": {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                         "high_freq_factor": 4.0, "original_max_position_embeddings": 8192},
    },
}
