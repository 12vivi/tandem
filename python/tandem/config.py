from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SpeculativeConfig:
    """Draft-and-verify speculative decoding (design doc §4.7).

    Only model-free n-gram drafting is implemented in the prototype; model drafters
    (EAGLE-3 / DFlash / MTP) plug into the same verify path later.
    """

    method: str = "none"  # "none" | "ngram"
    num_speculative_tokens: int = 4
    ngram_min: int = 2
    ngram_max: int = 4

    @property
    def enabled(self) -> bool:
        return self.method != "none" and self.num_speculative_tokens > 0


@dataclass
class HybridConfig:
    """Three-path MoE expert execution: GPU-resident / page-in / CPU compute (§4.6)."""

    enabled: bool = False
    gpu_expert_slots: Optional[int] = None  # per MoE layer; None -> from the placement plan
    vram_budget_gb: Optional[float] = None  # planner input when gpu_expert_slots is None
    hw_profile: str = "probe"  # preset name, path to a JSON profile, or "probe"
    cpu_threads: int = 1  # worker threads that run CPU experts
    reuse_bonus: float = 0.5  # discount on page-in cost for expected future cache hits


@dataclass
class EngineConfig:
    model: str
    tokenizer: Optional[str] = None
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "cuda:N"
    dtype: str = "auto"  # "auto" | "float32" | "bfloat16" | "float16" | "float64"
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 2048  # per-step token budget (Sarathi-Serve chunked prefill)
    schedule_policy: str = "decode_first"  # "decode_first" (Sarathi-Serve) | "prefill_first" (SGLang)
    max_model_len: Optional[int] = None
    block_size: int = 16
    num_kv_blocks: Optional[int] = None
    kv_cache_memory_gb: float = 1.0
    enable_prefix_caching: bool = True
    seed: int = 0
    cpu_threads: int = 0  # torch intra-op threads; 0 keeps the torch default
    # "auto": FlashInfer kernels (attention, norms, SiLU-and-mul, RoPE) on CUDA with
    # float16 / bfloat16 when flashinfer is installed, the PyTorch reference path otherwise
    attention_backend: str = "auto"  # "auto" | "torch" | "flashinfer"
    cuda_graph: bool = True  # decode steps replay CUDA graphs (FlashInfer backend, dense models)
    cuda_graph_max_bs: int = 64  # largest captured decode batch (capped at max_num_seqs)
    # launch step N before step N-1's tokens reach the host (not with speculative decoding)
    overlap_schedule: bool = True
    # FlashInfer path: QK-norm + RoPE + KV write as one Triton kernel (tandem.layers.triton_ops);
    # falls back to the unfused kernels when Triton is missing
    fused_prologue: bool = True
    speculative: SpeculativeConfig = field(default_factory=SpeculativeConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
