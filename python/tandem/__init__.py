"""Tandem: a single-GPU + CPU heterogeneous LLM inference engine (prototype).

Design document: docs/设计方案.md
"""

from tandem.config import EngineConfig, HybridConfig, SpeculativeConfig
from tandem.sampling_params import SamplingParams

__version__ = "0.2.0.dev0"
__all__ = ["EngineConfig", "HybridConfig", "SpeculativeConfig", "SamplingParams", "LLMEngine"]


def __getattr__(name):  # lazy import keeps `import tandem` cheap
    if name == "LLMEngine":
        from tandem.engine.llm_engine import LLMEngine

        return LLMEngine
    raise AttributeError(name)
