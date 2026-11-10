from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass
class SamplingParams:
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1  # <= 0 disables
    min_p: float = 0.0
    seed: Optional[int] = None
    stop: Union[str, list[str], None] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)
    ignore_eos: bool = False
    skip_special_tokens: bool = True

    def __post_init__(self) -> None:
        if self.stop is None:
            self.stop = []
        elif isinstance(self.stop, str):
            self.stop = [self.stop]
        self.stop_token_ids = list(self.stop_token_ids or [])
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if not 0 <= self.min_p <= 1:
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}")

    @property
    def greedy(self) -> bool:
        return self.temperature < 1e-5
