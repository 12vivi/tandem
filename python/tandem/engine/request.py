from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from tandem.sampling_params import SamplingParams

if TYPE_CHECKING:
    import torch

    from tandem.cache.radix_cache import RadixNode


class RequestStatus(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED_STOPPED = "stop"
    FINISHED_LENGTH = "length"
    FINISHED_ABORTED = "abort"

    @property
    def finished(self) -> bool:
        return self in (
            RequestStatus.FINISHED_STOPPED,
            RequestStatus.FINISHED_LENGTH,
            RequestStatus.FINISHED_ABORTED,
        )


class Request:
    """One generation request.

    KV bookkeeping follows the vLLM-V1 style "computed tokens" model: ``num_computed_tokens``
    counts tokens whose KV is stored; each step the scheduler hands out tokens until the
    request catches up with ``num_tokens`` (+ draft tokens when speculating).
    """

    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        eos_token_ids: Optional[list[int]] = None,
        prompt: Optional[str] = None,
        arrival_time: Optional[float] = None,
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = list(prompt_token_ids)
        self.sampling_params = sampling_params
        self.eos_token_ids = set() if sampling_params.ignore_eos else set(eos_token_ids or [])
        self.stop_token_ids = set(sampling_params.stop_token_ids)
        self.output_token_ids: list[int] = []
        self.status = RequestStatus.WAITING

        # KV state
        self.num_computed_tokens = 0
        self.block_ids: list[int] = []
        self.prefix_node: Optional["RadixNode"] = None
        self.num_cached_tokens = 0  # prefix-cache hit at the last admission

        # speculative decoding
        self.spec_token_ids: list[int] = []
        self.num_draft_tokens = 0
        self.num_accepted_tokens = 0

        # sampling RNG (only when the request carries a seed)
        self.generator: Optional["torch.Generator"] = None

        # detokenization state
        self.output_text = ""
        self.sent_text_len = 0
        self.new_token_ids: list[int] = []  # tokens appended in the latest step

        # metrics
        self.arrival_time = arrival_time if arrival_time is not None else time.monotonic()
        self.first_token_time: Optional[float] = None
        self.finish_time: Optional[float] = None
        self.num_preemptions = 0

    # ----------------------------------------------------------------- properties
    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return self.num_tokens + len(self.spec_token_ids)

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    def token_ids_in_range(self, start: int, end: int) -> list[int]:
        """Token ids for positions [start, end), including draft tokens past ``num_tokens``."""
        tokens = self.all_token_ids + self.spec_token_ids
        return tokens[start:end]

    @property
    def is_decoding(self) -> bool:
        return self.num_computed_tokens == self.num_tokens - 1 and self.num_output_tokens > 0

    @property
    def finished(self) -> bool:
        return self.status.finished

    @property
    def finish_reason(self) -> Optional[str]:
        return self.status.value if self.status.finished else None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Request({self.request_id}, status={self.status.name}, prompt={self.num_prompt_tokens}, "
            f"out={self.num_output_tokens}, computed={self.num_computed_tokens}, blocks={len(self.block_ids)})"
        )


@dataclass
class RequestOutput:
    request_id: str
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    text: str
    delta_text: str
    delta_token_ids: list[int]
    finished: bool
    finish_reason: Optional[str]
    metrics: dict = field(default_factory=dict)
