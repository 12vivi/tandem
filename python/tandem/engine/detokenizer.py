"""Incremental detokenization and stop strings, shared by the in-process engine and the API
process of the multi-process server.

Each step decodes only a short window of output tokens: ``[prefix_offset:end]`` and
``[prefix_offset:read_offset]``, and appends what the first adds beyond the second (vLLM's
``detokenize_incrementally``). Text that ends in an incomplete UTF-8 sequence is held back
until it completes or the request finishes. ``state`` is anything with the attributes
``detok_prefix_offset``, ``detok_read_offset``, ``output_text`` and ``sent_text_len``
(``tandem.engine.request.Request`` or ``DetokState``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class DetokState:
    detok_prefix_offset: int = 0
    detok_read_offset: int = 0
    output_text: str = ""
    sent_text_len: int = 0
    tokens: list[int] = field(default_factory=list)


def detokenize_step(
    tokenizer: Any,
    state: Any,
    tokens: Sequence[int],
    end: int,
    finished: bool,
    skip_special_tokens: bool,
    stops: Sequence[str],
) -> tuple[str, bool]:
    """-> (new text to emit, whether a stop string ended the output)."""
    prefix = tokenizer.decode(tokens[state.detok_prefix_offset : state.detok_read_offset],
                              skip_special_tokens=skip_special_tokens)
    window = tokenizer.decode(tokens[state.detok_prefix_offset : end], skip_special_tokens=skip_special_tokens)
    text = state.output_text
    if len(window) > len(prefix) and (finished or not window.endswith("�")):
        text = text + window[len(prefix) :]
        state.detok_prefix_offset, state.detok_read_offset = state.detok_read_offset, end
    stopped = False
    if stops:
        cut = min((pos for pos in (text.find(s) for s in stops) if pos >= 0), default=-1)
        if cut >= 0:
            text, stopped = text[:cut], True
    state.output_text = text
    delta = text[state.sent_text_len :]
    state.sent_text_len = len(text)
    return delta, stopped
