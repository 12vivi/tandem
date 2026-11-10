"""Model-free n-gram drafting (prompt lookup decoding).

Find the most recent earlier occurrence of the context's last n tokens (n from ``n_max``
down to ``n_min``) and propose the tokens that followed it. Costs no model compute,
which matters most where compute is scarce (CPU-only and offloaded execution, §4.7).
Successor plan: suffix-tree drafting (SuffixDecoding, NeurIPS'25) over the radix cache.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class NgramProposer:
    def __init__(self, n_min: int = 2, n_max: int = 4, max_context: int = 4096) -> None:
        if not 1 <= n_min <= n_max:
            raise ValueError("require 1 <= n_min <= n_max")
        self.n_min = n_min
        self.n_max = n_max
        self.max_context = max_context

    def propose(self, token_ids: list[int], k: int) -> list[int]:
        if k <= 0:
            return []
        arr = np.asarray(token_ids[-self.max_context :], dtype=np.int64)
        length = arr.shape[0]
        for n in range(self.n_max, self.n_min - 1, -1):
            if length <= n:
                continue
            suffix = arr[-n:]
            # candidate windows start at 0 .. length - n - 1 (exclude the suffix itself)
            windows = sliding_window_view(arr[:-1], n)
            hits = np.flatnonzero((windows == suffix).all(axis=1))
            if hits.size:
                start = int(hits[-1]) + n
                return arr[start : start + k].tolist()
        return []
