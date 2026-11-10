from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import torch

ExpertWeights = tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # gate, up, down


class ExpertCache:
    """GPU-resident expert slots of one MoE layer, LRU replacement.

    Master copies of all experts stay in host memory; the cache holds device copies.
    Experts used in the current step are touched first (``touch``) so a page-in in the
    same step never evicts them.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._slots: OrderedDict[int, ExpertWeights] = OrderedDict()

    def __contains__(self, expert: int) -> bool:
        return expert in self._slots

    def __len__(self) -> int:
        return len(self._slots)

    @property
    def resident(self) -> set[int]:
        return set(self._slots)

    def touch(self, expert: int) -> None:
        self._slots.move_to_end(expert)

    def get(self, expert: int) -> ExpertWeights:
        self._slots.move_to_end(expert)
        return self._slots[expert]

    def put(self, expert: int, weights: ExpertWeights) -> Optional[int]:
        """Insert an expert; returns the evicted expert id, if any."""
        if self.capacity <= 0:
            raise RuntimeError("expert cache has no slots")
        evicted = None
        if expert not in self._slots and len(self._slots) >= self.capacity:
            evicted, _ = self._slots.popitem(last=False)
        self._slots[expert] = weights
        self._slots.move_to_end(expert)
        return evicted
