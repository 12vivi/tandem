"""Hybrid MoE executor: GPU-resident / page-in / CPU paths per step (design doc §3.3, §4.6).

Reference implementation in PyTorch:

* master expert weights stay on the CPU (pinned when CUDA is present);
* each MoE layer has an ``ExpertCache`` of GPU copies;
* the CPU path runs in a worker thread, overlapping the GPU path (PyTorch ops release
  the GIL); only hidden states and expert outputs cross PCIe.

Planned replacements behind the same interface (P2/P3): kt-kernel AMX/AVX-512 kernels for
the CPU path, CUDA-graph-captured steps with ``cudaLaunchHostFunc`` synchronisation, and
NUMA-sharded expert weights.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

import torch

from tandem.hybrid.assign import ExpertAssignment, assign_experts
from tandem.hybrid.expert_cache import ExpertCache, ExpertWeights
from tandem.layers.mlp import swiglu
from tandem.layers.moe import SparseMoeBlock
from tandem.planner.cost_model import ExpertCostModel


class HybridMoEExecutor:
    def __init__(
        self,
        blocks: dict[int, SparseMoeBlock],
        gpu_device: torch.device,
        slots_per_layer: int,
        cost: ExpertCostModel,
        cpu_threads: int = 1,
        reuse_bonus: float = 0.5,
    ) -> None:
        self.blocks = blocks
        self.gpu_device = gpu_device
        self.cost = cost
        self.reuse_bonus = reuse_bonus
        self.caches = {layer: ExpertCache(slots_per_layer) for layer in blocks}
        self.pool = ThreadPoolExecutor(max_workers=max(1, cpu_threads), thread_name_prefix="tandem-cpu-expert")
        self.counters: Counter = Counter()
        self.last_assignment: dict[int, ExpertAssignment] = {}
        pin = gpu_device.type == "cuda"
        for block in blocks.values():  # master copies live on the CPU
            block.experts.to("cpu")
            if pin:
                for p in block.experts.parameters():
                    p.data = p.data.pin_memory()
        for layer, block in blocks.items():  # warm start: the first experts fill the slots
            for e in range(min(slots_per_layer, block.num_experts)):
                self.caches[layer].put(e, self._upload(block, e))

    # ------------------------------------------------------------------ helpers
    def _upload(self, block: SparseMoeBlock, e: int) -> ExpertWeights:
        ex = block.experts[e]
        nb = self.gpu_device.type == "cuda"
        return tuple(  # type: ignore[return-value]
            w.weight.detach().to(self.gpu_device, non_blocking=nb) for w in (ex.gate_proj, ex.up_proj, ex.down_proj)
        )

    @staticmethod
    def _cpu_path(block: SparseMoeBlock, experts: list[int], hidden: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            out = torch.zeros_like(hidden)
            for e in experts:
                tok, slot = torch.where(ids == e)
                y = block.experts[e](hidden[tok]) * weights[tok, slot, None]
                out.index_add_(0, tok, y.to(out.dtype))
            return out

    # --------------------------------------------------------------------- run
    def run(self, layer_id: int, hidden: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, block: SparseMoeBlock) -> torch.Tensor:
        cache = self.caches[layer_id]
        counts = torch.bincount(ids.flatten(), minlength=block.num_experts).tolist()
        loads = {e: c for e, c in enumerate(counts) if c > 0}
        for e in loads:
            if e in cache:
                cache.touch(e)  # protect experts needed this step from eviction
        max_page_in = max(0, cache.capacity - sum(1 for e in loads if e in cache))
        plan = assign_experts(loads, cache.resident, self.cost, max_page_in, self.reuse_bonus)
        self.last_assignment[layer_id] = plan

        # CPU path first so it overlaps with the GPU path
        fut: Optional[Future] = None
        if plan.cpu:
            fut = self.pool.submit(self._cpu_path, block, plan.cpu, hidden.to("cpu"), ids.to("cpu"), weights.to("cpu"))

        for e in plan.page_in:
            cache.put(e, self._upload(block, e))

        out = torch.zeros_like(hidden)
        for e in sorted(plan.resident + plan.page_in):
            w_gate, w_up, w_down = cache.get(e)
            tok, slot = torch.where(ids == e)
            y = swiglu(hidden[tok], w_gate, w_up, w_down) * weights[tok, slot, None]
            out.index_add_(0, tok, y.to(out.dtype))

        if fut is not None:
            out = out + fut.result().to(out.device, non_blocking=True)

        self.counters["resident_experts"] += len(plan.resident)
        self.counters["page_in_experts"] += len(plan.page_in)
        self.counters["cpu_experts"] += len(plan.cpu)
        self.counters["cpu_tokens"] += sum(loads[e] for e in plan.cpu)
        self.counters["gpu_tokens"] += sum(loads[e] for e in plan.resident + plan.page_in)
        return out

    def stats(self) -> dict:
        total = sum(self.counters[k] for k in ("resident_experts", "page_in_experts", "cpu_experts"))
        stats = dict(self.counters)
        if total:
            stats["resident_hit_rate"] = round(self.counters["resident_experts"] / total, 3)
        return stats

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False)
