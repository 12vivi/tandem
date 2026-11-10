"""Per-step three-path expert assignment (design doc §4.6 b).

Experts already cached on the GPU run there. Each *missing* expert either pages in over
PCIe (then runs on the GPU and stays cached) or runs on the CPU where its weights live.
We pick the page-in set so the GPU path (resident compute + page-ins) and the CPU path
finish as close together as possible - the balancing rule of adaptive expert assignment
(arXiv 2610.23006). Experts with high token load favour page-in (CPU time grows with
tokens, transfer time does not); single-token decode favours the CPU when DRAM bandwidth
exceeds PCIe bandwidth, as with KTransformers (SOSP'25). With a large cache and high hit
rate this degenerates to SGLang Paged Experts; with no cache, to KTransformers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tandem.planner.cost_model import ExpertCostModel


@dataclass
class ExpertAssignment:
    resident: list[int] = field(default_factory=list)
    page_in: list[int] = field(default_factory=list)
    cpu: list[int] = field(default_factory=list)
    est_time: float = 0.0


def assign_experts(
    loads: dict[int, int],
    resident: set[int],
    cost: ExpertCostModel,
    max_page_in: int,
    reuse_bonus: float = 0.0,
) -> ExpertAssignment:
    """loads: expert id -> tokens routed to it this step."""
    res = sorted(e for e in loads if e in resident)
    missing = [e for e in loads if e not in resident]
    gpu_time = sum(cost.t_gpu(loads[e]) for e in res)
    t_cpu = {e: cost.t_cpu(loads[e]) for e in missing}
    # A paged-in expert stays cached; reuse_bonus credits the expected future hits.
    t_pin = {e: cost.t_pagein(loads[e]) * (1.0 - reuse_bonus) for e in missing}
    cpu_time = sum(t_cpu.values())
    page_in: list[int] = []
    pin_time = 0.0
    if max_page_in > 0:
        for e in sorted(missing, key=lambda x: t_cpu[x] / max(t_pin[x], 1e-12), reverse=True):
            if len(page_in) >= max_page_in:
                break
            new_pin, new_cpu = pin_time + t_pin[e], cpu_time - t_cpu[e]
            if max(gpu_time + new_pin, new_cpu) < max(gpu_time + pin_time, cpu_time):
                page_in.append(e)
                pin_time, cpu_time = new_pin, new_cpu
    chosen = set(page_in)
    return ExpertAssignment(
        resident=res,
        page_in=sorted(page_in),
        cpu=sorted(e for e in missing if e not in chosen),
        est_time=max(gpu_time + pin_time, cpu_time),
    )
