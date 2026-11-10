"""Roofline cost model (design doc §4.6 b).

For an operator with N parameters (b bytes each) processing T tokens:

    t_gpu    = max(N*b / B_hbm,  2*N*T / P_gpu)
    t_cpu    = max(N*b / B_dram, 2*N*T / P_cpu)
    t_pagein = N*b / B_pcie + t_gpu          (copy the weights, then compute on GPU)
    T*       = P_cpu * b / (2 * B_pcie)       (above T* tokens, page-in/streaming beats CPU)

Sources for the approach: FlexGen (ICML'23) LP policy search, MoE-Lightning (ASPLOS'25)
hierarchical roofline, Fiddler (ICLR'25) latency model, adaptive expert assignment
(arXiv 2610.23006).
"""

from __future__ import annotations

from dataclasses import dataclass

from tandem.planner.hardware import HardwareProfile

GB = 1e9
TFLOP = 1e12


@dataclass
class ExpertCostModel:
    hw: HardwareProfile
    expert_params: int
    bytes_per_param: float

    @property
    def weight_bytes(self) -> float:
        return self.expert_params * self.bytes_per_param

    def t_gpu(self, tokens: int) -> float:
        gpu = self.hw.gpu
        if gpu is None:
            return float("inf")
        return max(self.weight_bytes / (gpu.hbm_gbps * GB), 2 * self.expert_params * tokens / (gpu.tflops * TFLOP))

    def t_cpu(self, tokens: int) -> float:
        cpu = self.hw.cpu
        return max(self.weight_bytes / (cpu.dram_gbps * GB), 2 * self.expert_params * tokens / (cpu.tflops * TFLOP))

    def t_pagein(self, tokens: int) -> float:
        if self.hw.gpu is None or self.hw.pcie_gbps <= 0:
            return float("inf")
        return self.weight_bytes / (self.hw.pcie_gbps * GB) + self.t_gpu(tokens)


def stream_crossover_tokens(hw: HardwareProfile, bytes_per_param: float) -> float:
    """T*: tokens per expert/operator above which streaming weights to the GPU beats CPU compute."""
    if hw.pcie_gbps <= 0:
        return float("inf")
    return hw.cpu.tflops * TFLOP * bytes_per_param / (2 * hw.pcie_gbps * GB)
