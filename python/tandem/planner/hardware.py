"""Hardware profiles: presets for planning and a lightweight micro-benchmark probe.

Units: bandwidth in GB/s, compute in TFLOPS (dense, effective), memory in GB.
Presets are planning estimates; the numbers come from the sources noted next to them
(see docs/设计方案.md §3.2 and §4.6). Probe the real machine before trusting a plan.
"""

from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch


@dataclass
class GPUSpec:
    name: str
    vram_gb: float
    hbm_gbps: float
    tflops: float  # dense BF16 tensor-core throughput


@dataclass
class CPUSpec:
    name: str
    isa: list[str] = field(default_factory=list)
    cores: int = 0
    numa_nodes: int = 1
    dram_gbps: float = 0.0  # achievable (not theoretical) bandwidth
    tflops: float = 0.0  # effective GEMM throughput for expert kernels


@dataclass
class HardwareProfile:
    name: str
    cpu: CPUSpec
    gpu: Optional[GPUSpec] = None
    pcie_gbps: float = 0.0  # pinned host-to-device bandwidth
    source: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "HardwareProfile":
        gpu = GPUSpec(**d["gpu"]) if d.get("gpu") else None
        return cls(name=d["name"], cpu=CPUSpec(**d["cpu"]), gpu=gpu, pcie_gbps=d.get("pcie_gbps", 0.0), source=d.get("source", ""))


PRESETS: dict[str, HardwareProfile] = {
    # Consumer desktop: RTX 4090 (1008 GB/s, ~165 BF16 TFLOPS dense) on PCIe 4.0 x16
    # (~25 GB/s measured pinned H2D), dual-channel DDR5 (~80 GB/s achievable), AVX2 CPU.
    "rtx4090-desktop": HardwareProfile(
        name="rtx4090-desktop",
        gpu=GPUSpec("RTX 4090", 24, 1008, 165),
        cpu=CPUSpec("desktop AVX2, 2ch DDR5", ["avx2"], 16, 1, 80, 1.0),
        pcie_gbps=25,
        source="estimate; PCIe from NVIDIA forum measurements (A100 Gen4 24.9 GB/s)",
    ),
    # Workstation from the OSDI'26 hybrid MoE paper (arXiv 2606.10493): RTX 5090 on PCIe 5.0,
    # 2x EPYC 9355 with 24 channels DDR5-6400 (1228 GB/s theoretical, ~947 GB/s achieved by an
    # AVX-512 FP8 GEMV).
    "rtx5090-epyc9355x2": HardwareProfile(
        name="rtx5090-epyc9355x2",
        gpu=GPUSpec("RTX 5090", 32, 1792, 209),
        cpu=CPUSpec("2x EPYC 9355 (AVX-512)", ["avx2", "avx512f"], 64, 2, 947, 8.0),
        pcie_gbps=50,
        source="arXiv 2606.10493; PCIe Gen5 ~50 GB/s measured on H100 (NVIDIA forum)",
    ),
    # KTransformers (SOSP'25) testbed: 2x Xeon 8452Y (AMX), 220 GB/s local per socket,
    # AMX MoE kernel 21.3 TFLOPS per socket; A100-40GB on PCIe 4.0.
    "a100-xeon8452yx2": HardwareProfile(
        name="a100-xeon8452yx2",
        gpu=GPUSpec("A100 40GB", 40, 1555, 312),
        cpu=CPUSpec("2x Xeon 8452Y (AMX)", ["avx2", "avx512f", "amx_tile"], 72, 2, 440, 42.6),
        pcie_gbps=25,
        source="KTransformers SOSP'25 (DOI 10.1145/3731569.3764843)",
    ),
    # CPU-only laptop class (e.g. i5-1240P): AVX2, ~50 GB/s.
    "laptop-cpu": HardwareProfile(
        name="laptop-cpu",
        cpu=CPUSpec("laptop AVX2", ["avx2"], 12, 1, 50, 0.4),
        source="estimate",
    ),
}


def load_profile(spec: str) -> HardwareProfile:
    """``spec`` is a preset name, a JSON file path, or ``"probe"``."""
    if spec == "probe":
        return probe()
    if spec in PRESETS:
        return PRESETS[spec]
    if os.path.exists(spec):
        with open(spec, encoding="utf-8") as f:
            return HardwareProfile.from_dict(json.load(f))
    raise ValueError(f"unknown hardware profile {spec!r}; presets: {sorted(PRESETS)}")


# --------------------------------------------------------------------------- probing
def _cpu_isa() -> list[str]:
    flags: set[str] = set()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = set(line.split(":", 1)[1].split())
                    break
    except OSError:
        pass
    wanted = ["avx2", "avx_vnni", "avx512f", "avx512_vnni", "avx512_bf16", "amx_tile", "amx_bf16", "amx_int8"]
    isa = [f for f in wanted if f in flags]
    if not isa:  # non-Linux: fall back to what torch dispatches to
        cap = torch.backends.cpu.get_cpu_capability().lower()
        isa = ["avx2"] if "avx2" in cap else []
        if "avx512" in cap:
            isa.append("avx512f")
    return isa


def _numa_nodes() -> int:
    try:
        return max(1, len([d for d in os.listdir("/sys/devices/system/node") if d.startswith("node")]))
    except OSError:
        return 1


def _time(fn, repeat: int = 5) -> float:
    fn()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def probe(quick: bool = True) -> HardwareProfile:
    """Measure copy bandwidth and GEMM throughput with torch (rough, seconds-long)."""
    n = (64 if quick else 256) * 2**20 // 4  # floats
    a, b = torch.ones(n), torch.empty(n)
    dram = 2 * n * 4 / _time(lambda: b.copy_(a)) / 1e9  # read + write
    m = 1024 if quick else 2048
    x, y = torch.randn(m, m), torch.randn(m, m)
    cpu_tflops = 2 * m**3 / _time(lambda: x @ y) / 1e12
    cpu = CPUSpec(platform.processor() or platform.machine(), _cpu_isa(), os.cpu_count() or 1, _numa_nodes(), round(dram, 1), round(cpu_tflops, 2))

    gpu, pcie = None, 0.0
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        dev = torch.device("cuda")

        def sync_time(fn):
            def run():
                fn()
                torch.cuda.synchronize()

            return _time(run)

        g = torch.empty(n, device=dev)
        g2 = torch.empty(n, device=dev)
        hbm = 2 * n * 4 / sync_time(lambda: g2.copy_(g)) / 1e9
        host = torch.ones(n).pin_memory()
        pcie = n * 4 / sync_time(lambda: g.copy_(host, non_blocking=True)) / 1e9
        gm = 4096
        gx = torch.randn(gm, gm, device=dev, dtype=torch.bfloat16)
        gtf = 2 * gm**3 / sync_time(lambda: gx @ gx) / 1e12
        gpu = GPUSpec(props.name, round(props.total_memory / 2**30, 1), round(hbm, 1), round(gtf, 1))
    return HardwareProfile(name="probed", cpu=cpu, gpu=gpu, pcie_gbps=round(pcie, 1), source="tandem probe")
