"""Build (once, cached by torch in ~/.cache/torch_extensions) and load the W4A16 CUDA kernels
(tandem/csrc/w4_gemv.cu: w4_gemv on CUDA cores and w4_mma on tensor cores for M <= 16, w4_gemm for
M > 16). Needs nvcc (CUDA_HOME) on first use; without it tandem.layers.awq dequantizes and runs
cuBLAS instead."""

from __future__ import annotations

import functools
import hashlib
import logging
import os

import torch

logger = logging.getLogger(__name__)

_DECL = """
torch::Tensor w4_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor s, torch::Tensor z, int64_t rows, int64_t warps);
torch::Tensor w4_mma(torch::Tensor x, torch::Tensor w, torch::Tensor s, torch::Tensor z, int64_t warps_k, int64_t unroll, int64_t ntile, int64_t act);
torch::Tensor w4_gemm(torch::Tensor x, torch::Tensor w, torch::Tensor s, torch::Tensor z, torch::Tensor ws, torch::Tensor locks, int64_t mt, int64_t nt, int64_t warps_n, int64_t warps_k, int64_t stages, int64_t splits, int64_t act);
"""


WORKSPACE_FLOATS = 1 << 23  # 32 MB of float32 split-K partial sums per GPU
WORKSPACE_LOCKS = 1 << 15
_workspaces: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def workspace(device: torch.device | str | int) -> tuple[torch.Tensor, torch.Tensor]:
    """w4_gemm's split-K buffers on a GPU: float32 partial sums and int32 counters, zero between
    calls (the kernel resets them). Allocated once, before CUDA graph capture; the GEMMs that use
    them must run on one stream at a time."""
    dev = torch.device(device) if not isinstance(device, int) else torch.device("cuda", device)
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    buf = _workspaces.get(index)
    if buf is None:
        dev = torch.device("cuda", index)
        buf = (torch.empty(WORKSPACE_FLOATS, dtype=torch.float32, device=dev),
               torch.zeros(WORKSPACE_LOCKS, dtype=torch.int32, device=dev))
        _workspaces[index] = buf
    return buf


@functools.lru_cache(maxsize=None)
def load():
    if os.environ.get("TANDEM_NO_CUDA_EXT") == "1" or not torch.cuda.is_available():
        return None
    try:
        from torch.utils.cpp_extension import load_inline

        src = open(os.path.join(os.path.dirname(__file__), "..", "csrc", "w4_gemv.cu"), encoding="utf-8").read()
        major, minor = torch.cuda.get_device_capability()
        digest = hashlib.sha1(src.encode()).hexdigest()[:8]  # a new build whenever the source changes
        return load_inline(
            name=f"tandem_w4_sm{major}{minor}_{digest}", cpp_sources=_DECL, cuda_sources=src,
            functions=["w4_gemv", "w4_mma", "w4_gemm"],
            extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"],
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 - no nvcc / compiler: dequantize + cuBLAS
        logger.warning("W4A16 CUDA kernels unavailable (%s); using dequantize + cuBLAS", str(exc).splitlines()[0][:200])
        return None
