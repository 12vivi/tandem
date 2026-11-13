"""Build (once, cached by torch in ~/.cache/torch_extensions) and load the W4A16 decode CUDA kernels
(tandem/csrc/w4_gemv.cu: w4_gemv on CUDA cores, w4_mma on tensor cores). Needs nvcc (CUDA_HOME)
on first use; without it tandem.layers.awq dequantizes and runs cuBLAS instead."""

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
"""


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
            functions=["w4_gemv", "w4_mma"],
            extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"],
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 - no nvcc / compiler: dequantize + cuBLAS
        logger.warning("W4A16 CUDA kernels unavailable (%s); using dequantize + cuBLAS", str(exc).splitlines()[0][:200])
        return None
