"""Build (once, cached by torch in ~/.cache/torch_extensions) and load the W4A16 GEMV CUDA kernel
(tandem/csrc/w4_gemv.cu). Needs nvcc (CUDA_HOME) on first use; without it tandem falls back to
the Triton kernels in tandem.layers.awq."""

from __future__ import annotations

import functools
import logging
import os

import torch

logger = logging.getLogger(__name__)

_DECL = "torch::Tensor w4_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor s, torch::Tensor z, int64_t rows, int64_t warps);"


@functools.lru_cache(maxsize=None)
def load():
    if os.environ.get("TANDEM_NO_CUDA_EXT") == "1" or not torch.cuda.is_available():
        return None
    try:
        from torch.utils.cpp_extension import load_inline

        src = open(os.path.join(os.path.dirname(__file__), "..", "csrc", "w4_gemv.cu"), encoding="utf-8").read()
        major, minor = torch.cuda.get_device_capability()
        return load_inline(
            name=f"tandem_w4_gemv_sm{major}{minor}", cpp_sources=_DECL, cuda_sources=src, functions=["w4_gemv"],
            extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"],
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 - no nvcc / compiler: Triton fallback
        logger.warning("W4A16 CUDA kernel unavailable (%s); using the Triton kernels", str(exc).splitlines()[0][:200])
        return None
