"""Build (once, cached by torch in ~/.cache/torch_extensions) and load the KV page-copy kernels
(tandem/csrc/kv_io.cu). Needs nvcc (CUDA_HOME) on first use; without it tandem.cache.host_pool
falls back to per-layer tensor copies."""

from __future__ import annotations

import functools
import hashlib
import logging
import os

import torch

logger = logging.getLogger(__name__)

_DECL = """
torch::Tensor pinned_empty(int64_t nbytes);
int64_t device_ptr(torch::Tensor pinned);
void kv_page_copy(torch::Tensor layer_ptrs, torch::Tensor gpu_blocks, torch::Tensor host_pages, int64_t host_dev_ptr,
                  int64_t page_bytes, bool to_host);
"""


@functools.lru_cache(maxsize=None)
def load():
    if os.environ.get("TANDEM_NO_CUDA_EXT") == "1" or not torch.cuda.is_available():
        return None
    try:
        from torch.utils.cpp_extension import load_inline

        src = open(os.path.join(os.path.dirname(__file__), "..", "csrc", "kv_io.cu"), encoding="utf-8").read()
        major, minor = torch.cuda.get_device_capability()
        digest = hashlib.sha1(src.encode()).hexdigest()[:8]
        return load_inline(
            name=f"tandem_kv_io_sm{major}{minor}_{digest}", cpp_sources=_DECL, cuda_sources=src,
            functions=["pinned_empty", "device_ptr", "kv_page_copy"],
            extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"],
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 - no nvcc / compiler: per-layer tensor copies
        logger.warning("KV page-copy kernels unavailable (%s); using per-layer copies", str(exc).splitlines()[0][:200])
        return None
