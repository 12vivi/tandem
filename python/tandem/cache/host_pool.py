"""L2 of the multi-tier KV cache: pinned host memory, page-first (design doc §4.4, docs/P3实施计划.md).

A page is one GPU block's KV for every layer, ``[layers][K, V][block_size * kv_heads * head_dim]``,
so it is read and written as a unit. Pages are handed out by a ``BlockAllocator``.

Copies between GPU blocks and host pages run on a side CUDA stream that first waits for the
compute stream, i.e. for every step already queued: the KV being written to the host is then
complete, and blocks being loaded into are no longer touched by in-flight steps. Each copy
returns an event. Without CUDA (a CPU KV cache, in tests) the copies are tensor indexing and
complete at once (``None``).
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import torch

from tandem.cache.block_allocator import BlockAllocator
from tandem.cache.kv_cache import KVCache

logger = logging.getLogger(__name__)


class HostKVPool:
    def __init__(self, kv_cache: KVCache, num_pages: int) -> None:
        if num_pages <= 0:
            raise ValueError("num_pages must be positive")
        self.kv = kv_cache
        self.num_layers = len(kv_cache.k)
        self.num_pages = num_pages
        self.block_size = kv_cache.block_size
        slots, heads, dim = kv_cache.k[0].shape
        self.num_gpu_blocks = slots // self.block_size
        self.page_elems = self.block_size * heads * dim  # one layer's K (or V) of one page
        dtype = kv_cache.k[0].dtype
        self.page_bytes = self.page_elems * kv_cache.k[0].element_size()
        self.bytes_per_page = 2 * self.num_layers * self.page_bytes
        self.device = kv_cache.k[0].device
        self.allocator = BlockAllocator(num_pages)
        self.ext = None
        if self.device.type == "cuda":
            from tandem.cache import kv_io

            self.ext = kv_io.load()
            self.stream = torch.cuda.Stream(self.device)
        if self.ext is not None:
            raw = self.ext.pinned_empty(num_pages * self.bytes_per_page)
            self.dev_ptr = self.ext.device_ptr(raw)
            self.data = raw.view(dtype).view(num_pages, self.num_layers, 2, self.page_elems)
            ptrs = [t.data_ptr() for i in range(self.num_layers) for t in (kv_cache.k[i], kv_cache.v[i])]
            self.layer_ptrs = torch.tensor(ptrs, dtype=torch.int64, device=self.device)
        else:
            pin = self.device.type == "cuda"
            self.data = torch.empty((num_pages, self.num_layers, 2, self.page_elems), dtype=dtype, pin_memory=pin)
        logger.info("host KV pool: %d pages x %.2f MB = %.1f GB (%s copies)", num_pages, self.bytes_per_page / 1e6,
                    num_pages * self.bytes_per_page / 1e9, "kernel" if self.ext is not None else "per-layer")

    # ------------------------------------------------------------------ copies
    def to_host(self, gpu_blocks: Sequence[int], host_pages: Sequence[int]) -> Optional[torch.cuda.Event]:
        """Copy GPU blocks to host pages (write-back / write-through)."""
        return self._copy(gpu_blocks, host_pages, to_host=True)

    def to_gpu(self, host_pages: Sequence[int], gpu_blocks: Sequence[int]) -> Optional[torch.cuda.Event]:
        """Copy host pages into GPU blocks (load)."""
        return self._copy(gpu_blocks, host_pages, to_host=False)

    def _copy(self, gpu_blocks: Sequence[int], host_pages: Sequence[int], to_host: bool) -> Optional[torch.cuda.Event]:
        if len(gpu_blocks) != len(host_pages):
            raise ValueError("gpu_blocks and host_pages differ in length")
        if not gpu_blocks:
            return None
        if self.device.type != "cuda":
            self._copy_layers(gpu_blocks, host_pages, to_host)
            return None
        main = torch.cuda.current_stream(self.device)
        with torch.cuda.stream(self.stream):
            if self.ext is not None:
                # index lists through pinned memory: a pageable H2D copy would sync this stream
                g = torch.tensor(gpu_blocks, dtype=torch.int32).pin_memory().to(self.device, non_blocking=True)
                h = torch.tensor(host_pages, dtype=torch.int32).pin_memory().to(self.device, non_blocking=True)
                self.stream.wait_stream(main)
                self.ext.kv_page_copy(self.layer_ptrs, g, h, self.dev_ptr, self.page_bytes, to_host)
            else:
                self.stream.wait_stream(main)
                self._copy_layers(gpu_blocks, host_pages, to_host)
            event = torch.cuda.Event()
            event.record(self.stream)
        return event

    def _copy_layers(self, gpu_blocks: Sequence[int], host_pages: Sequence[int], to_host: bool) -> None:
        g = torch.tensor(gpu_blocks, dtype=torch.long, device=self.device)
        h = torch.tensor(host_pages, dtype=torch.long)
        for layer in range(self.num_layers):
            for kv, t in enumerate((self.kv.k[layer], self.kv.v[layer])):
                pages = t.view(self.num_gpu_blocks, self.page_elems)
                if to_host:
                    self.data[h, layer, kv] = pages[g].to("cpu")
                else:
                    pages[g] = self.data[h, layer, kv].to(self.device)
