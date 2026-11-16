"""L2 of the multi-tier KV cache: pinned host memory, page-first (design doc §4.4, docs/P3实施计划.md).

A page is one GPU block's KV for every layer, ``[layers][K, V][block_size * kv_heads * head_dim]``,
so it is read and written as a unit. Pages are handed out by a ``BlockAllocator``.

Copies between GPU blocks and host pages run on two side CUDA streams, one for loads (high
priority: a step is waiting for them) and one for writes (background), so that a load never
queues behind a write; each copy first waits for the compute stream, i.e. for every step already
queued: the KV being written to the host is then complete, and blocks being loaded into are no
longer touched by in-flight steps. The two directions never touch the same blocks or pages at
once: a node's GPU copy is dropped (and its blocks reused, or its pages loaded) only after its
write has completed.

* Loads (host -> GPU) use the page-copy kernel (tandem/csrc/kv_io.cu) on a few CTAs, one launch
  per layer: a ``LoadHandle`` holds one event per layer, so that the step that uses the loaded
  prefix can start right away and wait for each layer's KV just before that layer
  (``ForwardContext.layer_waits``).
* Writes (GPU -> host) gather the pages into a page-first staging buffer in GPU memory (the same
  kernel, device to device) and send it with the copy engines (cudaMemcpyAsync), one copy per run
  of consecutive host pages. SM stores to host memory would be as fast but stall every other
  memory access on the GPU while PCIe is saturated: a GEMM stream ran 2-5x slower beside them
  (benchmarks/bench_kv_io.py), even with two CTAs.

Without CUDA (a CPU KV cache, in tests) the copies are tensor indexing and complete at once
(``None``).
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

import torch

from tandem.cache.block_allocator import BlockAllocator
from tandem.cache.kv_cache import KVCache

logger = logging.getLogger(__name__)


class LoadHandle:
    """Events of one host -> GPU copy on the copy stream, one per layer: layer i's KV is in place
    once ``events[i]`` has completed. In stream order, so the last one means all of them."""

    __slots__ = ("events",)

    def __init__(self, events: list) -> None:
        self.events = events

    def query(self) -> bool:
        return self.events[-1].query()

    def synchronize(self) -> None:
        self.events[-1].synchronize()


class HostKVPool:
    # CTAs of a load (host -> GPU, per layer; benchmarks/bench_kv_io.py, p3-kv-io-v2.jsonl): two
    # already saturate PCIe (12.3 GB/s here); a GEMM stream beside the copy runs 2.4% slower with
    # 2-4, 3.7% with 8, 19% with 32, and 5.3x slower with one CTA per chunk (the first version)
    LOAD_CTAS = int(os.environ.get("TANDEM_KV_LOAD_CTAS", "4"))
    GATHER_CTAS = 64  # device-to-device gather into the staging buffer (DRAM speed)
    STAGING_PAGES = 16  # GPU staging for writes: 16 pages of Qwen3-8B = 38 MB

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
        self.load_ctas = self.LOAD_CTAS
        self.ext = None
        if self.device.type == "cuda":
            from tandem.cache import kv_io

            self.ext = kv_io.load()
            self.load_stream = torch.cuda.Stream(self.device, priority=-1)
            self.write_stream = torch.cuda.Stream(self.device)
        if self.ext is not None:
            self.raw = self.ext.pinned_empty(num_pages * self.bytes_per_page)
            self.dev_ptr = self.ext.device_ptr(self.raw)
            self.data = self.raw.view(dtype).view(num_pages, self.num_layers, 2, self.page_elems)
            ptrs = [t.data_ptr() for i in range(self.num_layers) for t in (kv_cache.k[i], kv_cache.v[i])]
            self.layer_ptrs = torch.tensor(ptrs, dtype=torch.int64, device=self.device)
            self.staging = torch.empty(self.STAGING_PAGES * self.bytes_per_page, dtype=torch.uint8, device=self.device)
            self.staging_idx = torch.arange(self.STAGING_PAGES, dtype=torch.int32, device=self.device)
        else:
            pin = self.device.type == "cuda"
            self.data = torch.empty((num_pages, self.num_layers, 2, self.page_elems), dtype=dtype, pin_memory=pin)
        logger.info("host KV pool: %d pages x %.2f MB = %.1f GB (%s copies)", num_pages, self.bytes_per_page / 1e6,
                    num_pages * self.bytes_per_page / 1e9, "kernel" if self.ext is not None else "per-layer")

    # ------------------------------------------------------------------ copies
    def to_host(self, gpu_blocks: Sequence[int], host_pages: Sequence[int]) -> Optional[torch.cuda.Event]:
        """Copy GPU blocks to host pages (write-through / write-back); one event."""
        events = self._copy(gpu_blocks, host_pages, to_host=True, per_layer=False)
        return None if events is None else events[-1]

    def to_gpu(self, host_pages: Sequence[int], gpu_blocks: Sequence[int]) -> Optional[LoadHandle]:
        """Copy host pages into GPU blocks (load), layer by layer."""
        events = self._copy(gpu_blocks, host_pages, to_host=False, per_layer=True)
        return None if events is None else LoadHandle(events)

    def _copy(self, gpu_blocks: Sequence[int], host_pages: Sequence[int], to_host: bool,
              per_layer: bool) -> Optional[list]:
        if len(gpu_blocks) != len(host_pages):
            raise ValueError("gpu_blocks and host_pages differ in length")
        if not gpu_blocks:
            return None
        if self.device.type != "cuda":
            self._copy_layers(gpu_blocks, host_pages, to_host, range(self.num_layers))
            return None
        main = torch.cuda.current_stream(self.device)
        layer_groups = [range(i, i + 1) for i in range(self.num_layers)] if per_layer else [range(self.num_layers)]
        events = []
        stream = self.write_stream if to_host else self.load_stream
        with torch.cuda.stream(stream):
            if self.ext is not None:
                # index lists through pinned memory: a pageable H2D copy would sync this stream
                g = torch.tensor(gpu_blocks, dtype=torch.int32).pin_memory().to(self.device, non_blocking=True)
                if not to_host:
                    h = torch.tensor(host_pages, dtype=torch.int32).pin_memory().to(self.device, non_blocking=True)
            stream.wait_stream(main)
            if self.ext is not None and to_host:
                self._write_staged(g, list(host_pages))
                layer_groups = []
                events.append(torch.cuda.Event())
                events[-1].record(stream)
            for layers in layer_groups:
                if self.ext is not None:
                    self.ext.kv_page_copy(self.layer_ptrs, g, h, self.dev_ptr, self.page_bytes, to_host,
                                          2 * layers.start, 2 * len(layers), self.load_ctas)
                else:
                    self._copy_layers(gpu_blocks, host_pages, to_host, layers)
                event = torch.cuda.Event()
                event.record(stream)
                events.append(event)
        return events

    def _write_staged(self, g: torch.Tensor, host_pages: list[int]) -> None:
        """On the copy stream: per chunk of STAGING_PAGES pages, gather them page-first into the
        staging buffer, then copy-engine copies into the host pages, one per consecutive run.
        Stream order keeps a chunk's gather behind the previous chunk's copies."""
        bpp, chunk = self.bytes_per_page, self.STAGING_PAGES
        for at in range(0, len(host_pages), chunk):
            pages = host_pages[at : at + chunk]
            self.ext.kv_page_copy(self.layer_ptrs, g[at : at + len(pages)], self.staging_idx[: len(pages)],
                                  self.staging.data_ptr(), self.page_bytes, True, 0, -1, self.GATHER_CTAS)
            i = 0
            while i < len(pages):
                j = i + 1
                while j < len(pages) and pages[j] == pages[j - 1] + 1:
                    j += 1
                self.raw[pages[i] * bpp : (pages[j - 1] + 1) * bpp].copy_(self.staging[i * bpp : j * bpp],
                                                                          non_blocking=True)
                i = j

    def _copy_layers(self, gpu_blocks: Sequence[int], host_pages: Sequence[int], to_host: bool, layers) -> None:
        g = torch.tensor(gpu_blocks, dtype=torch.long, device=self.device)
        h = torch.tensor(host_pages, dtype=torch.long)
        for layer in layers:
            for kv, t in enumerate((self.kv.k[layer], self.kv.v[layer])):
                pages = t.view(self.num_gpu_blocks, self.page_elems)
                if to_host:
                    self.data[h, layer, kv] = pages[g].to("cpu")
                else:
                    pages[g] = self.data[h, layer, kv].to(self.device)
