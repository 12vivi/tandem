"""CUDA graphs for decode steps (design doc §4.2).

One graph per batch-size bucket (1, 2, 4, 8, 16, 24, ... up to ``max_bs``), captured from the
largest down with a shared memory pool. A decode step with b requests replays the smallest
bucket >= b; the padding rows decode a dummy token whose KV lives in a scratch block that the
allocator never hands out, so they cannot touch real requests.

Each bucket has its own CUDA-graph-safe FlashInfer decode wrapper (fixed indptr / indices /
last-page-len buffers). Per step, outside the graph: copy token ids, positions and KV slots
into the static buffers and re-plan the bucket's wrapper; then replay. Logits are computed
outside the graph on the real rows only.

Buckets from the attention backend's ``nosplit_min_batch`` up get a second graph with its own
wrapper, planned without KV splitting (see FlashInferAttentionBackend); a step replays it when
its longest context is at most ``nosplit_max_len`` tokens. Each wrapper is always planned the
way it was captured.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import torch
from torch import nn

from tandem.executor.forward_context import ForwardContext
from tandem.executor.step_timer import TIMER
from tandem.layers.flashinfer_attention import FlashInferAttentionBackend, FlashInferMetadata, paged_layout

logger = logging.getLogger(__name__)


def bucket_sizes(max_bs: int) -> list[int]:
    sizes = [b for b in (1, 2, 4, 8) if b <= max_bs] + list(range(16, max_bs + 1, 8))
    if sizes[-1] != max_bs:
        sizes.append(max_bs)
    return sizes


class CudaGraphRunner:
    def __init__(
        self,
        model: nn.Module,
        kv_cache: Any,
        backend: FlashInferAttentionBackend,
        ops: Any,
        moe_executor: Any,
        max_bs: int,
        scratch_block: int,
        max_pages_per_seq: int,
    ) -> None:
        self.max_pages_per_seq = max_pages_per_seq  # requests sharing cached prefix blocks list them again
        self.model = model
        self.kv_cache = kv_cache
        self.backend = backend
        self.ops = ops
        self.moe_executor = moe_executor
        self.block_size = backend.block_size
        self.scratch_block = scratch_block
        self.scratch_slot = scratch_block * self.block_size
        self.sizes = bucket_sizes(max_bs)
        self.max_bs = self.sizes[-1]
        device = backend.device
        self.input_ids = torch.zeros(self.max_bs, dtype=torch.long, device=device)
        self.positions = torch.zeros(self.max_bs, dtype=torch.long, device=device)
        self.slot_mapping = torch.full((self.max_bs,), self.scratch_slot, dtype=torch.long, device=device)
        self.wrappers: dict[int, Any] = {}
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.outputs: dict[int, torch.Tensor] = {}
        self.nosplit: dict[int, tuple[Any, torch.cuda.CUDAGraph, torch.Tensor]] = {}  # bucket -> (wrapper, graph, out)
        self._capture_all()

    # ------------------------------------------------------------------ capture
    def _dummy(self, n: int) -> list[tuple[list[int], int, int]]:
        return [([self.scratch_block], 0, 1)] * n

    def _capture(self, bs: int, pool: Any, split: bool) -> tuple[Any, torch.cuda.CUDAGraph, torch.Tensor]:
        device = self.backend.device
        wrapper = self.backend.make_decode_wrapper(
            paged_kv_indptr_buffer=torch.zeros(bs + 1, dtype=torch.int32, device=device),
            paged_kv_indices_buffer=torch.zeros(bs * self.max_pages_per_seq, dtype=torch.int32, device=device),
            paged_kv_last_page_len_buffer=torch.zeros(bs, dtype=torch.int32, device=device),
        )
        self.backend.plan_decode(wrapper, paged_layout(self._dummy(bs), self.block_size), split=split)
        ctx = ForwardContext(FlashInferMetadata(self.slot_mapping[:bs], wrapper, bs), self.kv_cache,
                             self.backend, self.moe_executor, ops=self.ops)

        def run() -> torch.Tensor:
            return self.model(self.input_ids[:bs], self.positions[:bs], ctx)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            out = run()
        return wrapper, graph, out

    @torch.inference_mode()
    def _capture_all(self) -> None:
        pool = torch.cuda.graph_pool_handle()
        min_nosplit = getattr(self.backend, "nosplit_min_batch", None)
        for bs in reversed(self.sizes):
            self.wrappers[bs], self.graphs[bs], self.outputs[bs] = self._capture(bs, pool, split=True)
            if min_nosplit is not None and bs >= min_nosplit:
                self.nosplit[bs] = self._capture(bs, pool, split=False)
        torch.cuda.synchronize()
        logger.info("captured CUDA graphs for decode batch sizes %s (also without KV split: %s)",
                    self.sizes, sorted(self.nosplit))

    # ------------------------------------------------------------------ replay
    def can_run(self, num_seqs: int) -> bool:
        return num_seqs <= self.max_bs

    @torch.inference_mode()
    def run(
        self,
        seqs: Sequence[tuple[Sequence[int], int, int]],
        input_ids: Sequence[int],
        fixups: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """seqs: decode-only (block_ids, start, 1) per request -> hidden states [b, H].
        fixups: (positions, token values) on the device for inputs still in flight."""
        b = len(seqs)
        bs = next(s for s in self.sizes if s >= b)
        layout = paged_layout(list(seqs) + self._dummy(bs - b), self.block_size)
        TIMER.mark("graph_layout")
        ids = torch.zeros(bs, dtype=torch.long)
        ids[:b] = torch.tensor(input_ids, dtype=torch.long)
        self.input_ids[:bs].copy_(ids, non_blocking=True)
        if fixups is not None:
            self.input_ids[fixups[0]] = fixups[1]
        self.positions[:bs].copy_(layout.positions, non_blocking=True)
        self.slot_mapping[:bs].copy_(layout.slot_mapping, non_blocking=True)
        TIMER.mark("graph_copies")
        variant = self.nosplit.get(bs)
        if variant is not None and max(start + n for _, start, n in seqs) <= self.backend.nosplit_max_len:
            wrapper, graph, out = variant
            self.backend.plan_decode(wrapper, layout, split=False)
        else:
            wrapper, graph, out = self.wrappers[bs], self.graphs[bs], self.outputs[bs]
            self.backend.plan_decode(wrapper, layout)
        TIMER.mark("graph_plan")
        graph.replay()
        TIMER.mark("graph_replay")
        return out[:b]
