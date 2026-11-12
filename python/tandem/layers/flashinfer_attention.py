"""FlashInfer paged attention (GPU fast path, design doc §4.3).

Uses the KV buffers of ``tandem.cache.kv_cache.KVCache`` as they are: per layer
``[num_blocks * block_size, num_kv_heads, head_dim]``, viewed as FlashInfer's NHD paged layout
``[num_blocks, block_size, num_kv_heads, head_dim]`` with one page per KV block.

Each step is planned once (outside the layers):
* every request has exactly one new token -> ``BatchDecodeWithPagedKVCacheWrapper``; with
  GQA group size >= 4 (Qwen3-8B) the tensor-core decode kernel, as SGLang picks it;
* otherwise (any prefill chunk, mixed batches) -> ``BatchPrefillWithPagedKVCacheWrapper``,
  causal, ragged queries.
New K/V are written with ``index_copy_`` into the slots computed by the scheduler's block
tables (two small kernels per layer; a fused QK-norm + RoPE + KV-write kernel is a P2 item).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from tandem.executor.forward_context import ForwardContext


@dataclass
class PagedLayout:
    """Host-side description of one step (int32 for FlashInfer's planner, int64 for indexing)."""

    qo_indptr: torch.Tensor  # [B + 1] int32
    kv_indptr: torch.Tensor  # [B + 1] int32
    kv_indices: torch.Tensor  # [num_pages] int32
    kv_last_page_len: torch.Tensor  # [B] int32
    positions: torch.Tensor  # [T] int64
    slot_mapping: torch.Tensor  # [T] int64
    is_decode: bool


def paged_layout(seqs: Sequence[tuple[Sequence[int], int, int]], block_size: int) -> PagedLayout:
    """seqs: (block_ids, start, num_new) per request."""
    qo, kvp, indices, last, positions, slots = [0], [0], [], [], [], []
    is_decode = True
    for block_ids, start, n in seqs:
        end = start + n
        pages = (end + block_size - 1) // block_size
        indices.extend(block_ids[:pages])
        kvp.append(kvp[-1] + pages)
        last.append(end - (pages - 1) * block_size)
        qo.append(qo[-1] + n)
        if n == 1:
            positions.append(start)
            slots.append(block_ids[start // block_size] * block_size + start % block_size)
        else:
            is_decode = False
            for p in range(start, end):
                positions.append(p)
                slots.append(block_ids[p // block_size] * block_size + p % block_size)
    i32 = torch.int32
    return PagedLayout(
        qo_indptr=torch.tensor(qo, dtype=i32),
        kv_indptr=torch.tensor(kvp, dtype=i32),
        kv_indices=torch.tensor(indices, dtype=i32),
        kv_last_page_len=torch.tensor(last, dtype=i32),
        positions=torch.tensor(positions, dtype=torch.long),
        slot_mapping=torch.tensor(slots, dtype=torch.long),
        is_decode=is_decode,
    )


@dataclass
class FlashInferMetadata:
    slot_mapping: torch.Tensor  # [T] on the device
    wrapper: Any  # the planned prefill or decode wrapper
    num_tokens: int


class FlashInferAttentionBackend:
    name = "flashinfer"

    def __init__(
        self,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device,
        workspace_bytes: int = 256 << 20,
    ) -> None:
        import flashinfer

        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"FlashInfer attention needs float16 / bfloat16, got {dtype}")
        self.flashinfer = flashinfer
        self.num_qo_heads, self.num_kv_heads, self.head_dim = num_qo_heads, num_kv_heads, head_dim
        self.block_size, self.dtype, self.device = block_size, dtype, device
        self.scale = head_dim**-0.5
        self.use_tensor_cores = num_qo_heads // num_kv_heads >= 4
        self.workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.workspace, "NHD")
        self.decode_wrapper = self.make_decode_wrapper()
        self._plan_events: dict[int, torch.cuda.Event] = {}

    def _guarded_plan(self, wrapper: Any, plan) -> None:
        """plan() writes the wrapper's pinned host buffer and queues a copy to the device. With
        overlap scheduling the previous plan's copy may still be queued behind the running step,
        so wait for it before overwriting the buffer (it has almost always run already)."""
        event = self._plan_events.get(id(wrapper))
        if event is not None:
            event.synchronize()
        plan()
        event = torch.cuda.Event()
        event.record()
        self._plan_events[id(wrapper)] = event

    def make_decode_wrapper(self, **graph_buffers) -> Any:
        """A decode wrapper; with graph_buffers (paged_kv_*_buffer=...) it is CUDA-graph safe."""
        return self.flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace, "NHD", use_cuda_graph=bool(graph_buffers), use_tensor_cores=self.use_tensor_cores,
            **graph_buffers,
        )

    def plan_decode(self, wrapper: Any, layout: PagedLayout) -> None:
        self._guarded_plan(wrapper, lambda: wrapper.plan(
            layout.kv_indptr, layout.kv_indices, layout.kv_last_page_len,
            self.num_qo_heads, self.num_kv_heads, self.head_dim, self.block_size,
            q_data_type=self.dtype, kv_data_type=self.dtype, sm_scale=self.scale,
        ))

    def prepare(
        self, seqs: Sequence[tuple[Sequence[int], int, int]], block_size: int, device: torch.device
    ) -> tuple[torch.Tensor, FlashInferMetadata]:
        layout = paged_layout(seqs, block_size)
        if layout.is_decode:
            self.plan_decode(self.decode_wrapper, layout)
            wrapper = self.decode_wrapper
        else:
            wrapper = self.prefill_wrapper
            self._guarded_plan(wrapper, lambda: wrapper.plan(
                layout.qo_indptr, layout.kv_indptr, layout.kv_indices, layout.kv_last_page_len,
                self.num_qo_heads, self.num_kv_heads, self.head_dim, self.block_size,
                causal=True, q_data_type=self.dtype, kv_data_type=self.dtype, sm_scale=self.scale,
            ))
        positions = layout.positions.to(device, non_blocking=True)
        slots = layout.slot_mapping.to(device, non_blocking=True)
        return positions, FlashInferMetadata(slots, wrapper, int(slots.shape[0]))

    def paged_cache(self, ctx: ForwardContext, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        k, v = ctx.kv_cache.k[layer_id], ctx.kv_cache.v[layer_id]
        shape = (k.shape[0] // self.block_size, self.block_size, self.num_kv_heads, self.head_dim)
        return k.view(shape), v.view(shape)

    def forward(
        self,
        layer_id: int,
        q: torch.Tensor,  # [T, nh, hd]
        k: torch.Tensor,  # [T, nkv, hd]
        v: torch.Tensor,  # [T, nkv, hd]
        ctx: ForwardContext,
        scale: float,
    ) -> torch.Tensor:
        meta: FlashInferMetadata = ctx.attn
        k_cache, v_cache = ctx.kv_cache.k[layer_id], ctx.kv_cache.v[layer_id]
        k_cache.index_copy_(0, meta.slot_mapping, k)
        v_cache.index_copy_(0, meta.slot_mapping, v)
        return meta.wrapper.run(q.contiguous(), self.paged_cache(ctx, layer_id))

    def attend(self, layer_id: int, q: torch.Tensor, ctx: ForwardContext) -> torch.Tensor:
        """Attention only: the new K/V are already in the cache (fused prologue)."""
        return ctx.attn.wrapper.run(q, self.paged_cache(ctx, layer_id))
