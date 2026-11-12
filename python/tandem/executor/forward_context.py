from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import torch

from tandem.layers.ops import TorchOps

if TYPE_CHECKING:
    from tandem.cache.kv_cache import KVCache


@dataclass
class AttentionMetadata:
    """Per-step inputs of the reference attention backend, shared by all layers.

    The reference backend pads the batch to ``[B, Qmax]`` queries x ``[B, Lmax]`` keys.
    """

    slot_mapping: torch.Tensor  # [T] KV slot of every new token
    q_index: torch.Tensor  # [B, Qmax] flat token index per (request, query) - padded with 0
    q_valid: torch.Tensor  # [B, Qmax] bool
    kv_slots: torch.Tensor  # [B, Lmax] KV slot per (request, key position) - padded with 0
    attn_mask: torch.Tensor  # [B, 1, Qmax, Lmax] bool, True = may attend
    num_tokens: int


@dataclass
class ForwardContext:
    attn: Any  # backend-specific metadata (AttentionMetadata for the reference backend)
    kv_cache: "KVCache"
    attn_backend: Any
    moe_executor: Any
    positions: torch.Tensor = None  # [T] position of every token (RoPE)
    ops: Any = field(default_factory=TorchOps)  # tandem.layers.ops.TorchOps / FlashInferOps


def build_attention_metadata(
    seqs: Sequence[tuple[Sequence[int], int, int]],
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, AttentionMetadata]:
    """seqs: (block_ids, start, num_new) per request. Returns (positions, metadata)."""
    B = len(seqs)
    q_max = max(n for _, _, n in seqs)
    l_max = max(start + n for _, start, n in seqs)
    positions, slot_mapping = [], []
    q_index = torch.zeros(B, q_max, dtype=torch.long)
    q_valid = torch.zeros(B, q_max, dtype=torch.bool)
    kv_slots = torch.zeros(B, l_max, dtype=torch.long)
    starts = torch.empty(B, dtype=torch.long)
    lens = torch.empty(B, dtype=torch.long)
    offset = 0
    for i, (block_ids, start, n) in enumerate(seqs):
        bt = torch.tensor(block_ids, dtype=torch.long)
        ctx_len = start + n
        pos = torch.arange(ctx_len, dtype=torch.long)
        slots = bt[pos // block_size] * block_size + pos % block_size
        kv_slots[i, :ctx_len] = slots
        positions.append(pos[start:])
        slot_mapping.append(slots[start:])
        q_index[i, :n] = torch.arange(offset, offset + n)
        q_valid[i, :n] = True
        starts[i] = start
        lens[i] = ctx_len
        offset += n
    q_pos = starts[:, None] + torch.arange(q_max)[None, :]  # [B, Qmax]
    k_pos = torch.arange(l_max)[None, None, :]  # [1, 1, Lmax]
    mask = (k_pos <= q_pos[:, :, None]) & (k_pos < lens[:, None, None])
    mask[~q_valid] = False
    mask[..., 0] |= ~q_valid  # padded query rows attend to one key so softmax stays finite
    meta = AttentionMetadata(
        slot_mapping=torch.cat(slot_mapping).to(device),
        q_index=q_index.to(device),
        q_valid=q_valid.to(device),
        kv_slots=kv_slots.to(device),
        attn_mask=mask[:, None].to(device),
        num_tokens=offset,
    )
    return torch.cat(positions).to(device), meta
