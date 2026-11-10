from __future__ import annotations

import torch


class KVCache:
    """Paged KV storage, one K and one V tensor per layer.

    Layout ``[num_blocks * block_size, num_kv_heads, head_dim]`` ("flat slots"): token at
    position p of a request lives in slot ``block_ids[p // bs] * bs + p % bs``. Viewed as
    ``[num_blocks, block_size, num_kv_heads, head_dim]`` it matches FlashInfer's NHD paged
    layout, so the GPU attention backend can reuse the same buffers.
    """

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        shape = (num_blocks * block_size, num_kv_heads, head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)]

    @staticmethod
    def bytes_per_block(num_layers: int, block_size: int, num_kv_heads: int, head_dim: int, dtype: torch.dtype) -> int:
        elem = torch.empty((), dtype=dtype).element_size()
        return 2 * num_layers * block_size * num_kv_heads * head_dim * elem
