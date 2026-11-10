from __future__ import annotations

from typing import Any

import torch
from torch import nn

from tandem.cache.kv_cache import KVCache
from tandem.executor.forward_context import ForwardContext, build_attention_metadata
from tandem.layers.sampler import sample_token, verify_draft
from tandem.scheduler.scheduler import SchedulerOutput


class ModelRunner:
    """Turns a ``SchedulerOutput`` into one forward pass plus sampling / verification."""

    def __init__(
        self,
        model: nn.Module,
        kv_cache: KVCache,
        block_size: int,
        device: torch.device,
        attn_backend: Any,
        moe_executor: Any,
    ) -> None:
        self.model = model
        self.kv_cache = kv_cache
        self.block_size = block_size
        self.device = device
        self.attn_backend = attn_backend
        self.moe_executor = moe_executor

    @torch.inference_mode()
    def execute(self, sched: SchedulerOutput) -> dict[str, list[int]]:
        input_ids: list[int] = []
        seqs = []
        logit_rows: list[int] = []
        logit_owners: list[tuple[int, int]] = []  # (scheduled index, number of rows)
        offset = 0
        for idx, s in enumerate(sched.scheduled):
            req, n = s.request, s.num_tokens
            start = req.num_computed_tokens
            input_ids.extend(req.token_ids_in_range(start, start + n))
            seqs.append((req.block_ids, start, n))
            if start + n >= req.num_tokens_with_spec:  # reaches the newest token -> needs logits
                rows = 1 + len(req.spec_token_ids)
                logit_rows.extend(range(offset + n - rows, offset + n))
                logit_owners.append((idx, rows))
            offset += n

        positions, attn_meta = build_attention_metadata(seqs, self.block_size, self.device)
        ctx = ForwardContext(attn_meta, self.kv_cache, self.attn_backend, self.moe_executor)
        ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        hidden = self.model(ids, positions, ctx)
        if not logit_rows:
            return {}
        logits = self.model.compute_logits(hidden[torch.tensor(logit_rows, device=self.device)])

        results: dict[str, list[int]] = {}
        row = 0
        for idx, rows in logit_owners:
            req = sched.scheduled[idx].request
            req_logits = logits[row : row + rows]
            row += rows
            if req.spec_token_ids:
                results[req.request_id] = verify_draft(req_logits, req.spec_token_ids, req.sampling_params, req.generator)
            else:
                results[req.request_id] = [sample_token(req_logits[0], req.sampling_params, req.generator)]
        return results
