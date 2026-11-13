from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch import nn

from tandem.cache.kv_cache import KVCache
from tandem.engine.request import PLACEHOLDER, Request
from tandem.executor.forward_context import ForwardContext
from tandem.executor.step_timer import TIMER
from tandem.layers.ops import TorchOps
from tandem.layers.sampler import sample_token, sample_tokens, verify_draft
from tandem.scheduler.scheduler import SchedulerOutput


@dataclass
class InFlightStep:
    """A launched step whose sampled tokens may still be on their way to the host."""

    sched: SchedulerOutput
    owners: list[Request] = field(default_factory=list)  # one sampled token each, in row order
    tokens_host: Optional[torch.Tensor] = None  # pinned copy of the sampled tokens
    event: Any = None  # CUDA event recorded after the copy (None on the CPU)
    results: Optional[dict[str, list[int]]] = None  # already known (synchronous paths)

    def resolve(self) -> dict[str, list[int]]:
        if self.results is None:
            if self.event is not None:
                self.event.synchronize()
            self.results = {r.request_id: [t] for r, t in zip(self.owners, self.tokens_host.tolist())}
        return self.results


class ModelRunner:
    """Turns a ``SchedulerOutput`` into one forward pass plus sampling / verification.

    ``launch`` enqueues a step and returns at once with an ``InFlightStep``: tokens are
    sampled on the device and copied to pinned host memory behind a CUDA event. A request
    whose newest token is still a PLACEHOLDER (overlap scheduling) gets it on the device from
    the previous step's sampled tokens. ``execute`` is launch + resolve (synchronous engine,
    speculative decoding). Decode-only steps replay a CUDA graph when ``graph_runner`` is set.
    """

    def __init__(
        self,
        model: nn.Module,
        kv_cache: KVCache,
        block_size: int,
        device: torch.device,
        attn_backend: Any,
        moe_executor: Any,
        ops: Any = None,
        graph_runner: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.kv_cache = kv_cache
        self.block_size = block_size
        self.device = device
        self.attn_backend = attn_backend
        self.moe_executor = moe_executor
        self.ops = ops if ops is not None else TorchOps()
        self.graph_runner = graph_runner
        self.num_graph_steps = 0
        self.prev_tokens: Optional[torch.Tensor] = None  # sampled tokens of the latest launched step
        # two pinned buffers: step N copies into one while the host may still read step N-1's
        self._host_tokens: list[Optional[torch.Tensor]] = [None, None]
        self._host_turn = 0

    def execute(self, sched: SchedulerOutput) -> dict[str, list[int]]:
        return self.launch(sched).resolve()

    @torch.inference_mode()
    def launch(self, sched: SchedulerOutput) -> InFlightStep:
        input_ids: list[int] = []
        seqs = []
        logit_rows: list[int] = []
        owners: list[tuple[Request, int]] = []  # (request, number of logit rows)
        fix_pos: list[int] = []  # input positions holding a PLACEHOLDER ...
        fix_rows: list[int] = []  # ... and the row of prev_tokens that holds its value
        offset = 0
        for s in sched.scheduled:
            req, n = s.request, s.num_tokens
            start = req.num_computed_tokens
            ids = req.token_ids_in_range(start, start + n)
            if ids[-1] == PLACEHOLDER:  # only the newest token can be in flight
                fix_pos.append(offset + n - 1)
                fix_rows.append(req.inflight_row)
            if n > 1 and PLACEHOLDER in ids[:-1]:
                raise RuntimeError(f"{req.request_id}: unresolved token inside a multi-token chunk")
            input_ids.extend(ids)
            seqs.append((req.block_ids, start, n))
            if start + n >= req.num_tokens_with_spec:  # reaches the newest token -> needs logits
                rows = 1 + len(req.spec_token_ids)
                logit_rows.extend(range(offset + n - rows, offset + n))
                owners.append((req, rows))
            offset += n
        TIMER.mark("inputs")
        fixups = None
        if fix_pos:
            fixups = (torch.tensor(fix_pos, dtype=torch.long).to(self.device, non_blocking=True),
                      self.prev_tokens[torch.tensor(fix_rows, dtype=torch.long).to(self.device, non_blocking=True)])
            TIMER.mark("fixups")

        decode_only = offset == len(seqs)  # one new token per request
        if self.graph_runner is not None and decode_only and self.graph_runner.can_run(len(seqs)):
            hidden = self.graph_runner.run(seqs, input_ids, fixups)
            self.num_graph_steps += 1
        else:
            positions, attn_meta = self.attn_backend.prepare(seqs, self.block_size, self.device)
            ctx = ForwardContext(attn_meta, self.kv_cache, self.attn_backend, self.moe_executor, ops=self.ops)
            ids = torch.tensor(input_ids, dtype=torch.long).to(self.device, non_blocking=True)
            if fixups is not None:
                ids[fixups[0]] = fixups[1]
            hidden = self.model(ids, positions, ctx)
            TIMER.mark("eager_forward")
        if not logit_rows:
            return InFlightStep(sched, results={})
        if len(logit_rows) != offset:
            hidden = hidden[torch.tensor(logit_rows, dtype=torch.long).to(self.device, non_blocking=True)]
        reqs = [r for r, _ in owners]
        # all-greedy steps sample with one argmax: on the model-dtype logits, no float32 copy
        greedy = all(r.sampling_params.greedy and not r.spec_token_ids for r in reqs)
        logits = self.model.compute_logits(hidden, upcast=not greedy)

        if any(r.spec_token_ids for r in reqs):  # speculative verification: synchronous
            return InFlightStep(sched, results=self._verify(logits, owners))
        tokens = sample_tokens(logits, [r.sampling_params for r in reqs], [r.generator for r in reqs])
        TIMER.mark("logits_sample")
        for i, r in enumerate(reqs):
            r.inflight_row = i
        self.prev_tokens = tokens
        if tokens.device.type != "cuda":
            return InFlightStep(sched, reqs, tokens_host=tokens)
        turn = self._host_turn = 1 - self._host_turn
        buf = self._host_tokens[turn]
        if buf is None or buf.numel() < tokens.numel():
            buf = self._host_tokens[turn] = torch.empty(max(256, tokens.numel()), dtype=torch.long, pin_memory=True)
        host = buf[: tokens.numel()]
        host.copy_(tokens, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        return InFlightStep(sched, reqs, tokens_host=host, event=event)

    @staticmethod
    def _verify(logits: torch.Tensor, owners: list[tuple[Request, int]]) -> dict[str, list[int]]:
        results: dict[str, list[int]] = {}
        row = 0
        for req, rows in owners:
            req_logits = logits[row : row + rows]
            row += rows
            if req.spec_token_ids:
                results[req.request_id] = verify_draft(req_logits, req.spec_token_ids, req.sampling_params, req.generator)
            else:
                results[req.request_id] = [sample_token(req_logits[0], req.sampling_params, req.generator)]
        return results
