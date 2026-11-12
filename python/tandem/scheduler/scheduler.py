"""Continuous batching scheduler with chunked prefill and prefix caching.

* Iteration-level scheduling / continuous batching (Orca, OSDI'22).
* Per-step token budget with decode-first, stall-free chunked prefill (Sarathi-Serve, OSDI'24).
* Unified "computed tokens" bookkeeping (vLLM V1): a request needs
  ``num_tokens_with_spec - num_computed_tokens`` tokens computed; prefill, chunked prefill,
  decode and speculative verification are the same operation.
* Prefix reuse through the radix cache (RadixAttention, SGLang).
* Preemption by recompute: the victim's computed blocks are inserted into the radix
  cache first, so resuming it is mostly a cache hit.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from tandem.cache.block_allocator import BlockAllocator
from tandem.cache.radix_cache import RadixCache
from tandem.engine.request import PLACEHOLDER, Request, RequestStatus
from tandem.utils import cdiv


@dataclass
class ScheduledRequest:
    request: Request
    num_tokens: int  # tokens computed this step, including draft tokens

    @property
    def num_draft_tokens(self) -> int:
        return len(self.request.spec_token_ids)


@dataclass
class SchedulerOutput:
    scheduled: list[ScheduledRequest] = field(default_factory=list)
    preempted: list[Request] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.scheduled

    @property
    def num_tokens(self) -> int:
        return sum(s.num_tokens for s in self.scheduled)


class Scheduler:
    def __init__(
        self,
        allocator: BlockAllocator,
        block_size: int,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        max_model_len: int,
        radix_cache: Optional[RadixCache] = None,
        policy: str = "decode_first",
    ) -> None:
        if policy not in ("decode_first", "prefill_first"):
            raise ValueError(f"unknown scheduling policy {policy!r}")
        # decode_first: every step carries all decodes, prefill chunks fill the rest of the
        #   token budget (Sarathi-Serve): first tokens come early, decodes slow down while
        #   prompts are being prefilled.
        # prefill_first: a step with prefill work carries no decodes (SGLang's default): bursts
        #   of new requests are prefilled before decoding resumes; lower TPOT, later first tokens.
        self.policy = policy
        self.allocator = allocator
        self.bs = block_size
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_model_len = max_model_len
        self.radix = radix_cache
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.num_preemptions = 0

    # ------------------------------------------------------------------ queue API
    def add_request(self, req: Request) -> None:
        req.status = RequestStatus.WAITING
        self.waiting.append(req)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def finish_request(self, req: Request, status: RequestStatus) -> None:
        """Finish a request from outside the step loop (stop string, abort)."""
        if req.finished:
            return
        if req in self.running:
            self.running.remove(req)
        elif req in self.waiting:
            self.waiting.remove(req)
        self._release(req, cache=status != RequestStatus.FINISHED_ABORTED)
        req.status = status
        req.finish_time = time.monotonic()

    # ------------------------------------------------------------- block helpers
    def _ensure_blocks(self, req: Request, num_tokens: int) -> bool:
        need = cdiv(num_tokens, self.bs) - len(req.block_ids)
        if need <= 0:
            return True
        if need > self.allocator.num_free and self.radix is not None:
            self.radix.evict(need - self.allocator.num_free)
        if need > self.allocator.num_free:
            return False
        req.block_ids.extend(self.allocator.allocate(need))
        return True

    def _match_prefix(self, req: Request) -> None:
        if self.radix is None or req.block_ids:
            return
        tokens = req.all_token_ids
        # Leave at least one token to compute so the step yields logits.
        max_match = (len(tokens) - 1) // self.bs * self.bs
        if max_match <= 0:
            return
        blocks, node = self.radix.match_prefix(tokens[:max_match])
        if not blocks:
            return
        self.allocator.incref(blocks)
        self.radix.inc_lock(node)
        req.block_ids = list(blocks)
        req.prefix_node = node
        req.num_computed_tokens = len(blocks) * self.bs
        req.num_cached_tokens = req.num_computed_tokens

    def _cache_computed_blocks(self, req: Request) -> None:
        """Insert the request's full computed blocks into the radix cache (request keeps running)."""
        if self.radix is None:
            return
        n_full = req.num_computed_tokens // self.bs
        if n_full == 0:
            return
        node, canonical = self.radix.insert(req.all_token_ids[: n_full * self.bs], req.block_ids[:n_full])
        for i, (old, new) in enumerate(zip(req.block_ids[:n_full], canonical)):
            if old != new:  # identical content already cached under another block: share it
                self.allocator.incref([new])
                self.allocator.decref([old])
                req.block_ids[i] = new
        self.radix.inc_lock(node)
        if req.prefix_node is not None:
            self.radix.dec_lock(req.prefix_node)
        req.prefix_node = node

    def _release(self, req: Request, cache: bool) -> None:
        if self.radix is not None and cache:
            n_full = min(req.num_computed_tokens // self.bs, len(req.block_ids))
            if n_full:
                self.radix.insert(req.all_token_ids[: n_full * self.bs], req.block_ids[:n_full])
        if self.radix is not None and req.prefix_node is not None:
            self.radix.dec_lock(req.prefix_node)
        req.prefix_node = None
        if req.block_ids:
            self.allocator.decref(req.block_ids)
        req.block_ids = []

    def _preempt(self, req: Request) -> None:
        self._release(req, cache=True)
        req.num_computed_tokens = 0
        req.spec_token_ids = []
        req.status = RequestStatus.WAITING
        req.num_preemptions += 1
        self.num_preemptions += 1
        self.waiting.appendleft(req)

    # ------------------------------------------------------------------ schedule
    def schedule(self) -> SchedulerOutput:
        if self.policy == "prefill_first" and (self.waiting or any(not r.is_decoding for r in self.running)):
            out = self._schedule(decodes=False)
            if not out.is_empty:  # nothing to prefill now (e.g. no KV room): decode below
                return out
        return self._schedule(decodes=True)

    def _schedule(self, decodes: bool) -> SchedulerOutput:
        out = SchedulerOutput()
        budget = self.max_num_batched_tokens

        # 1) running requests, decode first (stall-free batching), then prefill chunks
        order = ([r for r in self.running if r.is_decoding] if decodes else []) + [
            r for r in self.running if not r.is_decoding]
        i = 0
        while i < len(order) and budget > 0:
            req = order[i]
            if self._awaiting_last_token(req):  # overlap: the in-flight token ends it
                i += 1
                continue
            if req.spec_token_ids:
                req.spec_token_ids = req.spec_token_ids[: max(0, budget - 1)]
            n = min(req.num_tokens_with_spec - req.num_computed_tokens, budget)
            if n <= 0:
                i += 1
                continue
            preempted_self = False
            while not self._ensure_blocks(req, req.num_computed_tokens + n):
                victim = order.pop()  # lowest priority, not scheduled yet
                self.running.remove(victim)
                self._preempt(victim)
                out.preempted.append(victim)
                if victim is req:
                    preempted_self = True
                    break
            if preempted_self:
                break
            out.scheduled.append(ScheduledRequest(req, n))
            budget -= n
            i += 1

        # 2) admit waiting requests (skip this step if we just had to preempt)
        while not out.preempted and self.waiting and budget > 0 and len(self.running) < self.max_num_seqs:
            req = self.waiting[0]
            self._match_prefix(req)
            n = min(req.num_tokens - req.num_computed_tokens, budget)
            if not self._ensure_blocks(req, req.num_computed_tokens + n):
                self._release(req, cache=False)
                req.num_computed_tokens = 0
                break
            self.waiting.popleft()
            req.status = RequestStatus.RUNNING
            self.running.append(req)
            out.scheduled.append(ScheduledRequest(req, n))
            budget -= n
        return out

    # -------------------------------------------------------------------- update
    def update_from_output(self, out: SchedulerOutput, sampled: dict[str, list[int]]) -> list[Request]:
        """Apply one step's results. Returns requests that produced tokens or finished."""
        now = time.monotonic()
        updated: list[Request] = []
        for s in out.scheduled:
            req = s.request
            req.new_token_ids = []
            new_tokens = sampled.get(req.request_id)
            num_drafts = len(req.spec_token_ids)
            was_prefilling = req.num_computed_tokens < req.num_prompt_tokens
            if new_tokens is None:  # intermediate prefill chunk: no logits this step
                req.num_computed_tokens += s.num_tokens
                if self.radix is not None:
                    self._cache_computed_blocks(req)
                continue
            num_accepted = len(new_tokens) - 1
            req.num_computed_tokens += s.num_tokens - (num_drafts - num_accepted)
            req.num_draft_tokens += num_drafts
            req.num_accepted_tokens += num_accepted
            req.spec_token_ids = []
            if was_prefilling and self.radix is not None:
                self._cache_computed_blocks(req)
            if req.first_token_time is None:
                req.first_token_time = now
            self._append_tokens(req, new_tokens)
            updated.append(req)
            if req.finished:
                self.running.remove(req)
                self._release(req, cache=True)
                req.finish_time = now
        return updated

    def _append_tokens(self, req: Request, tokens: list[int]) -> None:
        sp = req.sampling_params
        for t in tokens:
            req.output_token_ids.append(t)
            req.num_resolved_outputs = len(req.output_token_ids)
            req.new_token_ids.append(t)
            if t in req.eos_token_ids or t in req.stop_token_ids:
                req.status = RequestStatus.FINISHED_STOPPED
                return
            if req.num_output_tokens >= sp.max_tokens or req.num_tokens >= self.max_model_len:
                req.status = RequestStatus.FINISHED_LENGTH
                return

    # ----------------------------------------------------------- overlap mode
    # The engine launches step N before it has the tokens of step N-1 (they stay on the GPU
    # and become step N's input there). The scheduler books each launched step right away,
    # with a PLACEHOLDER output token per request that will receive one, and fills the
    # placeholders in when the step's tokens reach the host, one step later. A request that
    # turns out to have finished (EOS, stop token, length, stop string, abort) may already
    # sit in the next launched step: that step's token for it is dropped.

    def _awaiting_last_token(self, req: Request) -> bool:
        """Its in-flight token is the last one it may produce (only with placeholders)."""
        return req.num_resolved_outputs < req.num_output_tokens and (
            req.num_output_tokens >= req.sampling_params.max_tokens or req.num_tokens >= self.max_model_len
        )

    def on_launch(self, out: SchedulerOutput) -> None:
        for s in out.scheduled:
            req = s.request
            was_prefilling = req.num_computed_tokens < req.num_prompt_tokens
            produces_token = req.num_computed_tokens + s.num_tokens >= req.num_tokens
            req.num_computed_tokens += s.num_tokens
            if was_prefilling and self.radix is not None:
                self._cache_computed_blocks(req)
            if produces_token:
                req.output_token_ids.append(PLACEHOLDER)

    def update_resolved(self, out: SchedulerOutput, sampled: dict[str, list[int]]) -> list[Request]:
        now = time.monotonic()
        updated: list[Request] = []
        for s in out.scheduled:
            req = s.request
            tokens = sampled.get(req.request_id)
            if tokens is None:  # intermediate prefill chunk: no token
                continue
            req.new_token_ids = []
            if req.finished:  # ended while this step was in flight
                del req.output_token_ids[req.num_resolved_outputs :]
                continue
            t = tokens[0]
            req.output_token_ids[req.num_resolved_outputs] = t
            req.num_resolved_outputs += 1
            req.new_token_ids.append(t)
            if req.first_token_time is None:
                req.first_token_time = now
            updated.append(req)
            if t in req.eos_token_ids or t in req.stop_token_ids:
                status = RequestStatus.FINISHED_STOPPED
            elif (req.num_resolved_outputs >= req.sampling_params.max_tokens
                  or req.num_prompt_tokens + req.num_resolved_outputs >= self.max_model_len):
                status = RequestStatus.FINISHED_LENGTH
            else:
                continue
            del req.output_token_ids[req.num_resolved_outputs :]  # the next step's token is dropped
            req.num_computed_tokens = min(req.num_computed_tokens, req.num_tokens)
            if req in self.running:
                self.running.remove(req)
            elif req in self.waiting:  # preempted while its token was in flight
                self.waiting.remove(req)
            self._release(req, cache=True)
            req.status = status
            req.finish_time = now
        return updated
