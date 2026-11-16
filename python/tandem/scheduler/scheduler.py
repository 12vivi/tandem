"""Continuous batching scheduler with chunked prefill and prefix caching.

* Iteration-level scheduling / continuous batching (Orca, OSDI'22).
* Per-step token budget with decode-first, stall-free chunked prefill (Sarathi-Serve, OSDI'24).
* Unified "computed tokens" bookkeeping (vLLM V1): a request needs
  ``num_tokens_with_spec - num_computed_tokens`` tokens computed; prefill, chunked prefill,
  decode and speculative verification are the same operation.
* Prefix reuse through the radix cache (RadixAttention, SGLang).
* Preemption by recompute: the victim's computed blocks are inserted into the radix
  cache first, so resuming it is mostly a cache hit.
* With a host-memory tier (design doc §4.4): a request whose cached prefix continues in host
  memory is admitted in the same step that starts copying those pages back; the step waits
  for each layer's KV just before that layer (``SchedulerOutput.load_handles``), so the copy
  overlaps the forward pass (layer-wise pipelining, HiCache). Completed prompts are written
  through to host memory, so evicting them from the GPU later costs nothing; a preempted
  request is written out whole (swap-out). Generated tokens of finished requests are not
  written through: they are rarely the prefix of a later prompt (chat templates re-render
  replies) and would push reusable prompts out of host memory; they reach it only if free
  host pages are left when they are evicted from the GPU.
* A waiting request is admitted only if the part of its prefix that is in host memory and its
  first chunk fit in free or evictable blocks; otherwise it waits whole (a partial load would
  be recomputed, and repeated on every step while it is blocked).
* Short prefills first, in steps of their own: a waiting request that has at most SHORT_PREFILL
  tokens left to compute once its cached prefix (GPU or host memory) is reused is admitted
  before the running prefill chunks and the waiting requests ahead of it, if its KV fits; the
  shortest first, until the step holds SHORT_STEP_TOKENS; and that step carries no chunk of a
  long prompt (only decodes, with decode_first). Otherwise a request that only needs its cached
  document loaded waits behind every chunk of a long uncached prompt, and its first token waits
  for the 2048-token chunk it shares a step with. Above ~256 tokens a prefill step is
  compute-bound, so splitting it costs no throughput and the first requests finish earlier.
  When a short request does not fit, the running prefills with the most left (more than
  SHORT_PREFILL) are preempted for it, i.e. swapped out to host memory; it is admitted once
  their copies have landed (eviction waits for them), and they resume after it.
"""

from __future__ import annotations

import time
import itertools
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
    # host -> GPU loads (tandem.cache.host_pool.LoadHandle) this step's prefixes wait for
    load_handles: list = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.scheduled

    @property
    def num_tokens(self) -> int:
        return sum(s.num_tokens for s in self.scheduled)


class Scheduler:
    SHORT_PREFILL = 512  # tokens left after the cached prefix: admitted ahead of long prefills
    SHORT_LOOKAHEAD = 8  # waiting requests examined for that per step
    SHORT_STEP_TOKENS = 256  # a step of short prefills takes more of them only below this

    def __init__(
        self,
        allocator: BlockAllocator,
        block_size: int,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        max_model_len: int,
        radix_cache: Optional[RadixCache] = None,
        policy: str = "prefill_first",
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

    def _match_prefix(self, req: Request) -> list:
        """Reuse the cached prefix of a waiting request, loading the part that is only in host
        memory. Returns the loads (LoadHandle) its prefix still waits for."""
        if self.radix is None or req.block_ids:
            return []
        tokens = req.all_token_ids
        # Leave at least one token to compute so the step yields logits.
        max_match = (len(tokens) - 1) // self.bs * self.bs
        if max_match <= 0:
            return []
        m = self.radix.match(tokens[:max_match], count=not req.prefix_counted)  # once per request
        req.prefix_counted = True
        if m.host_nodes and self.radix.host is not None:
            self.radix.inc_lock(m.host_nodes[-1])  # nothing on the path may be evicted to make room
            loaded, _ = self.radix.load(m.host_nodes)
            self.radix.dec_lock(m.host_nodes[-1])
            if loaded:  # now in L1 (the copy may still be running: m.pending)
                m = self.radix.match(tokens[:max_match], count=False)
        blocks, node = m.blocks, m.node
        if not blocks:
            return []
        self.allocator.incref(blocks)
        self.radix.inc_lock(node)
        req.block_ids = list(blocks)
        req.prefix_node = node
        req.num_computed_tokens = len(blocks) * self.bs
        if req.num_preemptions == 0:  # the hit of the first admission (as vLLM reports it), not its own
            req.num_cached_tokens = req.num_computed_tokens  # blocks found again after a preemption
        return m.pending

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
        if req.num_computed_tokens >= req.num_prompt_tokens:  # the prompt is complete: keep a host copy
            self.radix.write_through(node)

    def _release(self, req: Request, cache: bool, write: bool = False) -> None:
        """``write``: also start copying the request's KV to host memory (swap-out on preemption)."""
        if self.radix is not None and cache:
            n_full = min(req.num_computed_tokens // self.bs, len(req.block_ids))
            if n_full:
                node, _ = self.radix.insert(req.all_token_ids[: n_full * self.bs], req.block_ids[:n_full])
                if write:
                    self.radix.write_through(node)
        if self.radix is not None and req.prefix_node is not None:
            self.radix.dec_lock(req.prefix_node)
        req.prefix_node = None
        if req.block_ids:
            self.allocator.decref(req.block_ids)
        req.block_ids = []

    def _preempt(self, req: Request) -> None:
        self._release(req, cache=True, write=True)
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
        order = ([r for r in self.running if r.is_decoding] if decodes else []) + [
            r for r in self.running if not r.is_decoding]

        # 0) waiting requests whose prefix is (nearly) all cached, in a step of their own: no
        # chunk of a long prompt, no other admission (the running decodes still go along)
        short_step = False
        if self.radix is not None and self.waiting:
            budget = self._admit_short(out, budget)
            short_step = not out.is_empty
            if short_step:
                order = [r for r in order if r.is_decoding]
                budget = max(budget, len(order))  # decodes take one token each

        # 1) running requests, decode first (stall-free batching), then prefill chunks
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

        # 2) admit waiting requests (skip this step if we just had to preempt, or if it is a
        # step of short prefills)
        while (not out.preempted and not short_step and self.waiting and budget > 0
               and len(self.running) < self.max_num_seqs):
            req = self.waiting[0]
            if self.radix is not None and not req.block_ids:
                need, room = self._room(req, budget)
                if need > room:  # its cached prefix and first chunk do not fit: it waits whole
                    break
            loads = self._match_prefix(req)
            n = min(req.num_tokens - req.num_computed_tokens, budget)
            if not self._ensure_blocks(req, req.num_computed_tokens + n):
                self._release(req, cache=False)
                req.num_computed_tokens = 0
                break
            self.waiting.popleft()
            req.status = RequestStatus.RUNNING
            self.running.append(req)
            out.scheduled.append(ScheduledRequest(req, n))
            out.load_handles.extend(h for h in loads if h not in out.load_handles)
            budget -= n
        return out

    def _admit_short(self, out: SchedulerOutput, budget: int) -> int:
        """Among the first SHORT_LOOKAHEAD waiting requests, admit those that have at most
        SHORT_PREFILL tokens left once their cached prefix is reused, shortest first, while the
        step holds fewer than SHORT_STEP_TOKENS: ahead of the running prefill chunks and of the
        waiting requests in front of them. One whose blocks do not fit (free or evictable) has
        long running prefills preempted for it (_preempt_for). Returns the budget left."""
        shorts = []
        for k, req in enumerate(itertools.islice(self.waiting, self.SHORT_LOOKAHEAD)):
            tokens = req.all_token_ids
            max_match = (len(tokens) - 1) // self.bs * self.bs
            if max_match <= 0 or req.block_ids:
                continue
            m = self.radix.match(tokens[:max_match], count=False)  # a look only
            left = len(tokens) - (len(m.blocks) + m.num_host_blocks) * self.bs
            if left <= self.SHORT_PREFILL:
                shorts.append((left, k, req))
        step_tokens = 0
        for left, _, req in sorted(shorts, key=lambda x: x[:2]):
            if budget < left or len(self.running) >= self.max_num_seqs or step_tokens >= self.SHORT_STEP_TOKENS:
                break
            need, room = self._room(req, budget)
            if need > room and not self._preempt_for(need - room, out):
                continue
            loads = self._match_prefix(req)
            n = min(req.num_tokens - req.num_computed_tokens, budget)
            if not self._ensure_blocks(req, req.num_computed_tokens + n):
                self._release(req, cache=False)
                req.num_computed_tokens = 0
                continue
            self.waiting.remove(req)
            req.status = RequestStatus.RUNNING
            self.running.append(req)
            out.scheduled.append(ScheduledRequest(req, n))
            out.load_handles.extend(h for h in loads if h not in out.load_handles)
            budget -= n
            step_tokens += n
        return budget

    def _room(self, req: Request, budget: int) -> tuple[int, int]:
        """(blocks to allocate to admit ``req``: the part of its cached prefix that is in host
        memory and its first chunk; blocks that are free or evictable, not counting its own
        prefix in GPU memory)."""
        tokens = req.all_token_ids
        max_match = (len(tokens) - 1) // self.bs * self.bs
        m = self.radix.match(tokens[:max_match], count=False) if max_match > 0 else None
        cached = (len(m.blocks) + m.num_host_blocks) * self.bs if m else 0
        first = min(len(tokens) - cached, budget)
        need = cdiv(cached + first, self.bs) - (len(m.blocks) if m else 0)
        return need, self.allocator.num_free + self.radix.num_evictable_blocks(keep=m.node if m else None)

    def _preempt_for(self, blocks: int, out: SchedulerOutput) -> bool:
        """Preempt running requests that are still prefilling with more than SHORT_PREFILL
        tokens left, most left first, until about ``blocks`` blocks become evictable (once their
        copies to host memory land). False, and nothing preempted, if that is not enough."""
        if self.radix.host is None:
            return False
        long = sorted((r for r in self.running if not r.is_decoding
                       and r.num_prompt_tokens - r.num_computed_tokens > self.SHORT_PREFILL),
                      key=lambda r: r.num_prompt_tokens - r.num_computed_tokens, reverse=True)
        victims, gain = [], 0
        for r in long:
            if gain >= blocks:
                break
            victims.append(r)
            gain += len(r.block_ids)
        if gain < blocks:
            return False
        for r in victims:
            self.running.remove(r)
            self._preempt(r)
            out.preempted.append(r)
        return True

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
