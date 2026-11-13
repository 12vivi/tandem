from __future__ import annotations

import itertools
import logging
import time
from typing import Any, Optional, Sequence, Union

import torch
from torch import nn

from tandem.cache.block_allocator import BlockAllocator
from tandem.cache.kv_cache import KVCache
from tandem.cache.radix_cache import RadixCache
from tandem.config import EngineConfig
from tandem.engine.detokenizer import detokenize_step
from tandem.engine.request import Request, RequestOutput, RequestStatus
from tandem.executor.step_timer import TIMER, TRACE
from tandem.executor.model_runner import ModelRunner
from tandem.layers.attention import TorchAttentionBackend
from tandem.layers.moe import LocalMoEExecutor
from tandem.layers.ops import make_ops
from tandem.model_config import ModelConfig
from tandem.models import build_model
from tandem.models.loader import cast_parameters, load_weights
from tandem.sampling_params import SamplingParams
from tandem.scheduler.scheduler import Scheduler
from tandem.spec.ngram import NgramProposer
from tandem.utils import cdiv, resolve_device, resolve_dtype

logger = logging.getLogger(__name__)

PromptType = Union[str, Sequence[int]]


class LLMEngine:
    """Synchronous engine: ``add_request`` + ``step`` (or ``generate`` for offline use)."""

    def __init__(
        self,
        config: EngineConfig,
        *,
        model_config: Optional[ModelConfig] = None,
        model: Optional[nn.Module] = None,
        tokenizer: Any = None,
        load_tokenizer: bool = True,
    ) -> None:
        self.config = config
        self.model_config = model_config or ModelConfig.from_pretrained(config.model)
        mc = self.model_config
        self.device = resolve_device(config.device)
        self.dtype = resolve_dtype(config.dtype, self.device, mc.torch_dtype)
        if config.cpu_threads > 0:
            torch.set_num_threads(config.cpu_threads)
        torch.manual_seed(config.seed)
        self.max_model_len = config.max_model_len or min(mc.max_position_embeddings, 32768)

        # ---------------------------------------------------------------- model
        if model is None:
            model = build_model(mc, self.max_model_len)
            cast_parameters(model, self.dtype)
            load_weights(model, config.model)
        else:
            cast_parameters(model, self.dtype)
        self.hybrid = config.hybrid.enabled and mc.is_moe
        if self.hybrid:
            self._place_for_hybrid(model)
        else:
            model.to(self.device)
        self.model = model.eval()

        # ------------------------------------------------------------- KV cache
        bs = config.block_size
        per_block = KVCache.bytes_per_block(mc.num_hidden_layers, bs, mc.num_key_value_heads, mc.head_dim, self.dtype)
        num_blocks = config.num_kv_blocks or max(1, int(config.kv_cache_memory_gb * 2**30 // per_block))
        # one extra block that the allocator never hands out: padding rows of CUDA-graph
        # batches write their dummy KV there
        self.scratch_block = num_blocks
        self.kv_cache = KVCache(mc.num_hidden_layers, num_blocks + 1, bs, mc.num_key_value_heads, mc.head_dim,
                                self.dtype, self.device)
        self.allocator = BlockAllocator(num_blocks)
        self.radix = RadixCache(bs, self.allocator) if config.enable_prefix_caching else None
        self.scheduler = Scheduler(
            self.allocator, bs, config.max_num_seqs, config.max_num_batched_tokens, self.max_model_len, self.radix,
            policy=config.schedule_policy,
        )

        # -------------------------------------------------------------- executors
        self.moe_executor = self._build_moe_executor() if self.hybrid else LocalMoEExecutor()
        self.backend_name = self._pick_backend()
        if self.backend_name == "flashinfer":
            from tandem.layers.flashinfer_attention import FlashInferAttentionBackend

            attn_backend = FlashInferAttentionBackend(mc.num_attention_heads, mc.num_key_value_heads, mc.head_dim,
                                                      bs, self.dtype, self.device)
            if config.small_batch_attention and config.fused_prologue:
                attn_backend.enable_triton_decode()
        else:
            attn_backend = TorchAttentionBackend()
        ops = make_ops(self.backend_name, fused_prologue=config.fused_prologue and self.backend_name == "flashinfer")
        self.num_gemv_layers = self.num_fused_layers = 0
        if self.backend_name == "flashinfer" and config.small_batch_gemv and self.dtype in (torch.float16, torch.bfloat16):
            from tandem.layers import triton_gemv

            self.num_gemv_layers = triton_gemv.install(self.model, triton_gemv.table_for(self.device, self.model))
            self.num_fused_layers = triton_gemv.install_fused(self.model, triton_gemv.fused_table_for(self.device, self.model))
        graph_runner = None
        # MoE routing syncs with the host (expert ids) and hybrid runs experts on the CPU:
        # CUDA graphs are for dense models
        if self.backend_name == "flashinfer" and config.cuda_graph and not mc.is_moe:
            from tandem.executor.cuda_graph import CudaGraphRunner

            graph_runner = CudaGraphRunner(
                self.model, self.kv_cache, attn_backend, ops, self.moe_executor,
                max_bs=min(config.cuda_graph_max_bs, config.max_num_seqs), scratch_block=self.scratch_block,
                max_pages_per_seq=cdiv(self.max_model_len, bs),
            )
        self.runner = ModelRunner(self.model, self.kv_cache, bs, self.device, attn_backend, self.moe_executor,
                                  ops=ops, graph_runner=graph_runner)
        spec = config.speculative
        self.proposer = NgramProposer(spec.ngram_min, spec.ngram_max) if spec.enabled and spec.method == "ngram" else None
        if spec.enabled and self.proposer is None:
            raise NotImplementedError(f"speculative method {spec.method!r} is not implemented yet")
        # drafts depend on the latest tokens, so speculative decoding keeps the synchronous loop
        self.overlap = config.overlap_schedule and self.proposer is None
        self.inflight = None  # launched step whose tokens are not processed yet (overlap)

        self.tokenizer = tokenizer
        if self.tokenizer is None and load_tokenizer:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer or config.model)
        self.requests: dict[str, Request] = {}
        self._ids = itertools.count()
        self.num_steps = 0
        logger.info(
            "engine ready: %s on %s/%s, %d KV blocks x %d tokens, hybrid=%s, backend=%s, cuda_graph=%s",
            mc.architecture, self.device, self.dtype, num_blocks, bs, self.hybrid, self.backend_name,
            graph_runner is not None,
        )

    # ------------------------------------------------------------------ setup
    def _pick_backend(self) -> str:
        name = self.config.attention_backend
        fast_ok = self.device.type == "cuda" and self.dtype in (torch.float16, torch.bfloat16) and not self.hybrid
        if name == "auto":
            if not fast_ok:
                return "torch"
            try:
                import flashinfer  # noqa: F401
            except ImportError:
                logger.warning("flashinfer is not installed: using the PyTorch reference attention")
                return "torch"
            return "flashinfer"
        if name == "flashinfer" and not fast_ok:
            raise ValueError("the flashinfer backend needs a CUDA device, float16 / bfloat16 and no hybrid MoE")
        if name not in ("torch", "flashinfer"):
            raise ValueError(f"unknown attention backend {name!r}")
        return name

    def _place_for_hybrid(self, model: nn.Module) -> None:
        """Everything but routed experts goes to the device; experts stay on the CPU."""
        dev = self.device
        model.rotary_emb.to(dev)
        model.model.embed_tokens.to(dev)
        model.model.norm.to(dev)
        if model.lm_head is not None:
            model.lm_head.to(dev)
        for layer in model.model.layers:
            layer.self_attn.to(dev)
            layer.input_layernorm.to(dev)
            layer.post_attention_layernorm.to(dev)
            if layer.is_moe:
                layer.mlp.gate.to(dev)
            else:
                layer.mlp.to(dev)

    def _build_moe_executor(self):
        from tandem.hybrid.moe_executor import HybridMoEExecutor
        from tandem.planner.cost_model import ExpertCostModel
        from tandem.planner.hardware import PRESETS, load_profile
        from tandem.planner.placement import footprint, make_plan

        hc = self.config.hybrid
        hw = load_profile(hc.hw_profile)
        if hw.gpu is None:  # CPU-only machine: simulate a GPU profile so the planner logic still runs
            logger.warning("no GPU in hardware profile %r; using preset 'rtx4090-desktop' for cost estimates", hw.name)
            hw = PRESETS["rtx4090-desktop"]
        elem = torch.empty((), dtype=self.dtype).element_size()
        fp = footprint(self.model_config)
        slots = hc.gpu_expert_slots
        if slots is None:
            plan = make_plan(self.model_config, hw, bytes_per_param=elem, vram_budget_gb=hc.vram_budget_gb,
                             kv_gb=self.config.kv_cache_memory_gb)
            slots = plan.expert_slots_per_layer if plan.expert_slots_per_layer is not None else self.model_config.num_experts
            logger.info("placement plan:\n%s", plan.explain())
        cost = ExpertCostModel(hw, fp.expert, elem)
        return HybridMoEExecutor(self.model.moe_blocks(), self.device, slots, cost, hc.cpu_threads, hc.reuse_bonus)

    # ---------------------------------------------------------------- requests
    def add_request(
        self,
        prompt: Optional[PromptType] = None,
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        prompt_token_ids: Optional[Sequence[int]] = None,
    ) -> str:
        sp = sampling_params or SamplingParams()
        text: Optional[str] = None
        if prompt_token_ids is None:
            if isinstance(prompt, str):
                if self.tokenizer is None:
                    raise ValueError("text prompts need a tokenizer")
                text = prompt
                prompt_token_ids = self.tokenizer.encode(prompt)
            elif prompt is not None:
                prompt_token_ids = list(prompt)
            else:
                raise ValueError("either prompt or prompt_token_ids is required")
        if not prompt_token_ids:
            raise ValueError("empty prompt")
        if len(prompt_token_ids) >= self.max_model_len:
            raise ValueError(f"prompt has {len(prompt_token_ids)} tokens; max_model_len is {self.max_model_len}")
        rid = request_id or f"req-{next(self._ids)}"
        if rid in self.requests:
            raise ValueError(f"duplicate request id {rid}")
        req = Request(rid, list(prompt_token_ids), sp, self.model_config.eos_token_ids, prompt=text)
        if sp.seed is not None:
            req.generator = torch.Generator(device=self.device).manual_seed(sp.seed)
        self.requests[rid] = req
        self.scheduler.add_request(req)
        return rid

    def abort_request(self, request_id: str) -> None:
        req = self.requests.pop(request_id, None)
        if req is not None and not req.finished:
            self.scheduler.finish_request(req, RequestStatus.FINISHED_ABORTED)

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished() or self.inflight is not None

    # -------------------------------------------------------------------- step
    def step(self) -> list[RequestOutput]:
        if self.overlap:
            return self._step_overlap()
        sched = self.scheduler.schedule()
        if sched.is_empty:
            return []
        sampled = self.runner.execute(sched)
        updated = self.scheduler.update_from_output(sched, sampled)
        self.num_steps += 1
        outputs = self._emit(updated)
        if self.proposer is not None:
            self._propose_drafts()
        return outputs

    def _step_overlap(self) -> list[RequestOutput]:
        """Launch step N, then finish step N-1 on the host while the GPU runs step N: its
        tokens become N's inputs on the device, so the host never waits before launching."""
        TIMER.start()
        sched = self.scheduler.schedule()
        TIMER.mark("schedule")
        if TRACE and not sched.is_empty:
            pre = [(s.request.request_id, s.num_tokens) for s in sched.scheduled if s.num_tokens > 1]
            TRACE("launch", prefill=pre, decodes=len(sched.scheduled) - len(pre),
                  waiting=len(self.scheduler.waiting))
        launched = None
        if not sched.is_empty:
            launched = self.runner.launch(sched)
            self.scheduler.on_launch(sched)
            self.num_steps += 1
            TIMER.mark("book")
        prev, self.inflight = self.inflight, launched
        if prev is None:
            TIMER.end_step()
            return []
        sampled = prev.resolve()
        TIMER.mark("wait_gpu")
        updated = self.scheduler.update_resolved(prev.sched, sampled)
        TIMER.mark("update")
        outputs = self._emit(updated)
        TIMER.mark("emit")
        TIMER.end_step()
        return outputs

    def _emit(self, updated: list[Request]) -> list[RequestOutput]:
        outputs = []
        for req in updated:
            delta_text = self._detokenize(req)
            outputs.append(self._make_output(req, delta_text))
            if req.finished:
                self.requests.pop(req.request_id, None)
        return outputs

    def _propose_drafts(self) -> None:
        k = self.config.speculative.num_speculative_tokens
        for req in self.scheduler.running:
            if not req.is_decoding:
                continue
            cap = min(k, req.sampling_params.max_tokens - req.num_output_tokens - 1, self.max_model_len - req.num_tokens - 1)
            req.spec_token_ids = self.proposer.propose(req.all_token_ids, cap) if cap > 0 else []

    def _detokenize(self, req: Request) -> str:
        """Incremental: decode only the window [prefix_offset:] of the output tokens and append
        what it adds beyond [prefix_offset:read_offset] (vLLM's ``detokenize_incrementally``).
        Text ending in an incomplete UTF-8 sequence is held back until the request finishes."""
        if self.tokenizer is None:
            return ""
        sp = req.sampling_params
        # tokens past num_resolved_outputs are still in flight (overlap scheduling)
        delta, stopped = detokenize_step(self.tokenizer, req, req.output_token_ids, req.num_resolved_outputs,
                                         req.finished, sp.skip_special_tokens, sp.stop)
        if stopped and not req.finished:
            self.scheduler.finish_request(req, RequestStatus.FINISHED_STOPPED)
        return delta

    def _make_output(self, req: Request, delta_text: str) -> RequestOutput:
        metrics = {}
        if req.first_token_time is not None:
            metrics["ttft"] = req.first_token_time - req.arrival_time
        if req.finished and req.finish_time is not None:
            metrics["latency"] = req.finish_time - req.arrival_time
            metrics["cached_tokens"] = req.num_cached_tokens
            if req.num_draft_tokens:
                metrics["spec_acceptance"] = req.num_accepted_tokens / req.num_draft_tokens
        n = req.num_resolved_outputs
        return RequestOutput(
            request_id=req.request_id,
            prompt_token_ids=req.prompt_token_ids,
            # the full list only once finished: copying it for every streamed delta is O(length)
            output_token_ids=req.output_token_ids[:n] if req.finished else [],
            text=req.output_text,
            delta_text=delta_text,
            delta_token_ids=list(req.new_token_ids),
            finished=req.finished,
            finish_reason=req.finish_reason,
            metrics=metrics,
            num_output_tokens=n,
        )

    # --------------------------------------------------------------- offline API
    def generate(
        self,
        prompts: Sequence[PromptType],
        sampling_params: Union[SamplingParams, Sequence[SamplingParams], None] = None,
    ) -> list[RequestOutput]:
        if sampling_params is None or isinstance(sampling_params, SamplingParams):
            params = [sampling_params or SamplingParams()] * len(prompts)
        else:
            params = list(sampling_params)
        ids = [self.add_request(p, sp) for p, sp in zip(prompts, params)]
        final: dict[str, RequestOutput] = {}
        while self.has_unfinished_requests():
            for out in self.step():
                if out.finished:
                    final[out.request_id] = out
        return [final[i] for i in ids]

    def stats(self) -> dict:
        s = {
            "steps": self.num_steps,
            "free_kv_blocks": self.allocator.num_free,
            "total_kv_blocks": self.allocator.num_blocks,
            "running": len(self.scheduler.running),
            "waiting": len(self.scheduler.waiting),
            "preemptions": self.scheduler.num_preemptions,
        }
        if self.radix is not None:
            s["prefix_cache_hit_rate"] = round(self.radix.hit_rate, 4)
            s["cached_blocks"] = self.radix.num_cached_blocks
        s.update({f"moe_{k}": v for k, v in self.moe_executor.stats().items()})
        return s
