"""Engine core in its own process (design doc §3.3, as in vLLM V1 / SGLang).

The API process (uvicorn) owns HTTP, tokenization, detokenization and stop strings; the
engine-core process owns the scheduler, the KV cache and the GPU, and never touches text. In
one process the event loop and the engine thread share the GIL: with thousands of streamed
chunks per second the engine may wait up to a switch interval (5 ms) to launch its next
step, longer than a whole decode step of a small model.

Two one-way pipes connect them:
  API -> core   ("add", rid, prompt_ids, SamplingParams) | ("abort", rid) | ("shutdown",)
  core -> API   ("ready",) once the model is loaded and graphs are captured,
                ("outputs", [(rid, new_token_ids, finished, finish_reason, num_output_tokens,
                              metrics), ...]) once per step, ("error", rid, message),
                ("dead", traceback)
``MPAsyncEngine`` has the interface of ``tandem.engine.async_engine.AsyncEngine`` (start /
shutdown / error / generate), so ``build_app`` takes either.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional, Sequence

from tandem.config import EngineConfig
from tandem.engine.detokenizer import detokenize_step
from tandem.engine.request import RequestOutput
from tandem.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


def _core_main(config: EngineConfig, inbox: Any, outbox: Any) -> None:
    """Entry point of the engine-core process."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s [core]: %(message)s")
    try:
        from tandem.engine.llm_engine import LLMEngine
        from tandem.executor.step_timer import TRACE

        engine = LLMEngine(config, load_tokenizer=False)
        outbox.send(("ready",))
        while True:
            busy = engine.has_unfinished_requests()
            if inbox.poll(0 if busy else 0.05):
                while inbox.poll(0):
                    msg = inbox.recv()
                    if msg[0] == "add":
                        _, rid, prompt_ids, params = msg
                        if TRACE:
                            TRACE("add", rid=rid, n=len(prompt_ids), max_tokens=params.max_tokens)
                        try:
                            engine.add_request(prompt_token_ids=prompt_ids, sampling_params=params, request_id=rid)
                        except Exception as exc:  # tell the caller instead of dying
                            outbox.send(("error", rid, str(exc)))
                    elif msg[0] == "abort":
                        engine.abort_request(msg[1])
                    elif msg[0] == "shutdown":
                        return
            if not engine.has_unfinished_requests():
                continue
            outs = engine.step()
            if outs:
                outbox.send(("outputs", [(o.request_id, o.delta_token_ids, o.finished, o.finish_reason,
                                          o.num_output_tokens, o.metrics) for o in outs]))
                if TRACE:
                    TRACE("sent", rids=[o.request_id for o in outs if o.num_output_tokens == 1])
    except BaseException:  # pragma: no cover - reported through /health
        outbox.send(("dead", traceback.format_exc()))
        raise


@dataclass
class _Stream:
    """API-side state of one request: tokens so far and incremental detokenization."""

    params: SamplingParams
    queue: asyncio.Queue
    prompt_ids: list[int]
    detok_prefix_offset: int = 0
    detok_read_offset: int = 0
    output_text: str = ""
    sent_text_len: int = 0
    tokens: list[int] = field(default_factory=list)
    done: bool = False


class MPAsyncEngine:
    def __init__(self, config: EngineConfig, tokenizer: Any, startup_timeout: float = 3600) -> None:
        ctx = mp.get_context("spawn")  # CUDA in the child; nothing CUDA here
        core_in, self._to_core = ctx.Pipe(duplex=False)
        self._from_core, core_out = ctx.Pipe(duplex=False)
        self.proc = ctx.Process(target=_core_main, args=(config, core_in, core_out), name="tandem-core", daemon=True)
        self.proc.start()
        core_in.close()
        core_out.close()
        self.tokenizer = tokenizer
        self.error: Optional[str] = None
        self._send_lock = threading.Lock()
        self._streams: dict[str, _Stream] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._reader: Optional[threading.Thread] = None
        if not self._from_core.poll(startup_timeout):
            raise RuntimeError("engine core did not start in time")
        msg = self._from_core.recv()
        if msg[0] != "ready":
            raise RuntimeError(f"engine core failed to start:\n{msg[1] if len(msg) > 1 else msg}")

    # --------------------------------------------------------------- lifecycle
    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._reader = threading.Thread(target=self._read_outputs, name="tandem-core-reader", daemon=True)
        self._reader.start()

    def shutdown(self) -> None:
        self._send(("shutdown",))
        self.proc.join(timeout=30)
        if self.proc.is_alive():
            self.proc.terminate()

    def _send(self, msg: tuple) -> None:
        with self._send_lock:
            try:
                self._to_core.send(msg)
            except (BrokenPipeError, OSError) as exc:
                self.error = self.error or f"engine core pipe closed: {exc!r}"

    # ----------------------------------------------------------------- outputs
    def _deliver(self, stream: _Stream, out: RequestOutput) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(stream.queue.put_nowait, out)

    def _read_outputs(self) -> None:
        while True:
            try:
                msg = self._from_core.recv()
            except (EOFError, OSError):
                self.error = self.error or "engine core exited"
                break
            kind = msg[0]
            if kind == "outputs":
                for rid, new_tokens, finished, reason, n_out, metrics in msg[1]:
                    stream = self._streams.get(rid)
                    if stream is not None and not stream.done:
                        self._on_tokens(rid, stream, new_tokens, finished, reason, n_out, metrics)
            elif kind == "error":
                stream = self._streams.get(msg[1])
                if stream is not None:
                    stream.done = True
                    self._deliver(stream, RequestOutput(msg[1], stream.prompt_ids, [], "", "", [], True, "error",
                                                        {"error": msg[2]}))
            elif kind == "dead":
                self.error = msg[1]
                logger.error("engine core crashed:\n%s", msg[1])
                for rid, stream in list(self._streams.items()):
                    stream.done = True
                    self._deliver(stream, RequestOutput(rid, stream.prompt_ids, [], "", "", [], True, "error",
                                                        {"error": "engine core crashed"}))
                break

    def _on_tokens(self, rid: str, stream: _Stream, new_tokens: list[int], finished: bool, reason: Optional[str],
                   n_out: int, metrics: dict) -> None:
        stream.tokens.extend(new_tokens)
        delta, stopped = "", False
        if self.tokenizer is not None:
            sp = stream.params
            delta, stopped = detokenize_step(self.tokenizer, stream, stream.tokens, len(stream.tokens),
                                             finished, sp.skip_special_tokens, sp.stop)
        if stopped and not finished:  # a stop string: end it here, drop the rest in the core
            finished, reason = True, "stop"
            self._send(("abort", rid))
        stream.done = finished
        self._deliver(stream, RequestOutput(
            request_id=rid, prompt_token_ids=stream.prompt_ids,
            output_token_ids=list(stream.tokens) if finished else [], text=stream.output_text,
            delta_text=delta, delta_token_ids=list(new_tokens), finished=finished, finish_reason=reason,
            metrics=metrics, num_output_tokens=len(stream.tokens) if stopped else n_out,
        ))

    # ----------------------------------------------------------------- requests
    async def generate(
        self, prompt_token_ids: Sequence[int], params: SamplingParams, request_id: Optional[str] = None
    ) -> AsyncIterator[RequestOutput]:
        if self.error is not None:
            raise RuntimeError(f"engine is dead: {self.error}")
        rid = request_id or uuid.uuid4().hex
        stream = _Stream(params, asyncio.Queue(), list(prompt_token_ids))
        self._streams[rid] = stream
        self._send(("add", rid, list(prompt_token_ids), params))
        finished = False
        try:
            while True:
                out = await stream.queue.get()
                if out.finish_reason == "error":
                    raise ValueError(out.metrics.get("error", "request failed"))
                finished = out.finished
                yield out
                if finished:
                    break
        finally:
            self._streams.pop(rid, None)
            if not finished:
                self._send(("abort", rid))
