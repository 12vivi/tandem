"""Asyncio front-end over ``LLMEngine``.

The engine runs in a dedicated thread; new requests and aborts arrive through a
thread-safe inbox, and outputs go back to per-request asyncio queues. The design doc's
target is a separate engine-core process with ZMQ (vLLM V1 / SGLang); this thread
version keeps the same boundary so it can be split into a process later.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import uuid
from typing import AsyncIterator, Optional, Sequence

from tandem.engine.llm_engine import LLMEngine
from tandem.engine.request import RequestOutput
from tandem.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


class AsyncEngine:
    def __init__(self, engine: LLMEngine) -> None:
        self.engine = engine
        self._inbox: queue.Queue = queue.Queue()
        self._queues: dict[str, asyncio.Queue] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[BaseException] = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._thread = threading.Thread(target=self._run, name="tandem-engine", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _deliver(self, out: RequestOutput) -> None:
        q = self._queues.get(out.request_id)
        if q is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(q.put_nowait, out)

    def _drain_inbox(self) -> None:
        while True:
            try:
                kind, payload = self._inbox.get_nowait()
            except queue.Empty:
                return
            if kind == "add":
                rid, token_ids, params = payload
                try:
                    self.engine.add_request(prompt_token_ids=token_ids, sampling_params=params, request_id=rid)
                except Exception as exc:  # report to the waiting caller instead of killing the loop
                    self._deliver(RequestOutput(rid, list(token_ids), [], "", "", [], True, "error", {"error": str(exc)}))
            elif kind == "abort":
                self.engine.abort_request(payload)

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._drain_inbox()
                if not self.engine.has_unfinished_requests():
                    self._wake.wait(timeout=0.05)
                    self._wake.clear()
                    continue
                for out in self.engine.step():
                    self._deliver(out)
        except BaseException as exc:  # pragma: no cover - surfaced through /health
            logger.exception("engine loop crashed")
            self.error = exc

    async def generate(
        self, prompt_token_ids: Sequence[int], params: SamplingParams, request_id: Optional[str] = None
    ) -> AsyncIterator[RequestOutput]:
        if self.error is not None:
            raise RuntimeError(f"engine is dead: {self.error}")
        rid = request_id or uuid.uuid4().hex
        q: asyncio.Queue = asyncio.Queue()
        self._queues[rid] = q
        self._inbox.put(("add", (rid, list(prompt_token_ids), params)))
        self._wake.set()
        finished = False
        try:
            while True:
                out = await q.get()
                if out.finish_reason == "error":
                    raise ValueError(out.metrics.get("error", "request failed"))
                finished = out.finished
                yield out
                if finished:
                    break
        finally:
            self._queues.pop(rid, None)
            if not finished:
                self._inbox.put(("abort", rid))
                self._wake.set()
