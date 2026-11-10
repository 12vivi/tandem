"""OpenAI-compatible HTTP API: /v1/completions, /v1/chat/completions, /v1/models, /health.

Streaming uses SSE and honours ``stream_options.include_usage`` so that standard load
generators (``vllm bench serve``, AIPerf) can measure TTFT / TPOT against tandem.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional, Union

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from tandem.engine.async_engine import AsyncEngine
from tandem.sampling_params import SamplingParams


class _SamplingFields(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    n: int = 1
    seed: Optional[int] = None
    stop: Union[str, list[str], None] = None
    stop_token_ids: Optional[list[int]] = None
    ignore_eos: bool = False
    stream: bool = False
    stream_options: Optional[dict[str, Any]] = None

    def sampling_params(self, default_max_tokens: int) -> SamplingParams:
        return SamplingParams(
            max_tokens=self.max_tokens or default_max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            seed=self.seed,
            stop=self.stop,
            stop_token_ids=self.stop_token_ids or [],
            ignore_eos=self.ignore_eos,
        )

    @property
    def include_usage(self) -> bool:
        return bool(self.stream_options and self.stream_options.get("include_usage"))


class CompletionRequest(_SamplingFields):
    prompt: Union[str, list[int]]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: Union[str, list[dict[str, Any]], None] = None


class ChatCompletionRequest(_SamplingFields):
    messages: list[ChatMessage]
    max_completion_tokens: Optional[int] = None
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": "invalid_request_error"}})


def _sse(obj: Any) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _usage(prompt_tokens: int, completion_tokens: int) -> dict:
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}


def _message_text(content: Union[str, list[dict[str, Any]], None]) -> str:
    if content is None or isinstance(content, str):
        return content or ""
    return "".join(part.get("text", "") for part in content if part.get("type") == "text")


def build_app(async_engine: AsyncEngine, tokenizer: Any, model_name: str, default_max_tokens: int = 256) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        import asyncio

        async_engine.start(asyncio.get_running_loop())
        yield
        async_engine.shutdown()

    app = FastAPI(title="tandem", lifespan=lifespan)

    @app.get("/health")
    async def health():
        if async_engine.error is not None:
            return _error(500, f"engine error: {async_engine.error}")
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "tandem"}]}

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest):
        if body.n != 1:
            return _error(400, "only n=1 is supported")
        prompt_ids = list(body.prompt) if isinstance(body.prompt, list) else tokenizer.encode(body.prompt)
        try:
            params = body.sampling_params(default_max_tokens)
        except ValueError as exc:
            return _error(400, str(exc))
        return await _respond(body, prompt_ids, params, chat=False)

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest):
        if body.n != 1:
            return _error(400, "only n=1 is supported")
        messages = [{"role": m.role, "content": _message_text(m.content)} for m in body.messages]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **body.chat_template_kwargs)
        prompt_ids = tokenizer.encode(text, add_special_tokens=False)
        if body.max_completion_tokens and not body.max_tokens:
            body.max_tokens = body.max_completion_tokens
        try:
            params = body.sampling_params(default_max_tokens)
        except ValueError as exc:
            return _error(400, str(exc))
        return await _respond(body, prompt_ids, params, chat=True)

    async def _respond(body: _SamplingFields, prompt_ids: list[int], params: SamplingParams, chat: bool):
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex
        created = int(time.time())
        obj = "chat.completion" if chat else "text_completion"

        if not body.stream:
            text, n_out, reason = "", 0, None
            try:
                async for out in async_engine.generate(prompt_ids, params, rid):
                    text, n_out, reason = out.text, len(out.output_token_ids), out.finish_reason
            except ValueError as exc:
                return _error(400, str(exc))
            choice = (
                {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": reason}
                if chat
                else {"index": 0, "text": text, "logprobs": None, "finish_reason": reason}
            )
            return {"id": rid, "object": obj, "created": created, "model": model_name, "choices": [choice], "usage": _usage(len(prompt_ids), n_out)}

        chunk_obj = "chat.completion.chunk" if chat else "text_completion"

        async def stream():
            n_out = 0
            if chat:
                yield _sse({"id": rid, "object": chunk_obj, "created": created, "model": model_name,
                            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
            try:
                async for out in async_engine.generate(prompt_ids, params, rid):
                    n_out = len(out.output_token_ids)
                    if not out.delta_text and not out.finished:
                        continue
                    if chat:
                        choice = {"index": 0, "delta": {"content": out.delta_text} if out.delta_text else {}, "finish_reason": out.finish_reason}
                    else:
                        choice = {"index": 0, "text": out.delta_text, "logprobs": None, "finish_reason": out.finish_reason}
                    yield _sse({"id": rid, "object": chunk_obj, "created": created, "model": model_name, "choices": [choice]})
            except ValueError as exc:
                yield _sse({"error": {"message": str(exc)}})
            if body.include_usage:
                yield _sse({"id": rid, "object": chunk_obj, "created": created, "model": model_name, "choices": [], "usage": _usage(len(prompt_ids), n_out)})
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app
