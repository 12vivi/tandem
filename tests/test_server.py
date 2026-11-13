from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from conftest import make_engine
from tandem.config import EngineConfig
from tandem.engine.async_engine import AsyncEngine
from tandem.engine.mp_engine import MPAsyncEngine
from tandem.entrypoints.openai_server import build_app


class ByteTokenizer:
    """Byte-level stand-in tokenizer for a 256-token vocabulary."""

    def encode(self, text, add_special_tokens=True):
        return list(text.encode("utf-8"))

    def decode(self, ids, skip_special_tokens=True):
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")


def _client(tiny_model, kind: str):
    tok = ByteTokenizer()
    path = tiny_model("qwen3").path
    if kind == "thread":  # engine thread inside the API process
        engine = AsyncEngine(make_engine(path, tokenizer=tok))
    else:  # engine core in a spawned process; detokenization and stop strings in the API process
        cfg = EngineConfig(model=path, device="cpu", dtype="float64", block_size=4, num_kv_blocks=256,
                           max_num_batched_tokens=64, max_model_len=256)
        engine = MPAsyncEngine(cfg, tok)
    return TestClient(build_app(engine, tok, "tiny-qwen3"))


KINDS = pytest.mark.parametrize("kind", ["thread", "process"])


def _events(resp):
    return [line[len("data: "):] for line in resp.iter_lines() if line.startswith("data: ")]


@KINDS
def test_completion_and_chat(tiny_model, kind):
    with _client(tiny_model, kind) as client:
        assert client.get("/health").json() == {"status": "ok"}
        req = {"prompt": "hello", "max_tokens": 5, "temperature": 0, "ignore_eos": True}
        r = client.post("/v1/completions", json=req)
        body = r.json()
        assert r.status_code == 200
        assert body["usage"] == {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10,
                                 "prompt_tokens_details": {"cached_tokens": 0}}
        assert body["choices"][0]["finish_reason"] == "length"
        # the same prompt again: its first whole block (4 tokens) comes from the prefix cache
        assert client.post("/v1/completions", json=req).json()["usage"]["prompt_tokens_details"] == {"cached_tokens": 4}

        r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, "ignore_eos": True})
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["role"] == "assistant"

        r = client.post("/v1/completions", json={"prompt": "x", "n": 2})
        assert r.status_code == 400


@KINDS
def test_streaming_with_usage(tiny_model, kind):
    with _client(tiny_model, kind) as client:
        payload = {"prompt": [1, 2, 3, 4], "max_tokens": 6, "temperature": 0, "ignore_eos": True,
                   "stream": True, "stream_options": {"include_usage": True}}
        with client.stream("POST", "/v1/completions", json=payload) as resp:
            events = _events(resp)
        assert events[-1] == "[DONE]"
        chunks = [json.loads(e) for e in events[:-1]]
        assert chunks[-1]["usage"]["completion_tokens"] == 6
        assert chunks[-2]["choices"][0]["finish_reason"] == "length"

        # non-stream output equals the concatenated stream (greedy, same prompt)
        full = client.post("/v1/completions", json={**payload, "stream": False}).json()["choices"][0]["text"]
        assert "".join(c["choices"][0]["text"] for c in chunks[:-1]) == full


@KINDS
def test_stop_string(tiny_model, kind):
    with _client(tiny_model, kind) as client:
        base = {"prompt": [5, 6, 7], "max_tokens": 16, "temperature": 0, "ignore_eos": True}
        full = client.post("/v1/completions", json=base).json()["choices"][0]["text"]
        stop = next((c for c in full[2:] if c.isascii() and c.isprintable()), None)
        if stop is None:
            pytest.skip("greedy output has no printable ASCII character to stop on")
        body = client.post("/v1/completions", json={**base, "stop": [stop]}).json()
        assert body["choices"][0]["text"] == full[: full.index(stop)]
        assert body["choices"][0]["finish_reason"] == "stop"
