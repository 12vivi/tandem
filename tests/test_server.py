from __future__ import annotations

import json

from fastapi.testclient import TestClient

from conftest import make_engine
from tandem.engine.async_engine import AsyncEngine
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


def _client(tiny_model):
    tok = ByteTokenizer()
    eng = make_engine(tiny_model("qwen3").path, tokenizer=tok)
    return TestClient(build_app(AsyncEngine(eng), tok, "tiny-qwen3"))


def _events(resp):
    return [line[len("data: "):] for line in resp.iter_lines() if line.startswith("data: ")]


def test_completion_and_chat(tiny_model):
    with _client(tiny_model) as client:
        assert client.get("/health").json() == {"status": "ok"}
        r = client.post("/v1/completions", json={"prompt": "hello", "max_tokens": 5, "temperature": 0, "ignore_eos": True})
        body = r.json()
        assert r.status_code == 200
        assert body["usage"] == {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
        assert body["choices"][0]["finish_reason"] == "length"

        r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, "ignore_eos": True})
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["role"] == "assistant"

        r = client.post("/v1/completions", json={"prompt": "x", "n": 2})
        assert r.status_code == 400


def test_streaming_with_usage(tiny_model):
    with _client(tiny_model) as client:
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
