"""Load generator for any OpenAI-compatible server (tandem, vLLM, SGLang, llama.cpp ...).

Every engine is measured with this same client (design doc §6). Prompts are random text
built from the model's tokenizer with a fixed seed, so every engine sees identical inputs.
Streaming is used to measure TTFT / TPOT / ITL; token counts come from the server's
`usage` (stream_options.include_usage) when available.

Example:
    python benchmarks/bench_serving.py --base-url http://127.0.0.1:8000 --model qwen3-0.6b \
        --tokenizer ~/models/Qwen3-0.6B --input-len 512 --output-len 128 \
        --num-prompts 64 --concurrency 1,4,16 --save benchmarks/results/tandem.jsonl --tag tandem

Limitation: a single asyncio process; at very high request rates such clients inflate
TTFT/TPOT (arXiv 2605.24217). Fine for the concurrency levels used on one GPU.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx


@dataclass
class Result:
    ok: bool = False
    ttft: float = 0.0
    latency: float = 0.0
    itl: list[float] = field(default_factory=list)
    prompt_tokens: int = 0
    output_tokens: int = 0
    error: str = ""


def build_prompts(tokenizer_path: str, n: int, input_len: int, seed: int) -> list[str]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    rng = random.Random(seed)
    special = set(tok.all_special_ids)
    vocab = [i for i in range(min(tok.vocab_size, 100_000)) if i not in special]
    prompts = []
    for _ in range(n):
        ids = [rng.choice(vocab) for _ in range(input_len)]
        text = tok.decode(ids)
        ids = tok.encode(text, add_special_tokens=False)[:input_len]  # re-tokenize, trim to length
        prompts.append(tok.decode(ids))
    return prompts


async def one_request(client: httpx.AsyncClient, url: str, model: str, prompt: str, output_len: int, extra: dict) -> Result:
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_len,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        **extra,
    }
    res = Result()
    t0 = time.perf_counter()
    last = t0
    chunks = 0
    try:
        async with client.stream("POST", url, json=body) as resp:
            if resp.status_code != 200:
                res.error = f"HTTP {resp.status_code}: {(await resp.aread())[:200]!r}"
                return res
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if obj.get("usage"):
                    res.prompt_tokens = obj["usage"].get("prompt_tokens", 0)
                    res.output_tokens = obj["usage"].get("completion_tokens", 0)
                choices = obj.get("choices") or []
                if choices and (choices[0].get("text") or (choices[0].get("delta") or {}).get("content")):
                    now = time.perf_counter()
                    if chunks == 0:
                        res.ttft = now - t0
                    else:
                        res.itl.append(now - last)
                    last = now
                    chunks += 1
        res.latency = time.perf_counter() - t0
        res.output_tokens = res.output_tokens or chunks
        res.ok = chunks > 0
    except Exception as exc:  # network errors count as failed requests
        res.error = repr(exc)
    return res


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q / 100 * (len(s) - 1))))]


async def run_level(args, prompts: list[str], concurrency: int) -> dict:
    url = args.base_url.rstrip("/") + "/v1/completions"
    sem = asyncio.Semaphore(concurrency)
    extra = json.loads(args.extra_body) if args.extra_body else {}
    limits = httpx.Limits(max_connections=concurrency + 4, max_keepalive_connections=concurrency + 4)

    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout), limits=limits, trust_env=False) as client:
        # warm-up (not measured)
        await one_request(client, url, args.model, prompts[0], min(8, args.output_len), extra)

        async def worker(p: str) -> Result:
            async with sem:
                return await one_request(client, url, args.model, p, args.output_len, extra)

        t0 = time.perf_counter()
        tasks = []
        for p in prompts:
            tasks.append(asyncio.create_task(worker(p)))
            if args.request_rate != float("inf"):
                await asyncio.sleep(random.expovariate(args.request_rate))
        results = await asyncio.gather(*tasks)
        duration = time.perf_counter() - t0

    ok = [r for r in results if r.ok]
    out_tokens = sum(r.output_tokens for r in ok)
    tpot = [(r.latency - r.ttft) / (r.output_tokens - 1) for r in ok if r.output_tokens > 1]
    itl = [x for r in ok for x in r.itl]
    summary = {
        "tag": args.tag,
        "model": args.model,
        "concurrency": concurrency,
        "request_rate": args.request_rate,
        "num_prompts": len(prompts),
        "completed": len(ok),
        "failed": len(results) - len(ok),
        "input_len": args.input_len,
        "output_len": args.output_len,
        "duration_s": round(duration, 2),
        "request_throughput": round(len(ok) / duration, 3),
        "output_tok_per_s": round(out_tokens / duration, 2),
        "mean_prompt_tokens": round(statistics.mean(r.prompt_tokens for r in ok), 1) if ok else 0,
        "ttft_ms": {"mean": round(statistics.mean(r.ttft for r in ok) * 1e3, 1) if ok else None,
                    "p50": round(pct([r.ttft for r in ok], 50) * 1e3, 1), "p99": round(pct([r.ttft for r in ok], 99) * 1e3, 1)},
        "tpot_ms": {"mean": round(statistics.mean(tpot) * 1e3, 2) if tpot else None,
                    "p50": round(pct(tpot, 50) * 1e3, 2), "p99": round(pct(tpot, 99) * 1e3, 2)},
        "itl_ms": {"p50": round(pct(itl, 50) * 1e3, 2), "p99": round(pct(itl, 99) * 1e3, 2)},
    }
    if results and not ok:
        summary["first_error"] = results[0].error
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True, help="model name the server expects")
    ap.add_argument("--tokenizer", required=True, help="local tokenizer directory used to build prompts")
    ap.add_argument("--input-len", type=int, default=512)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--num-prompts", type=int, default=64)
    ap.add_argument("--concurrency", default="1,4,16", help="comma-separated sweep")
    ap.add_argument("--request-rate", type=float, default=float("inf"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--extra-body", default="", help="JSON merged into every request body")
    ap.add_argument("--tag", default="", help="label stored with the results, e.g. the engine name")
    ap.add_argument("--save", default="", help="append JSON lines here")
    args = ap.parse_args()

    prompts = build_prompts(os.path.expanduser(args.tokenizer), args.num_prompts, args.input_len, args.seed)
    for c in [int(x) for x in args.concurrency.split(",")]:
        n = min(len(prompts), max(c * 4, 8)) if c < len(prompts) // 4 else len(prompts)
        summary = asyncio.run(run_level(args, prompts[:n], c))
        print(json.dumps(summary, ensure_ascii=False))
        if args.save:
            os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
            with open(args.save, "a", encoding="utf-8") as f:
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
