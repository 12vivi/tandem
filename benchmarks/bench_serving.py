"""Load generator for any OpenAI-compatible server (tandem, vLLM, SGLang, llama.cpp ...).

Every engine is measured with this same client (design doc §6). Inputs are identical for
every engine: either random text built from the model's tokenizer with a fixed seed, or a
fixed-seed ShareGPT sample. Streaming is used to measure TTFT / TPOT / ITL; token counts
come from the server's `usage` (stream_options.include_usage) when available.

Examples:
    # fixed lengths, concurrency sweep, 3 runs per level
    python benchmarks/bench_serving.py --base-url http://127.0.0.1:8000 --model qwen3-8b \
        --tokenizer ~/models/Qwen3-8B --input-len 1024 --output-len 256 \
        --concurrency 1,4,16 --repeat 3 --save benchmarks/results/p0.jsonl --tag vllm-0.32

    # ShareGPT; "inf" sends every request at once (offline-throughput style)
    python benchmarks/bench_serving.py ... --dataset sharegpt \
        --dataset-path benchmarks/data/sharegpt_qwen3.jsonl --num-prompts 1000 --concurrency inf

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

import httpx


@dataclass
class Req:
    prompt: str
    output_len: int
    prompt_len: int = 0


@dataclass
class Result:
    ok: bool = False
    ttft: float = 0.0
    latency: float = 0.0
    itl: list[float] = field(default_factory=list)
    prompt_tokens: int = 0
    output_tokens: int = 0
    error: str = ""


def load_tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(os.path.expanduser(path))


def build_random(tok, n: int, input_len: int, output_len: int, seed: int) -> list[Req]:
    rng = random.Random(seed)
    special = set(tok.all_special_ids)
    vocab = [i for i in range(min(tok.vocab_size, 100_000)) if i not in special]
    reqs = []
    for _ in range(n):
        ids = [rng.choice(vocab) for _ in range(input_len)]
        text = tok.decode(ids)
        ids = tok.encode(text, add_special_tokens=False)[:input_len]  # re-tokenize, trim to length
        reqs.append(Req(tok.decode(ids), output_len, len(ids)))
    return reqs


def sample_sharegpt(path: str, tok, n: int, seed: int, max_prompt: int = 1024, max_total: int = 2048) -> list[Req]:
    """ShareGPT requests with the same rules as `vllm bench serve --dataset-name sharegpt`.

    Prompt = first human turn, output length = token length of the first reply (sent with
    ignore_eos, so every engine generates exactly that many tokens). Pairs with fewer than
    4 prompt or output tokens, prompts over 1024 tokens, or totals over 2048 are skipped.
    `path` is either the original ShareGPT JSON or a JSONL subset written by
    make_sharegpt_subset.py (fields: prompt, prompt_len, output_len).
    """
    path = os.path.expanduser(path)
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if len(rows) < n:
            raise SystemExit(f"{path} has {len(rows)} requests, {n} requested")
        return [Req(r["prompt"], int(r["output_len"]), int(r.get("prompt_len", 0))) for r in rows[:n]]

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    convs = [d["conversations"] for d in data if len(d.get("conversations", [])) >= 2]
    random.Random(seed).shuffle(convs)
    reqs: list[Req] = []
    for conv in convs:
        prompt, completion = conv[0]["value"], conv[1]["value"]
        p_len = len(tok.encode(prompt, add_special_tokens=False))
        o_len = len(tok.encode(completion, add_special_tokens=False))
        if p_len < 4 or o_len < 4 or p_len > max_prompt or p_len + o_len > max_total:
            continue
        reqs.append(Req(prompt, o_len, p_len))
        if len(reqs) == n:
            break
    if len(reqs) < n:
        raise SystemExit(f"only {len(reqs)} ShareGPT requests pass the filters, {n} requested")
    return reqs


async def one_request(client: httpx.AsyncClient, url: str, model: str, req: Req, extra: dict) -> Result:
    body = {
        "model": model,
        "prompt": req.prompt,
        "max_tokens": req.output_len,
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


def ms(x: float) -> float | None:
    return None if x != x else round(x * 1e3, 2)  # NaN -> None


async def run_level(args, reqs: list[Req], concurrency: int, run_idx: int, meta: dict) -> dict:
    url = args.base_url.rstrip("/") + "/v1/completions"
    sem = asyncio.Semaphore(concurrency)
    extra = json.loads(args.extra_body) if args.extra_body else {}
    limits = httpx.Limits(max_connections=concurrency + 4, max_keepalive_connections=concurrency + 4)

    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout), limits=limits, trust_env=False) as client:
        # warm-up at this concurrency with short outputs (not measured): triggers lazy
        # compilation / graph capture for this batch size before timing starts
        n_warm = min(concurrency, args.warmup_max)
        await asyncio.gather(*(one_request(client, url, args.model, Req(r.prompt, 8), extra) for r in reqs[:n_warm]))

        async def worker(r: Req) -> Result:
            async with sem:
                return await one_request(client, url, args.model, r, extra)

        t0 = time.perf_counter()
        tasks = []
        for r in reqs:
            tasks.append(asyncio.create_task(worker(r)))
            if args.request_rate != float("inf"):
                await asyncio.sleep(random.expovariate(args.request_rate))
        results = await asyncio.gather(*tasks)
        duration = time.perf_counter() - t0

    ok = [r for r in results if r.ok]
    out_tokens = sum(r.output_tokens for r in ok)
    tpot = [(r.latency - r.ttft) / (r.output_tokens - 1) for r in ok if r.output_tokens > 1]
    itl = [x for r in ok for x in r.itl]
    ttft = [r.ttft for r in ok]
    summary = {
        "tag": args.tag,
        "model": args.model,
        "dataset": args.dataset,
        "concurrency": concurrency,
        "request_rate": args.request_rate,
        "run": run_idx,
        "num_prompts": len(reqs),
        "completed": len(ok),
        "failed": len(results) - len(ok),
        "input_len": args.input_len if args.dataset == "random" else round(statistics.mean(r.prompt_len for r in reqs), 1),
        "output_len": args.output_len if args.dataset == "random" else round(statistics.mean(r.output_len for r in reqs), 1),
        "duration_s": round(duration, 2),
        "request_throughput": round(len(ok) / duration, 3),
        "output_tok_per_s": round(out_tokens / duration, 2),
        "mean_prompt_tokens": round(statistics.mean(r.prompt_tokens for r in ok), 1) if ok else 0,
        "ttft_ms": {"mean": ms(statistics.mean(ttft)) if ttft else None, "p50": ms(pct(ttft, 50)), "p99": ms(pct(ttft, 99))},
        "tpot_ms": {"mean": ms(statistics.mean(tpot)) if tpot else None, "p50": ms(pct(tpot, 50)), "p99": ms(pct(tpot, 99))},
        "itl_ms": {"p50": ms(pct(itl, 50)), "p99": ms(pct(itl, 99))},
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **meta,
    }
    if results and not ok:
        summary["first_error"] = results[0].error
    return summary


def parse_levels(spec: str) -> list[float]:
    return [float("inf") if x.strip() in ("inf", "all") else int(x) for x in spec.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True, help="model name the server expects")
    ap.add_argument("--tokenizer", required=True, help="local tokenizer directory used to build prompts")
    ap.add_argument("--dataset", choices=["random", "sharegpt"], default="random")
    ap.add_argument("--dataset-path", default="", help="ShareGPT JSON or JSONL subset (--dataset sharegpt)")
    ap.add_argument("--input-len", type=int, default=512, help="random dataset only")
    ap.add_argument("--output-len", type=int, default=128, help="random dataset only")
    ap.add_argument("--num-prompts", type=int, default=0,
                    help="size of the request pool (default: enough for every level and run)")
    ap.add_argument("--concurrency", default="1,4,16", help='comma-separated sweep; "inf" = all requests at once')
    ap.add_argument("--requests-per-concurrency", type=int, default=4, help="requests per level = max(this x c, --min-requests)")
    ap.add_argument("--min-requests", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=1, help="runs per level (each run uses fresh prompts)")
    ap.add_argument("--warmup-max", type=int, default=4, help="warm-up requests per level = min(c, this)")
    ap.add_argument("--request-rate", type=float, default=float("inf"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--extra-body", default="", help="JSON merged into every request body")
    ap.add_argument("--meta", default="", help="JSON merged into every result line (engine version, KV capacity ...)")
    ap.add_argument("--tag", default="", help="label stored with the results, e.g. the engine name")
    ap.add_argument("--save", default="", help="append JSON lines here")
    args = ap.parse_args()

    levels = parse_levels(args.concurrency)
    per_level = {c: (None if c == float("inf") else max(args.requests_per_concurrency * int(c), args.min_requests)) for c in levels}
    finite = [n for n in per_level.values() if n]
    pool = args.num_prompts or max([n * args.repeat for n in finite] or [args.min_requests])

    tok = load_tokenizer(args.tokenizer)
    if args.dataset == "random":
        reqs = build_random(tok, pool, args.input_len, args.output_len, args.seed)
    else:
        if not args.dataset_path:
            raise SystemExit("--dataset sharegpt needs --dataset-path")
        reqs = sample_sharegpt(args.dataset_path, tok, pool, args.seed)
    meta = json.loads(args.meta) if args.meta else {}

    for c in levels:
        for run in range(args.repeat):
            if c == float("inf"):
                batch, conc = reqs, len(reqs)
            else:
                n = min(per_level[c], len(reqs))
                start = (run * n) % len(reqs)
                batch = (reqs[start:] + reqs[:start])[:n]  # a fresh slice per run
                conc = int(c)
            summary = asyncio.run(run_level(args, batch, conc, run, meta))
            if c == float("inf"):
                summary["concurrency"] = "inf"
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            if args.save:
                os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
                with open(args.save, "a", encoding="utf-8") as f:
                    f.write(json.dumps(summary, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
