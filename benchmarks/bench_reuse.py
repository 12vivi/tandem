"""Load generator for the KV-reuse scenarios B1 / B2 (design doc §1.3, §6.3).

Prompts are token-id lists sent to /v1/completions, so every engine gets exactly the same
tokens and the reuse an infinitely large cache could get ("ideal reuse") is known exactly.

  subset     build the multi-turn ShareGPT subset (run once, locally)
  multiturn  B1: sessions replay ShareGPT conversations with >= 4 turns. Turn k sends the
             whole history: turn k-1's prompt + the dataset's reply k-1 + user turn k, so
             the reusable prefix of turn k is turn k-1's prompt. Replies are the dataset's
             text, not the engine's output (same tokens for every engine); the KV of the
             generated tokens is therefore never reused. Think time between turns is
             exponential (--think-mean); sessions start uniformly over --ramp seconds.
  docs       B2: --num-docs random-token documents of --doc-min..--doc-max tokens; requests
             arrive as a Poisson process (--rate), pick a document by Zipf(--zipf) and add
             a unique short question. The reusable prefix is the document once it has been
             requested before.

Cached tokens per request come from usage.prompt_tokens_details.cached_tokens (vLLM:
--enable-prompt-tokens-details, SGLang: --enable-cache-report). One JSON line per run.

  python benchmarks/bench_reuse.py subset --sharegpt benchmarks/data/ShareGPT_V3_unfiltered_cleaned_split.json \\
      --tokenizer models/Qwen3-8B --out benchmarks/data/sharegpt_multiturn_qwen3.jsonl
  python benchmarks/bench_reuse.py multiturn --base-url http://127.0.0.1:8000 --model qwen3-8b \\
      --dataset-path benchmarks/data/sharegpt_multiturn_qwen3.jsonl --sessions 48 --save out.jsonl
  python benchmarks/bench_reuse.py docs --base-url http://127.0.0.1:8000 --model qwen3-8b --rate 0.2
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import json
import os
import random
import statistics
import time
from dataclasses import dataclass, field

import httpx

VOCAB = 151_643  # Qwen3 regular tokens are 0 .. 151642; specials come after
USER, ASSISTANT, END = "<|im_start|>user\n", "<|im_start|>assistant\n", "<|im_end|>\n"


@dataclass
class Rec:
    group: str          # "first" / "follow" (multiturn), "warmup" / "measure" (docs)
    prompt_len: int
    reusable: int       # tokens an infinite cache could reuse
    output_len: int
    ok: bool = False
    ttft: float = 0.0
    latency: float = 0.0
    cached: int = -1    # reported by the engine; -1 = not reported
    out_tokens: int = 0
    error: str = ""
    itl: list[float] = field(default_factory=list)


async def send(client: httpx.AsyncClient, url: str, model: str, prompt: list[int], rec: Rec) -> Rec:
    body = {"model": model, "prompt": prompt, "max_tokens": rec.output_len, "temperature": 0.0,
            "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    last, chunks = t0, 0
    try:
        async with client.stream("POST", url, json=body) as resp:
            if resp.status_code != 200:
                rec.error = f"HTTP {resp.status_code}: {(await resp.aread())[:200]!r}"
                return rec
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                usage = obj.get("usage")
                if usage:
                    rec.out_tokens = usage.get("completion_tokens", 0)
                    details = usage.get("prompt_tokens_details") or {}
                    if details.get("cached_tokens") is not None:
                        rec.cached = int(details["cached_tokens"])
                choices = obj.get("choices") or []
                if choices and choices[0].get("text"):
                    now = time.perf_counter()
                    if chunks == 0:
                        rec.ttft = now - t0
                    else:
                        rec.itl.append(now - last)
                    last, chunks = now, chunks + 1
        rec.latency = time.perf_counter() - t0
        rec.out_tokens = rec.out_tokens or chunks
        rec.ok = chunks > 0
    except Exception as exc:  # counted as failed
        rec.error = repr(exc)
    return rec


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, max(0, round(q / 100 * (len(s) - 1))))]


def ms(x: float | None) -> float | None:
    return None if x is None else round(x * 1000, 1)


def summarize(recs: list[Rec], groups: tuple[str, ...], duration: float, slo_ttft: float, slo_tpot: float) -> dict:
    sel = [r for r in recs if r.group in groups]
    ok = [r for r in sel if r.ok]
    tpot = [(r.latency - r.ttft) / (r.out_tokens - 1) for r in ok if r.out_tokens > 1]
    good = [r for r in ok if r.ttft <= slo_ttft and (r.out_tokens <= 1 or (r.latency - r.ttft) / (r.out_tokens - 1) <= slo_tpot)]
    prompt = sum(r.prompt_len for r in sel)
    # SGLang leaves prompt_tokens_details out when nothing was cached (vLLM sends 0): once
    # any request of the run reported a value, a missing one means 0
    if any(r.cached >= 0 for r in recs):
        for r in ok:
            r.cached = max(r.cached, 0)
    reported = [r for r in ok if r.cached >= 0]
    return {
        "requests": len(sel), "failed": len(sel) - len(ok),
        "ttft_ms": {"mean": ms(statistics.mean(r.ttft for r in ok)) if ok else None,
                    "p50": ms(pct([r.ttft for r in ok], 50)), "p90": ms(pct([r.ttft for r in ok], 90)),
                    "p99": ms(pct([r.ttft for r in ok], 99))},
        "tpot_ms": {"mean": ms(statistics.mean(tpot)) if tpot else None, "p50": ms(pct(tpot, 50)),
                    "p99": ms(pct(tpot, 99))},
        "slo_ok_share": round(len(good) / len(sel), 4) if sel else None,
        "goodput_rps": round(len(good) / duration, 4),
        "prompt_tokens": prompt,
        "ideal_reuse": round(sum(r.reusable for r in sel) / prompt, 4) if prompt else None,
        "hit_rate": round(sum(r.cached for r in reported) / sum(r.prompt_len for r in reported), 4) if reported else None,
        "hit_of_ideal": (round(sum(r.cached for r in reported) / sum(r.reusable for r in reported), 4)
                         if reported and sum(r.reusable for r in reported) else None),
    }


def build_subset(a: argparse.Namespace) -> None:
    from bench_serving import load_tokenizer
    tok = load_tokenizer(a.tokenizer)
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    with open(a.sharegpt, encoding="utf-8") as f:
        data = json.load(f)
    rng = random.Random(a.seed)
    rng.shuffle(data)
    out = []
    for d in data:
        conv = d.get("conversations") or []
        pairs = []
        for i in range(0, len(conv) - 1, 2):
            if conv[i].get("from") != "human" or conv[i + 1].get("from") != "gpt":
                break
            pairs.append((conv[i]["value"], conv[i + 1]["value"]))
        if len(pairs) < a.min_turns:
            continue
        turns, total = [], 0
        for user, reply in pairs[:a.max_turns]:
            u = enc(USER + user + END + ASSISTANT)
            r = enc(reply)
            if len(r) < 4:
                break
            if total + len(u) + min(len(r), a.max_output) > a.max_context:
                break
            turns.append({"user": u, "assistant": r + enc(END), "out": min(len(r), a.max_output)})
            total += len(u) + len(r) + 2
        if len(turns) >= a.min_turns:
            out.append({"turns": turns})
            if len(out) == a.num:
                break
    with open(a.out, "w", encoding="utf-8") as f:
        for c in out:
            f.write(json.dumps(c) + "\n")
    n_turns = [len(c["turns"]) for c in out]
    final = [sum(len(t["user"]) + len(t["assistant"]) for t in c["turns"]) for c in out]
    print(f"{len(out)} conversations -> {a.out}; turns mean {statistics.mean(n_turns):.1f}, "
          f"final history mean {statistics.mean(final):.0f} tokens (max {max(final)})")


async def run_multiturn(a: argparse.Namespace) -> tuple[list[Rec], float, dict]:
    with open(os.path.expanduser(a.dataset_path), encoding="utf-8") as f:
        convs = [json.loads(line) for line in f if line.strip()]
    convs = convs[a.offset:a.offset + a.sessions]
    if len(convs) < a.sessions:
        raise SystemExit(f"only {len(convs)} conversations after offset {a.offset}")
    for c in convs:  # optional caps (smoke tests)
        if a.max_turns:
            c["turns"] = c["turns"][:a.max_turns]
        if a.max_output:
            for t in c["turns"]:
                t["out"] = min(t["out"], a.max_output)
    rng = random.Random(a.seed)
    starts = sorted(rng.uniform(0, a.ramp) for _ in convs)
    thinks = [[rng.expovariate(1 / a.think_mean) for _ in c["turns"]] for c in convs]
    recs: list[Rec] = []
    ctx = [0] * len(convs)  # context of each live session (its KV is what the next turn reuses)
    peak = [0]

    async def session(i: int, client: httpx.AsyncClient) -> None:
        await asyncio.sleep(starts[i])
        history: list[int] = []
        prev = 0
        turns = convs[i]["turns"]
        for k, turn in enumerate(turns):
            prompt = history + turn["user"]
            rec = Rec("first" if k == 0 else "follow", len(prompt), prev, turn["out"])
            recs.append(await send(client, a.url, a.model, prompt, rec))
            ctx[i] = len(prompt) + turn["out"] if k + 1 < len(turns) else 0
            peak[0] = max(peak[0], sum(ctx))
            history, prev = prompt + turn["assistant"], len(prompt)
            if k + 1 < len(turns):
                await asyncio.sleep(thinks[i][k])

    limits = httpx.Limits(max_connections=a.sessions + 8, max_keepalive_connections=a.sessions + 8)
    async with httpx.AsyncClient(timeout=httpx.Timeout(a.timeout), limits=limits) as client:
        t0 = time.perf_counter()
        await asyncio.gather(*(session(i, client) for i in range(len(convs))))
        duration = time.perf_counter() - t0
    params = {"sessions": a.sessions, "offset": a.offset, "think_mean_s": a.think_mean, "ramp_s": a.ramp,
              "turns": sum(len(c["turns"]) for c in convs), "working_set_peak_tokens": peak[0]}
    return recs, duration, params


async def run_docs(a: argparse.Namespace) -> tuple[list[Rec], float, dict]:
    rng = random.Random(a.seed)
    docs = [[rng.randrange(VOCAB) for _ in range(rng.randint(a.doc_min, a.doc_max))] for _ in range(a.num_docs)]
    weights = [1 / (i + 1) ** a.zipf for i in range(a.num_docs)]
    cum = [sum(weights[:i + 1]) for i in range(a.num_docs)]
    total = a.warmup + a.requests
    picks = [bisect.bisect_left(cum, rng.random() * cum[-1]) for _ in range(total)]
    gaps = [rng.expovariate(a.rate) for _ in range(total)]
    seen: set[int] = set()
    plan = []
    for j, d in enumerate(picks):
        question = [rng.randrange(VOCAB) for _ in range(a.question_len)]
        prompt = docs[d] + question
        plan.append((prompt, Rec("warmup" if j < a.warmup else "measure", len(prompt),
                                 len(docs[d]) if d in seen else 0, a.output_len)))
        seen.add(d)
    recs = [r for _, r in plan]
    limits = httpx.Limits(max_connections=total + 8, max_keepalive_connections=64)
    async with httpx.AsyncClient(timeout=httpx.Timeout(a.timeout), limits=limits) as client:
        tasks = []
        t0 = time.perf_counter()
        t_measure = t0
        for j, (prompt, rec) in enumerate(plan):
            await asyncio.sleep(gaps[j])
            if j == a.warmup:
                t_measure = time.perf_counter()
            tasks.append(asyncio.create_task(send(client, a.url, a.model, prompt, rec)))
        await asyncio.gather(*tasks)
        duration = time.perf_counter() - t_measure
    params = {"num_docs": a.num_docs, "doc_tokens_mean": round(statistics.mean(len(d) for d in docs)),
              "doc_tokens_total": sum(len(d) for d in docs), "zipf": a.zipf, "rate_rps": a.rate,
              "warmup": a.warmup, "question_len": a.question_len}
    return recs, duration, params


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("subset")
    s.add_argument("--sharegpt", required=True)
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--num", type=int, default=512)
    s.add_argument("--min-turns", type=int, default=4)
    s.add_argument("--max-turns", type=int, default=8)
    s.add_argument("--max-output", type=int, default=512)
    s.add_argument("--max-context", type=int, default=8192)
    s.add_argument("--seed", type=int, default=0)
    for name in ("multiturn", "docs"):
        p = sub.add_parser(name)
        p.add_argument("--base-url", default="http://127.0.0.1:8000")
        p.add_argument("--model", required=True)
        p.add_argument("--tag", default="")
        p.add_argument("--meta", default="", help="JSON merged into the result line")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--timeout", type=float, default=3600)
        p.add_argument("--slo-ttft", type=float, default=2.0, help="seconds")
        p.add_argument("--slo-tpot", type=float, default=0.1, help="seconds")
        p.add_argument("--save", default="")
        if name == "multiturn":
            p.add_argument("--dataset-path", required=True)
            p.add_argument("--sessions", type=int, required=True)
            p.add_argument("--offset", type=int, default=0, help="first conversation of the subset to use")
            p.add_argument("--think-mean", type=float, default=30.0)
            p.add_argument("--ramp", type=float, default=30.0)
            p.add_argument("--max-turns", type=int, default=0, help="cap turns per session (0 = all)")
            p.add_argument("--max-output", type=int, default=0, help="cap output tokens per turn (0 = dataset)")
        else:
            p.add_argument("--num-docs", type=int, default=32)
            p.add_argument("--doc-min", type=int, default=8192)
            p.add_argument("--doc-max", type=int, default=24576)
            p.add_argument("--zipf", type=float, default=1.0)
            p.add_argument("--rate", type=float, required=True, help="requests per second (Poisson)")
            p.add_argument("--requests", type=int, default=80)
            p.add_argument("--warmup", type=int, default=16)
            p.add_argument("--question-len", type=int, default=48)
            p.add_argument("--output-len", type=int, default=64)
    a = ap.parse_args()
    if a.cmd == "subset":
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        build_subset(a)
        return

    a.url = a.base_url.rstrip("/") + "/v1/completions"
    recs, duration, params = asyncio.run(run_multiturn(a) if a.cmd == "multiturn" else run_docs(a))
    groups = {"multiturn": [("all", ("first", "follow")), ("follow", ("follow",))],
              "docs": [("all", ("measure",))]}[a.cmd]
    row = {"tag": a.tag, "model": a.model, "workload": a.cmd, **params, "duration_s": round(duration, 1),
           "output_tok_per_s": round(sum(r.out_tokens for r in recs if r.ok and r.group != "warmup") / duration, 1),
           **{g: summarize(recs, sel, duration, a.slo_ttft, a.slo_tpot) for g, sel in groups},
           "slo": {"ttft_s": a.slo_ttft, "tpot_s": a.slo_tpot},
           "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **(json.loads(a.meta) if a.meta else {})}
    errors = sorted({r.error for r in recs if r.error})[:3]
    if errors:
        row["errors"] = errors
    print(json.dumps(row, ensure_ascii=False), flush=True)
    if a.save:
        with open(a.save, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
