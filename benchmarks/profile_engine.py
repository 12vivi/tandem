"""Per-operator GPU profile of one engine (design doc §8 P0).

Starts the engine's OpenAI server under Nsight Systems (`nsys launch`) with the same flags as
run_matrix.py and captures each phase as its own report with `nsys start` / `nsys stop`
from outside (the engines' own /start_profile -> cudaProfilerStart did not trigger an nsys
capture range in vLLM 0.32's EngineCore, while external start / stop works for both):
  decode-b   b requests in steady decode (input --input-len), about --window-steps steps
  prefill    one request with a --prefill-len prompt and max_tokens 1
Kernels are grouped into operator classes and divided by the number of decode steps in the
capture (CUDA graph launches; one per step for both engines), giving per-step GPU time per
class, the GPU idle share of a step (time not covered by any kernel) and the top kernels.

  python benchmarks/profile_engine.py --engine vllm --python ~/venvs/vllm-0.32/bin/python \\
      --model ~/models/Qwen3-8B --served-name qwen3-8b --env VLLM_USE_FLASHINFER_SAMPLER=0 \\
      --save benchmarks/results/p0-profile.jsonl
  python benchmarks/profile_engine.py --reanalyze benchmarks/results/p0-profile.jsonl
Needs nsys on PATH (the CUDA 12.4 toolkit ships nsys 2023.4 in /usr/local/cuda-12.4/bin).
Class names are approximate for fused kernels (e.g. torch.compile's triton_* kernels):
read them together with the top-kernel list.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from collections import Counter

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_matrix  # noqa: E402  server command line, readiness, shutdown
from bench_serving import build_random, load_tokenizer  # noqa: E402

CLASSES = [  # (class, regex on the lower-cased kernel name); first match wins
    ("memcpy/memset", r"^\[cuda mem"),
    ("kv write", r"reshape_and_cache|store_kv|set_kv|write_kv|copy_blocks|kv_buffer"),
    ("norm", r"rmsnorm|rms_norm|layernorm|norm_kernel|rsqrt"),  # before gemm: FlashInfer's CuTe norms say "cutlass"
    ("rope", r"rotary|rope|pos_encoding|cos_sin"),
    ("activation", r"silu|act_and_mul|swiglu|gelu"),
    ("attention", r"flash_fwd|fmha|attention|batchdecode|batchprefill|withpagedkv|withkvcache|merge_?states?|paged_attn"),
    ("dequant", r"dequant"),  # tandem's AWQ prefill path (dequantize, then cuBLAS)
    ("gemm", r"gemm|gemv|cutlass|cublas|xmma|s16816|splitk|marlin|matmul|w4_mma"),
    ("sampling", r"argmax|sampl|softmax|topk|top_k|top_p|sort|gumbel|penalt"),
]


def classify(name: str) -> str:
    low = name.lower()
    for cls, pat in CLASSES:
        if re.search(pat, low):
            return cls
    return "other"


def nsys(*args: str, timeout: float = 1800) -> str:
    r = subprocess.run(["nsys", *args], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise SystemExit(f"nsys {' '.join(args)} failed: {(r.stdout + r.stderr)[-500:]}")
    return r.stdout


class Stream(threading.Thread):
    """One streaming completion; counts text chunks (one token per chunk at stream interval 1)."""

    def __init__(self, url: str, body: dict):
        super().__init__(daemon=True)
        self.url, self.body, self.chunks, self.error = url, body, 0, ""

    def run(self) -> None:
        try:
            with httpx.Client(timeout=None) as c, c.stream("POST", self.url, json=self.body) as r:
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    ch = json.loads(data).get("choices") or []
                    if ch and ch[0].get("text"):
                        self.chunks += 1
        except Exception as exc:  # reported by the caller
            self.error = repr(exc)


def nsys_csv(rep: str, report: str, first_col: str) -> list[dict]:
    # --force-export: nsys skips the report (no output) when an older SQLite export exists
    out = subprocess.run(["nsys", "stats", "-q", "--force-export=true", "--report", report, "--format", "csv",
                          "--output", "-", rep], capture_output=True, text=True, timeout=1800).stdout
    lines = out.splitlines()
    for i, line in enumerate(lines):  # skip any notice before the (unquoted) CSV header
        if line.lstrip('"').startswith(first_col):
            return list(csv.DictReader(io.StringIO("\n".join(lines[i:]))))
    return []


def analyze(rep: str, client_steps: int, phase: str) -> dict:
    trace = nsys_csv(rep, "cuda_gpu_trace", "Start (ns)")
    api = nsys_csv(rep, "cuda_api_sum", "Time (%)")
    k = [(int(r["Start (ns)"]), int(r["Duration (ns)"]), r["Name"]) for r in trace if r.get("Duration (ns)")]
    if not k:
        raise SystemExit(f"{rep}: no GPU activity captured")
    launches = sum(int(r["Num Calls"]) for r in api if r.get("Name", "").startswith("cudaGraphLaunch"))
    # decode: both engines run one CUDA graph per forward pass, and all kernels of a graph
    # launch share its correlation id. nsys start / stop cut the first and last launch, so
    # steps = kernels in graph launches / kernels in a complete launch (the most common size).
    sizes = [n for n in Counter(r["CorrId"] for r in trace if r.get("Duration (ns)")).values() if n >= 20]
    if phase == "prefill":
        steps = 1
    elif sizes:
        full = Counter(sizes).most_common(1)[0][0]
        steps = round(sum(sizes) / full, 2)
    else:
        steps = client_steps
    k.sort()
    wall = k[-1][0] + k[-1][1] - k[0][0]
    busy, end = 0, k[0][0]  # union of kernel intervals (streams may overlap)
    for s, d, _ in k:
        if s + d > end:
            busy += s + d - max(s, end)
            end = s + d
    per_cls: dict[str, float] = {}
    per_kernel: dict[str, list[float]] = {}
    for _, d, n in k:
        per_cls[classify(n)] = per_cls.get(classify(n), 0) + d
        acc = per_kernel.setdefault(n, [0, 0])
        acc[0] += d
        acc[1] += 1
    us = lambda ns: round(ns / steps / 1e3, 2)  # noqa: E731
    top = sorted(per_kernel.items(), key=lambda kv: -kv[1][0])[:15]
    return {
        "steps": steps, "graph_launches": launches, "client_steps": client_steps,
        "wall_us": us(wall), "busy_us": us(busy), "idle_share": round(1 - busy / wall, 4),
        "kernels_per_step": round(len(k) / steps, 1),
        "classes_us": {c: us(v) for c, v in sorted(per_cls.items(), key=lambda kv: -kv[1])},
        "top": [{"name": n[:160], "class": classify(n), "us": us(v[0]), "calls": round(v[1] / steps, 2)}
                for n, v in top],
        "kernels": [{"name": n[:300], "us": us(v[0]), "calls": round(v[1] / steps, 2)}
                    for n, v in sorted(per_kernel.items(), key=lambda kv: -kv[1][0])],
    }


def reanalyze(path: str) -> None:
    """Recompute every row of a results file from its saved reports (e.g. after a class fix)."""
    root = os.path.dirname(HERE)
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    for r in rows:
        r.update(analyze(os.path.join(root, r["report"]), r["client_steps"], r["phase"]))
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"reanalyzed {len(rows)} rows in {path}")


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--reanalyze":
        reanalyze(sys.argv[2])
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for name, kw in [("--engine", dict(choices=["vllm", "sglang", "tandem"], required=True)), ("--python", dict(required=True)),
                     ("--model", dict(required=True)), ("--served-name", dict(required=True))]:
        ap.add_argument(name, **kw)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--mem", type=float, default=None)
    ap.add_argument("--extra-args", default="")
    ap.add_argument("--kv-cache-gb", type=float, default=4.0, help="tandem only")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--batches", default="1,16")
    ap.add_argument("--input-len", type=int, default=1024)
    ap.add_argument("--prefill-len", type=int, default=4096)
    ap.add_argument("--window-steps", type=int, default=64)
    ap.add_argument("--warm-tokens", type=int, default=16)
    ap.add_argument("--margin-tokens", type=int, default=2048,
                    help="output tokens per request beyond the window (8B on 24 GB: 512, to stay inside the KV pool)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results", "profiles"))
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    a.python, a.model = os.path.expanduser(a.python), os.path.expanduser(a.model)
    a.mem = a.mem if a.mem is not None else (0.85 if a.engine == "sglang" else 0.90)

    env_meta = run_matrix.environment(a)
    a.tag = a.tag or f"{a.engine}-{env_meta['engine_version']}"
    os.makedirs(a.out_dir, exist_ok=True)
    stem = os.path.join(a.out_dir, f"{a.tag}-{a.served_name}-{time.strftime('%Y%m%d-%H%M%S')}")
    session = f"prof-{os.getpid()}"
    cmd = ["nsys", "launch", f"--session-new={session}", "-t", "cuda,nvtx", "--cuda-graph-trace=node",
           "--trace-fork-before-exec=true", *run_matrix.launch_cmd(a)]
    print("server:", " ".join(shlex.quote(x) for x in cmd), flush=True)
    base = f"http://127.0.0.1:{a.port}"
    log_path = stem + ".log"
    env = dict(os.environ, **dict(kv.split("=", 1) for kv in a.env))
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    tok = load_tokenizer(a.model)

    def capture(phase: str, batch: int, run) -> None:
        rep = f"{stem}.{phase}"
        start = lambda: nsys("start", f"--session={session}", "-o", rep, "--force-overwrite=true",  # noqa: E731
                             "--sample=none", "--cpuctxsw=none")
        stop = lambda: nsys("stop", f"--session={session}")  # noqa: E731  returns once the report is written
        client_steps = run(start, stop)
        rep += ".nsys-rep"
        row = {"tag": a.tag, "model": a.served_name, "phase": phase, "batch": batch, **env_meta,
               "extra_args": a.extra_args, "server_env": a.env, **analyze(rep, client_steps, phase),
               "report": os.path.relpath(rep, os.path.dirname(HERE))}
        print(json.dumps({k: v for k, v in row.items() if k not in ("top",)}, ensure_ascii=False), flush=True)
        if a.save:
            with open(a.save, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def decode(batch: int):
        def run(start, stop) -> int:
            prompts = build_random(tok, batch, a.input_len, 0, seed=1000 + batch)
            # room for the window plus the steps that pass while nsys start / stop run; small enough
            # that batch x (input + max_tokens) stays well inside the KV pool (no admission limits,
            # no preemption: the whole batch must be running during the capture)
            max_tokens = a.warm_tokens + a.window_steps + a.margin_tokens
            streams = [Stream(base + "/v1/completions", {
                "model": a.served_name, "prompt": p.prompt, "max_tokens": max_tokens, "temperature": 0.0,
                "ignore_eos": True, "stream": True}) for p in prompts]
            for s in streams:
                s.start()
            while min(s.chunks for s in streams) < a.warm_tokens:
                if any(s.error for s in streams) or all(not s.is_alive() for s in streams):
                    raise SystemExit(f"decode-{batch}: {[s.error for s in streams]}")
                time.sleep(0.005)
            start()
            c0 = min(s.chunks for s in streams)
            while min(s.chunks for s in streams) < c0 + a.window_steps:
                if any(not s.is_alive() for s in streams):
                    raise SystemExit(f"decode-{batch}: a request ended before the window was full "
                                     "(raise --margin-tokens)")
                time.sleep(0.002)
            c1 = min(s.chunks for s in streams)
            stop()
            for s in streams:
                s.join()
            return c1 - c0
        capture(f"decode-{batch}", batch, run)

    def prefill() -> None:
        def run(start, stop) -> int:
            p = build_random(tok, 1, a.prefill_len, 1, seed=7)[0]
            start()
            s = Stream(base + "/v1/completions", {"model": a.served_name, "prompt": p.prompt, "max_tokens": 1,
                                                 "temperature": 0.0, "ignore_eos": True, "stream": True})
            s.start()
            s.join()
            stop()
            if s.error:
                raise SystemExit(f"prefill: {s.error}")
            return 1
        capture("prefill", 1, run)

    try:
        run_matrix.wait_ready(base, proc, log_path, 1200)
        for p in build_random(tok, 2, 256, 32, seed=1):  # warm-up
            Stream(base + "/v1/completions", {"model": a.served_name, "prompt": p.prompt, "max_tokens": 32,
                                              "temperature": 0.0, "stream": True}).run()
        for b in [int(x) for x in a.batches.split(",")]:
            decode(b)
        prefill()
    finally:
        run_matrix.stop(proc)
    print("done:", a.tag, a.served_name, flush=True)


if __name__ == "__main__":
    main()
