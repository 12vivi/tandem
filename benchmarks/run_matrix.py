"""Run the parity matrix (design doc §1.2) against one engine, end to end.

Starts the engine's OpenAI-compatible server, waits until it serves, reads its KV-cache
capacity from the log, runs bench_serving.py over every cell, then shuts the server down
and waits for the GPU to be released. Run it once per engine with the same matrix (same
--kv-budget) and compare the JSON lines.

Matrix v1 (Qwen3-8B BF16 on one GPU):
  fixed lengths  1024/256, 4096/256, 256/1024 at concurrency 1, 2, 4, 8, 16, 32
  ShareGPT       same concurrency levels
  offline        1,000 ShareGPT requests sent at once
A cell is kept only if concurrency x (input + output) fits in --kv-budget tokens (default:
this engine's own capacity; pass the smallest capacity across engines so that every engine
runs the same cells) and concurrency <= --max-num-seqs.

Example (on the GPU server, from the repo root):
  ~/tandem/.venv/bin/python benchmarks/run_matrix.py --engine vllm \\
      --python ~/venvs/vllm-0.32/bin/python --model ~/models/Qwen3-8B --served-name qwen3-8b \\
      --sharegpt benchmarks/data/sharegpt_qwen3.jsonl --save benchmarks/results/p0-parity.jsonl
  add --quick for a smoke test (one shape, concurrency 1 and 4, one run).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SHAPES = [(1024, 256), (4096, 256), (256, 1024)]
LEVELS = [1, 2, 4, 8, 16, 32]
SHAREGPT_MAX_TOKENS = 2048  # bench_serving's ShareGPT filter: prompt + output <= 2048
OFFLINE_REQUESTS = 1000

KV_PATTERNS = {  # how each engine reports its KV-cache capacity (tokens) at startup
    "vllm": [r"GPU KV cache size: ([\d,]+) tokens"],
    "sglang": [r"max_total_num_tokens=(\d+)"],
    "tandem": [r"(\d+) KV blocks x (\d+) tokens"],
}
VERSION_MODULE = {"vllm": "vllm", "sglang": "sglang", "tandem": "tandem"}


def launch_cmd(a: argparse.Namespace) -> list[str]:
    common_vllm_sglang = ["--host", "127.0.0.1", "--port", str(a.port)]
    if a.engine == "vllm":
        cmd = [a.python, "-m", "vllm.entrypoints.openai.api_server", "--model", a.model,
               "--served-model-name", a.served_name, "--dtype", a.dtype,
               "--max-model-len", str(a.max_model_len), "--gpu-memory-utilization", str(a.mem),
               "--max-num-seqs", str(a.max_num_seqs), "--no-enable-prefix-caching", "--seed", "0",
               *common_vllm_sglang]
    elif a.engine == "sglang":
        cmd = [a.python, "-m", "sglang.launch_server", "--model-path", a.model,
               "--served-model-name", a.served_name, "--dtype", a.dtype,
               "--context-length", str(a.max_model_len), "--mem-fraction-static", str(a.mem),
               "--max-running-requests", str(a.max_num_seqs), "--disable-radix-cache", "--random-seed", "0",
               *common_vllm_sglang]
    else:  # tandem
        cmd = [a.python, "-m", "tandem.entrypoints.cli", "serve", "--model", a.model,
               "--served-model-name", a.served_name, "--device", "cuda", "--dtype", a.dtype,
               "--max-model-len", str(a.max_model_len), "--max-num-seqs", str(a.max_num_seqs),
               "--kv-cache-gb", str(a.kv_cache_gb), "--no-prefix-caching", "--port", str(a.port)]
    return cmd + shlex.split(a.extra_args)


def sh(cmd: list[str], timeout: float = 60) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as exc:  # metadata only; never fatal
        return f"<{exc!r}>"


def environment(a: argparse.Namespace) -> dict:
    mod = VERSION_MODULE[a.engine]
    probe = (f"import torch, {mod}; print({mod}.__version__ if hasattr({mod}, '__version__') else '?', "
             f"torch.__version__, torch.version.cuda, sep='|')")
    ver = sh([a.python, "-c", probe], timeout=300).splitlines()
    ver = ver[-1].split("|") if ver else ["?", "?", "?"]
    gpu = sh(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]).splitlines()
    return {
        "engine": a.engine,
        "engine_version": ver[0] if len(ver) == 3 else "?",
        "torch": ver[1] if len(ver) == 3 else "?",
        "torch_cuda": ver[2] if len(ver) == 3 else "?",
        "gpu": gpu[0] if gpu else "?",
    }


def wait_ready(base: str, proc: subprocess.Popen, log_path: str, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"server exited with code {proc.returncode}; log tail:\n{tail(log_path)}")
        try:
            with urllib.request.urlopen(base + "/v1/models", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(3)
    raise SystemExit(f"server not ready after {timeout:.0f}s; log tail:\n{tail(log_path)}")


def tail(path: str, n: int = 40) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        return "".join(f.readlines()[-n:])


def kv_capacity(engine: str, log_path: str) -> int | None:
    with open(log_path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    for pat in KV_PATTERNS[engine]:
        found = re.findall(pat, text)
        if found:
            last = found[-1]
            if isinstance(last, tuple):  # tandem: blocks x block_size
                return int(last[0]) * int(last[1])
            return int(last.replace(",", ""))
    return None


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        for sig, wait in ((signal.SIGINT, 30), (signal.SIGTERM, 15), (signal.SIGKILL, 10)):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                break
            try:
                proc.wait(timeout=wait)
                break
            except subprocess.TimeoutExpired:
                continue
    for _ in range(30):  # child processes may hold GPU memory for a few more seconds
        used = sh(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        if used.split("\n")[0].strip().isdigit() and int(used.split("\n")[0]) < 1000:
            return
        time.sleep(2)
    print("warning: GPU memory not released after shutdown", file=sys.stderr)


def bench(a: argparse.Namespace, meta: dict, levels: list[str], extra: list[str]) -> None:
    cmd = [a.client_python, os.path.join(HERE, "bench_serving.py"),
           "--base-url", f"http://127.0.0.1:{a.port}", "--model", a.served_name, "--tokenizer", a.model,
           "--concurrency", ",".join(levels), "--repeat", str(a.repeat), "--tag", a.tag,
           "--meta", json.dumps(meta), *extra]
    if a.save:
        cmd += ["--save", a.save]
    print("+", " ".join(shlex.quote(x) for x in cmd[:4]), "...", " ".join(extra), "--concurrency", ",".join(levels), flush=True)
    if not a.dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--engine", choices=["vllm", "sglang", "tandem"], required=True)
    ap.add_argument("--python", required=True, help="python of the engine's environment")
    ap.add_argument("--client-python", default=sys.executable, help="python with httpx + transformers for the client")
    ap.add_argument("--model", required=True, help="model directory (also used as the tokenizer)")
    ap.add_argument("--served-name", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=64, help="same value for every engine (design doc §6.4 rule 2)")
    ap.add_argument("--mem", type=float, default=None,
                    help="vLLM --gpu-memory-utilization (default 0.90) / SGLang --mem-fraction-static (default 0.85)")
    ap.add_argument("--kv-cache-gb", type=float, default=4.0, help="tandem only")
    ap.add_argument("--extra-args", default="", help="extra server flags, e.g. workarounds; recorded in the results")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for the server process (repeatable)")
    ap.add_argument("--kv-budget", type=int, default=None, help="tokens; default = this engine's reported capacity")
    ap.add_argument("--sharegpt", default="", help="ShareGPT JSONL subset; enables the ShareGPT and offline cells")
    ap.add_argument("--shapes", default="", help='override, e.g. "1024x256,256x1024"')
    ap.add_argument("--levels", default="", help='override, e.g. "1,4,16"')
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="smoke test: 1024/256 at concurrency 1 and 4, one run")
    ap.add_argument("--startup-timeout", type=float, default=1200)
    ap.add_argument("--tag", default="", help="default: <engine>-<version>")
    ap.add_argument("--save", default="", help="append result lines here")
    ap.add_argument("--log-dir", default=os.path.join(HERE, "results", "logs"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    a.python = os.path.expanduser(a.python)
    a.model = os.path.expanduser(a.model)
    if a.mem is None:
        a.mem = 0.85 if a.engine == "sglang" else 0.90
    shapes = [tuple(int(x) for x in s.split("x")) for s in a.shapes.split(",")] if a.shapes else SHAPES
    levels = [int(x) for x in a.levels.split(",")] if a.levels else LEVELS
    if a.quick:
        shapes, levels, a.repeat, a.sharegpt = [(1024, 256)], [1, 4], 1, ""

    env_meta = environment(a)
    a.tag = a.tag or f"{a.engine}-{env_meta['engine_version']}"
    cmd = launch_cmd(a)
    print("server:", " ".join(shlex.quote(x) for x in cmd), flush=True)
    os.makedirs(a.log_dir, exist_ok=True)
    log_path = os.path.join(a.log_dir, f"{a.tag}-{time.strftime('%Y%m%d-%H%M%S')}.log")
    if a.dry_run:
        kv_cap, proc = a.kv_budget or 33000, None
    else:
        env = dict(os.environ, **dict(kv.split("=", 1) for kv in a.env))
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        t0 = time.time()
        try:
            wait_ready(f"http://127.0.0.1:{a.port}", proc, log_path, a.startup_timeout)
        except BaseException:
            stop(proc)
            raise
        print(f"ready in {time.time() - t0:.0f}s, log: {log_path}", flush=True)
        kv_cap = kv_capacity(a.engine, log_path)

    try:
        budget = a.kv_budget or kv_cap
        if not budget:
            raise SystemExit("could not read the KV capacity from the server log; pass --kv-budget")
        meta = {**env_meta, "kv_capacity_tokens": kv_cap, "kv_budget_tokens": budget, "max_num_seqs": a.max_num_seqs,
                "mem_fraction": a.mem, "max_model_len": a.max_model_len, "dtype": a.dtype, "matrix": "v1",
                "extra_args": a.extra_args, "server_env": a.env}
        print(f"KV capacity {kv_cap} tokens, budget {budget}", flush=True)
        fits = lambda c, tokens: c * tokens <= budget and c <= a.max_num_seqs  # noqa: E731
        for inp, out in shapes:
            ok = [str(c) for c in levels if fits(c, inp + out)]
            if ok:
                bench(a, meta, ok, ["--dataset", "random", "--input-len", str(inp), "--output-len", str(out)])
        if a.sharegpt:
            sg = ["--dataset", "sharegpt", "--dataset-path", a.sharegpt]
            ok = [str(c) for c in levels if fits(c, SHAREGPT_MAX_TOKENS)]
            if ok:
                bench(a, meta, ok, sg)
            bench(a, meta, ["inf"], sg + ["--num-prompts", str(OFFLINE_REQUESTS)])
    finally:
        if proc is not None:
            stop(proc)
    print("done:", a.tag, flush=True)


if __name__ == "__main__":
    main()
