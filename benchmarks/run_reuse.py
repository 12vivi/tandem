"""Run the KV-reuse scenarios (design doc §1.3 B1 / B2) against one engine configuration.

Prefix caching is on. --kv-tokens pins the GPU KV pool (L1) to the same size on every
engine (vLLM --kv-cache-memory-bytes, SGLang --max-total-tokens). --host-gb adds a host
memory tier (L2) of that size: vLLM's native KV offloading (--kv-offloading-size) or
SGLang HiCache (--enable-hierarchical-cache --hicache-size); 0 = the engine's default,
GPU prefix cache only. The server is restarted before every run, so no run sees the cache
of another. Runs:
  multiturn  bench_reuse.py multiturn at each --sessions level
  docs       bench_reuse.py docs at each --rates level
  noreuse    bench_serving.py, random 1024/256 at concurrency 16: the cost of the tiers on a
             load without any reuse (written to <save>-noreuse.jsonl in the matrix format)

  python benchmarks/run_reuse.py --engine sglang --python ~/venvs/sglang-0.5.10/bin/python \\
      --model ~/models/Qwen3-8B --served-name qwen3-8b --kv-tokens 32992 --host-gb 32 \\
      --env LD_LIBRARY_PATH=/usr/local/gcc-11.2/lib64 --extra-args "--cuda-graph-max-bs 64" \\
      --dataset-path benchmarks/data/sharegpt_multiturn_qwen3.jsonl --save benchmarks/results/p0-b.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_matrix  # noqa: E402


def kv_bytes_per_token(model_dir: str) -> int:
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        c = json.load(f)
    head_dim = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
    return c["num_hidden_layers"] * c["num_key_value_heads"] * head_dim * 2 * 2  # K and V, 16-bit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--engine", choices=["vllm", "sglang", "tandem"], required=True)
    ap.add_argument("--python", required=True)
    ap.add_argument("--client-python", default=sys.executable)
    ap.add_argument("--model", required=True)
    ap.add_argument("--served-name", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--mem", type=float, default=None)
    ap.add_argument("--kv-tokens", type=int, required=True, help="GPU KV pool (L1) in tokens, same for every engine")
    ap.add_argument("--host-gb", type=int, default=0,
                    help="host-memory KV tier (L2) in GB (10^9 bytes); 0 = GPU prefix cache only. vLLM's native "
                         "offloading lives in /dev/shm, which is half of RAM by default (31.4 GiB on a 62 GB host)")
    ap.add_argument("--extra-args", default="")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--runs", default="multiturn,docs,noreuse")
    ap.add_argument("--dataset-path", default=os.path.join(HERE, "data", "sharegpt_multiturn_qwen3.jsonl"))
    ap.add_argument("--sessions", default="32,96,192")
    ap.add_argument("--think-mean", type=float, default=60.0)
    ap.add_argument("--ramp", type=float, default=60.0)
    ap.add_argument("--rates", default="0.2,0.4")
    ap.add_argument("--multiturn-args", default="", help="extra bench_reuse.py multiturn flags (smoke tests)")
    ap.add_argument("--docs-args", default="", help="extra bench_reuse.py docs flags (smoke tests)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--save", required=True)
    ap.add_argument("--log-dir", default=os.path.join(HERE, "results", "logs"))
    a = ap.parse_args()
    a.python, a.model = os.path.expanduser(a.python), os.path.expanduser(a.model)
    a.mem = a.mem if a.mem is not None else (0.85 if a.engine == "sglang" else 0.90)
    a.prefix_cache, a.kv_cache_gb = True, 0.0

    if a.engine == "tandem":  # --kv-cache-gb is GiB; a hair more than the exact size so that it floors to kv_tokens
        pin = ""
        a.kv_cache_gb = round((a.kv_tokens * kv_bytes_per_token(a.model) + 4096) / 2**30, 6)
    else:
        pin = (f"--kv-cache-memory-bytes {a.kv_tokens * kv_bytes_per_token(a.model)}" if a.engine == "vllm"
               else f"--max-total-tokens {a.kv_tokens}")
    tier = ""
    if a.host_gb:  # same bytes for all: vLLM takes GiB, SGLang and tandem GB (10^9)
        tier = {"vllm": f"--kv-offloading-size {a.host_gb * 1e9 / 2**30:.2f} --kv-offloading-backend native",
                "sglang": f"--enable-hierarchical-cache --hicache-size {a.host_gb:g}",
                "tandem": f"--host-kv-gb {a.host_gb:g}"}[a.engine]
    a.extra_args = " ".join(x for x in (a.extra_args, pin, tier) if x)

    env_meta = run_matrix.environment(a)
    a.tag = a.tag or f"{a.engine}-{env_meta['engine_version']}-" + (f"host{a.host_gb:g}g" if a.host_gb else "gpu")
    meta = {**env_meta, "kv_tokens": a.kv_tokens, "host_gb": a.host_gb, "max_model_len": a.max_model_len,
            "max_num_seqs": a.max_num_seqs, "extra_args": a.extra_args, "server_env": a.env}
    os.makedirs(a.log_dir, exist_ok=True)
    base = f"http://127.0.0.1:{a.port}"
    client = [a.client_python, os.path.join(HERE, "bench_reuse.py")]
    plan: list[tuple[str, list[str]]] = []
    for run in a.runs.split(","):
        if run == "multiturn":
            plan += [(f"multiturn-{s}", client + ["multiturn", "--sessions", s, "--dataset-path", a.dataset_path,
                                                  "--think-mean", str(a.think_mean), "--ramp", str(a.ramp),
                                                  *shlex.split(a.multiturn_args)])
                     for s in a.sessions.split(",")]
        elif run == "docs":
            plan += [(f"docs-{r}", client + ["docs", "--rate", r, *shlex.split(a.docs_args)]) for r in a.rates.split(",")]
        elif run == "noreuse":
            stem, ext = os.path.splitext(a.save)
            plan.append(("noreuse", [a.client_python, os.path.join(HERE, "bench_serving.py"), "--dataset", "random",
                                     "--input-len", "1024", "--output-len", "256", "--concurrency", "16",
                                     "--repeat", "3", "--tokenizer", a.model, "--save", f"{stem}-noreuse{ext}"]))
        else:
            raise SystemExit(f"unknown run {run}")

    cmd = run_matrix.launch_cmd(a)
    print("server:", " ".join(shlex.quote(x) for x in cmd), flush=True)
    env = dict(os.environ, **dict(kv.split("=", 1) for kv in a.env))
    for name, ccmd in plan:
        log_path = os.path.join(a.log_dir, f"{a.tag}-{name}-{time.strftime('%Y%m%d-%H%M%S')}.log")
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            run_matrix.wait_ready(base, proc, log_path, 1200)
            cap = run_matrix.kv_capacity(a.engine, log_path)
            if cap != a.kv_tokens:
                print(f"warning: {name}: server reports {cap} KV tokens, pinned {a.kv_tokens}", flush=True)
            run_meta = {**meta, "kv_capacity_tokens": cap}
            ccmd = ccmd + ["--base-url", base, "--model", a.served_name, "--tag", a.tag, "--meta", json.dumps(run_meta)]
            if name != "noreuse":
                ccmd += ["--save", a.save]
            print(f"+ {name}: {' '.join(shlex.quote(x) for x in ccmd[1:4])} ...", flush=True)
            subprocess.run(ccmd, check=True)
        finally:
            run_matrix.stop(proc)
    print("done:", a.tag, flush=True)


if __name__ == "__main__":
    main()
