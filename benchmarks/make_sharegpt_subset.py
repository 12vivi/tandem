"""Write a fixed-seed ShareGPT sample as JSONL for bench_serving.py (--dataset sharegpt).

The full ShareGPT JSON is ~650 MB; the GPU server only needs the sampled requests (a few
MB). Sampling and filters are the same as `vllm bench serve --dataset-name sharegpt`
(see bench_serving.sample_sharegpt), so the subset is reproducible from the source file.

  python benchmarks/make_sharegpt_subset.py \\
      --sharegpt benchmarks/data/ShareGPT_V3_unfiltered_cleaned_split.json \\
      --tokenizer models/Qwen3-8B --out benchmarks/data/sharegpt_qwen3.jsonl

Source: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
(ShareGPT_V3_unfiltered_cleaned_split.json). benchmarks/data/ is git-ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_serving import load_tokenizer, sample_sharegpt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sharegpt", required=True, help="original ShareGPT JSON")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--num", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    reqs = sample_sharegpt(args.sharegpt, load_tokenizer(args.tokenizer), args.num, args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in reqs:
            f.write(json.dumps({"prompt": r.prompt, "prompt_len": r.prompt_len, "output_len": r.output_len},
                               ensure_ascii=False) + "\n")
    with open(args.out, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    print(f"{len(reqs)} requests -> {args.out}  sha256={digest[:16]}  "
          f"prompt tokens mean {statistics.mean(r.prompt_len for r in reqs):.0f}, "
          f"output tokens mean {statistics.mean(r.output_len for r in reqs):.0f}")


if __name__ == "__main__":
    main()
