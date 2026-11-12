"""Summarize run_matrix / bench_serving results (JSONL) into Markdown tables.

Each cell (scenario x concurrency) is reported as the median over its runs, with the spread
(max - min) / median of the output throughput. With --ours, every cell is also checked against
the parity rule of the design doc (§1.2): ours / best competitor >= 0.95 in at least 90% of
the cells, and no cell below 0.85.

  python benchmarks/summarize.py benchmarks/results/p0-0.6b.jsonl --model qwen3-0.6b
  python benchmarks/summarize.py results.jsonl --model qwen3-8b --ours tandem-0.2.0
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict

PASS, FLOOR, SHARE = 0.95, 0.85, 0.90  # design doc §1.2


def scenario(r: dict) -> tuple[int, str]:
    """(sort key, label) of the row's scenario."""
    if r["dataset"] == "random":
        order = {(1024, 256): 0, (4096, 256): 1, (256, 1024): 2}
        key = (r["input_len"], r["output_len"])
        return order.get(key, 3), f"{key[0]}/{key[1]}"
    if str(r["concurrency"]) in ("inf", "all"):
        return 5, "ShareGPT 离线"
    return 4, "ShareGPT"


def conc_key(c) -> float:
    return float("inf") if str(c) in ("inf", "all") else float(c)


def cells(rows: list[dict]) -> dict:
    """{(scenario, concurrency): {tag: [rows]}}"""
    out: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        out[(scenario(r), str(r["concurrency"]))][r["tag"]].append(r)
    return out


def stats(runs: list[dict]) -> dict:
    tput = [r["output_tok_per_s"] for r in runs]
    med = statistics.median(tput)
    return {
        "n": len(runs),
        "tput": med,
        "spread": (max(tput) - min(tput)) / med if med else 0.0,
        "ttft_p50": statistics.median(r["ttft_ms"]["p50"] for r in runs),
        "ttft_p99": statistics.median(r["ttft_ms"]["p99"] for r in runs),
        "tpot": statistics.median(r["tpot_ms"]["mean"] for r in runs),
        "failed": sum(r["failed"] for r in runs),
    }


def paired_ratio(a_runs: list[dict], b_runs: list[dict]) -> float | None:
    """Median over runs of a/b throughput. Run i of every engine uses the same request slice
    (same seed and rotation), so pairing by run removes the slice-to-slice variance."""
    b = {r["run"]: r["output_tok_per_s"] for r in b_runs}
    rs = [r["output_tok_per_s"] / b[r["run"]] for r in a_runs if b.get(r["run"])]
    return statistics.median(rs) if rs else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--model", default="", help="only rows for this served model name")
    ap.add_argument("--ours", default="", help="tag to check against the best of the other tags")
    a = ap.parse_args()

    rows = [json.loads(l) for f in a.files for l in open(f, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if not a.model or r.get("model") == a.model]
    table = cells(rows)
    order = sorted(table, key=lambda k: (k[0][0], conc_key(k[1])))
    tags = sorted({r["tag"] for r in rows})

    for tag in tags:
        meta = next(r for r in rows if r["tag"] == tag)
        print(f"\n### {tag}（{meta.get('model')}，KV 容量 {meta.get('kv_capacity_tokens')} token，"
              f"预算 {meta.get('kv_budget_tokens')}）\n")
        print("| 场景 | 并发 | 输出吞吐 tok/s | 波动 | TTFT p50 / p99 ms | TPOT ms | 失败 |")
        print("|---|---|---|---|---|---|---|")
        for k in order:
            if tag not in table[k]:
                continue
            s = stats(table[k][tag])
            print(f"| {k[0][1]} | {k[1]} | {s['tput']:.1f} | {s['spread']:.1%} | "
                  f"{s['ttft_p50']:.1f} / {s['ttft_p99']:.1f} | {s['tpot']:.2f} | {s['failed']} |")

    if len(tags) > 1:
        others = [t for t in tags if t != a.ours]
        pair = not a.ours and len(tags) == 2  # plain A vs B: show A / B
        print("\n### 对比（输出吞吐 tok/s，各格中位数；比值按同一请求切片逐次配对后取中位数）\n")
        head = others + ([a.ours, f"{a.ours} / 最强对手"] if a.ours else []) + ([f"{tags[0]} / {tags[1]}"] if pair else [])
        print("| 场景 | 并发 | " + " | ".join(head) + " |")
        print("|---|---|" + "---|" * len(head))
        ratios = []
        for k in order:
            got = {t: stats(table[k][t])["tput"] for t in tags if t in table[k]}
            cols = [f"{got[t]:.1f}" if t in got else "-" for t in others]
            if a.ours:
                best = max((t for t in others if t in got), key=got.get, default=None)
                ratio = paired_ratio(table[k][a.ours], table[k][best]) if a.ours in got and best else None
                if ratio is not None:
                    ratios.append(ratio)
                    mark = "" if ratio >= PASS else (" ✗" if ratio < FLOOR else " ·")
                    cols += [f"{got[a.ours]:.1f}", f"{ratio:.2f}{mark}"]
                else:
                    cols += [f"{got[a.ours]:.1f}" if a.ours in got else "-", "-"]
            if pair:
                ratio = paired_ratio(table[k][tags[0]], table[k][tags[1]]) if len(got) == 2 else None
                cols.append(f"{ratio:.2f}" if ratio is not None else "-")
            print(f"| {k[0][1]} | {k[1]} | " + " | ".join(cols) + " |")
        if ratios:
            ok = sum(r >= PASS for r in ratios) / len(ratios)
            verdict = "达标" if ok >= SHARE and min(ratios) >= FLOOR else "未达标"
            print(f"\n对齐判定（§1.2）：{ok:.0%} 的格子 ≥ {PASS}（要求 ≥ {SHARE:.0%}），"
                  f"最低 {min(ratios):.2f}（要求 ≥ {FLOOR}）→ {verdict}")


if __name__ == "__main__":
    main()
