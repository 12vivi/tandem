"""Summarize benchmark results (JSONL) into Markdown tables.

run_matrix / bench_serving rows: each cell (scenario x concurrency) is reported as the median
over its runs, with the spread (max - min) / median of the output throughput. With --ours,
every cell is also checked against the parity rule of the design doc (§1.2): ours / best
competitor >= 0.95 in at least 90% of the cells, and no cell below 0.85.
profile_engine rows (field "phase") and bench_reuse rows (field "workload") get their own
tables.

  python benchmarks/summarize.py benchmarks/results/p0-0.6b.jsonl --model qwen3-0.6b
  python benchmarks/summarize.py results.jsonl --model qwen3-8b --ours tandem-0.2.0
  python benchmarks/summarize.py benchmarks/results/p0-profile.jsonl
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
        "tpot_p50": statistics.median(r["tpot_ms"]["p50"] for r in runs),
        "tpot_p99": statistics.median(r["tpot_ms"]["p99"] for r in runs),
        "failed": sum(r["failed"] for r in runs),
    }


LATENCY_LIMITS = {"ttft_p50": 1.05, "ttft_p99": 1.10, "tpot_p50": 1.05, "tpot_p99": 1.10}  # design doc §1.2


def latency_misses(ours: dict, ref: dict) -> list[str]:
    """Latency percentiles of ours above the limit x those of the reference engine, i.e. the
    competitor with the higher throughput in the cell (§1.2: "compare with the better of vLLM
    and SGLang"; one engine, not the best value of each metric: their schedulers trade TTFT
    against TPOT differently, so the per-metric best is a point neither engine reaches)."""
    out = []
    for key, limit in LATENCY_LIMITS.items():
        if ref.get(key) and ours.get(key) and ours[key] > limit * ref[key]:
            out.append(f"{key} {ours[key] / ref[key]:.2f}")
    return out


def paired_ratio(a_runs: list[dict], b_runs: list[dict]) -> float | None:
    """Median over runs of a/b throughput. Run i of every engine uses the same request slice
    (same seed and rotation), so pairing by run removes the slice-to-slice variance."""
    b = {r["run"]: r["output_tok_per_s"] for r in b_runs}
    rs = [r["output_tok_per_s"] / b[r["run"]] for r in a_runs if b.get(r["run"])]
    return statistics.median(rs) if rs else None


PROFILE_CLASSES = ["gemm", "attention", "norm", "rope", "activation", "kv write", "sampling", "memcpy/memset", "other"]


def profile_tables(rows: list[dict]) -> None:
    """profile_engine.py rows: per-step GPU time by operator class (µs), one table per model."""
    for model in sorted({r["model"] for r in rows}):
        print(f"\n### {model}：每步 GPU 时间（µs）\n")
        print("| 阶段 | 引擎 | 墙钟 | GPU 忙 | 空闲 | 内核数 | " + " | ".join(PROFILE_CLASSES) + " |")
        print("|---|---|---|---|---|---|" + "---|" * len(PROFILE_CLASSES))
        sel = sorted((r for r in rows if r["model"] == model), key=lambda r: (r["phase"], r["tag"]))
        for r in sel:
            cls = r["classes_us"]
            print(f"| {r['phase']} | {r['tag']} | {r['wall_us']:.0f} | {r['busy_us']:.0f} | {r['idle_share']:.1%} | "
                  f"{r['kernels_per_step']:.0f} | " + " | ".join(f"{cls.get(c, 0):.0f}" for c in PROFILE_CLASSES) + " |")
        for r in sel:
            print(f"\n{model} {r['phase']} {r['tag']} 耗时最多的内核（µs / 步，调用次数 / 步）：\n")
            for t in r["top"][:8]:
                print(f"- {t['us']:.0f} µs，{t['calls']:.2f} 次，{t['class']}：`{t['name'][:110]}`")


def a_tables(rows: list[dict]) -> None:
    """A-scenario rows (run_matrix --scenario A1/A2): TPOT p50 at each batch, TTFT of the long prompt."""
    tags = sorted({r["tag"] for r in rows})
    for model in sorted({r["model"] for r in rows}):
        sel = [r for r in rows if r["model"] == model]
        table = cells(sel)
        runs = max(len(v) for c in table.values() for v in c.values())
        print(f"\n### {model}（{sel[0].get('matrix')}）：512/512 为 TPOT p50 ms，4096/16 为 TTFT p50 ms，各 {runs} 次的中位数\n")
        print("| 形状 | batch | " + " | ".join(tags) + " | 最强 |")
        print("|---|---|" + "---|" * (len(tags) + 1))
        for k in sorted(table, key=lambda k: (k[0][1].endswith("/16"), conc_key(k[1]))):
            st = {t: stats(table[k][t]) for t in tags if t in table[k]}
            long_prompt = k[0][1].endswith("/16")  # 4096/16: the TTFT cell
            val = {t: s["ttft_p50"] if long_prompt else s["tpot_p50"] for t, s in st.items()}
            best = min(val, key=val.get)
            print(f"| {k[0][1]} | {k[1]} | " + " | ".join(f"{val[t]:.2f}" if t in val else "-" for t in tags)
                  + f" | {best} |")


def reuse_tables(rows: list[dict]) -> None:
    """bench_reuse.py rows (B scenarios): one table per workload."""
    def fmt(x, spec):
        return "-" if x is None else format(x, spec)
    for wl in ("multiturn", "docs"):
        sel = [r for r in rows if r.get("workload") == wl]
        if not sel:
            continue
        level = "sessions" if wl == "multiturn" else "rate_rps"
        grp = "follow" if wl == "multiturn" else "all"
        print(f"\n### {wl}（{'后续轮次' if wl == 'multiturn' else '计量请求'}）\n")
        print("| 档位 | 配置 | 请求 | 失败 | TTFT p50 / p99 ms | TPOT p50 ms | SLO 达标 | 命中率 | 理想复用 | 输出 tok/s | 工作集峰值 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(sel, key=lambda r: (r[level], r["tag"])):
            g = r[grp]
            print(f"| {r[level]} | {r['tag']} | {g['requests']} | {g['failed']} | {fmt(g['ttft_ms']['p50'], '.0f')} / "
                  f"{fmt(g['ttft_ms']['p99'], '.0f')} | {fmt(g['tpot_ms']['p50'], '.1f')} | {fmt(g['slo_ok_share'], '.0%')} | "
                  f"{fmt(g['hit_rate'], '.0%')} | {fmt(g['ideal_reuse'], '.0%')} | {r['output_tok_per_s']:.0f} | "
                  f"{r.get('working_set_peak_tokens', '-')} |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--model", default="", help="only rows for this served model name")
    ap.add_argument("--ours", default="", help="tag to check against the best of the other tags")
    a = ap.parse_args()

    rows = [json.loads(l) for f in a.files for l in open(f, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if not a.model or r.get("model") == a.model]
    if rows and "phase" in rows[0]:
        profile_tables(rows)
        return
    if rows and "workload" in rows[0]:
        reuse_tables(rows)
        return
    if rows and all(str(r.get("matrix", "")).startswith("A") for r in rows):
        a_tables(rows)
        return
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
        head = (others + ([a.ours, f"{a.ours} / 最强对手", "延迟超限"] if a.ours else [])
                + ([f"{tags[0]} / {tags[1]}"] if pair else []))
        print("| 场景 | 并发 | " + " | ".join(head) + " |")
        print("|---|---|" + "---|" * len(head))
        ratios, passed = [], []
        for k in order:
            st = {t: stats(table[k][t]) for t in tags if t in table[k]}
            got = {t: s["tput"] for t, s in st.items()}
            cols = [f"{got[t]:.1f}" if t in got else "-" for t in others]
            if a.ours:
                best = max((t for t in others if t in got), key=got.get, default=None)
                ratio = paired_ratio(table[k][a.ours], table[k][best]) if a.ours in got and best else None
                if ratio is not None:
                    misses = latency_misses(st[a.ours], st[best])
                    ratios.append(ratio)
                    passed.append(ratio >= PASS and not misses)
                    mark = "" if ratio >= PASS else (" ✗" if ratio < FLOOR else " ·")
                    cols += [f"{got[a.ours]:.1f}", f"{ratio:.2f}{mark}", "；".join(misses) or "-"]
                else:
                    cols += [f"{got[a.ours]:.1f}" if a.ours in got else "-", "-", "-"]
            if pair:
                ratio = paired_ratio(table[k][tags[0]], table[k][tags[1]]) if len(got) == 2 else None
                cols.append(f"{ratio:.2f}" if ratio is not None else "-")
            print(f"| {k[0][1]} | {k[1]} | " + " | ".join(cols) + " |")
        if ratios:
            ok = sum(passed) / len(passed)
            verdict = "达标" if ok >= SHARE and min(ratios) >= FLOOR else "未达标"
            print(f"\n对齐判定（§1.2）：{ok:.0%} 的格子达标（吞吐 ≥ {PASS} 倍，TTFT / TPOT 的 p50 ≤ 1.05 倍、"
                  f"p99 ≤ 1.10 倍；要求 ≥ {SHARE:.0%}），吞吐最低 {min(ratios):.2f}（要求 ≥ {FLOOR}）→ {verdict}")


if __name__ == "__main__":
    main()
