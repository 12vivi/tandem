# Tandem

面向 Qwen3 的大模型推理引擎（Python 主干，热点用 Triton 和 CUDA C++ 自写内核）。先在同一张 RTX 3090 上与 vLLM、SGLang 的性能对齐，再在三类场景上超过它们：小模型低并发的 decode、4 bit AWQ 量化模型的 decode、多轮对话和共享长文档的分层 KV 缓存。

> 状态：0.2.0.dev0（2026-11-17）。P0 基线、P1 对齐已完成；P2 算子、P3 多级 KV 存储部分达成。下表除 AWQ 并发 16–32 一行（P2 时测的）外，都是最终代码（提交 `cb23f41`）复测的结果。每项结论的测量条件、没达到的部分和原因都在 `docs/` 的阶段报告里。

## 结果

同一台机器、同一块 RTX 3090（驱动 550），对手是 vLLM 0.32.0 和 SGLang 0.5.10（过渡版本）。每格与对手里最快的配置比；对齐矩阵和 A 场景每格重复 3 次取中位数，B 场景压测时间长，每档跑一次（最终代码的 B1 跑了两次）。倍数一栏都是"越大越好"：吞吐是 tandem / 对手，延迟是对手 / tandem。

| 场景 | 指标 | tandem | 最强对手 | 倍数 | 详见 |
|---|---|---|---|---|---|
| 对齐矩阵，Qwen3-8B BF16（19 格） | 达标格数（吞吐 ≥0.95 倍，延迟 p50 ≤1.05、p99 ≤1.10 倍） | 17/19（P1 时 19/19） | — | 吞吐 1.01–1.02 | [P1 §8](docs/P1对齐报告.md) |
| 对齐矩阵，Qwen3-0.6B（25 格） | 同上 | 24/25 | — | 吞吐 1.02–1.21 | [P1 §8](docs/P1对齐报告.md) |
| A1 小模型低并发，Qwen3-0.6B | TPOT p50，batch 1 / 4 / 8（ms） | 1.98 / 2.37 / 2.88 | 2.14 / 2.51 / 3.01 | 1.081 / 1.059 / 1.045 | [P2](docs/P2算子报告.md) |
| A1，Qwen3-1.7B | 同上 | 4.57 / 4.98 / 5.55 | 4.76 / 5.24 / 5.74 | 1.042 / 1.052 / 1.034 | [P2](docs/P2算子报告.md) |
| A2 4 bit AWQ，Qwen3-8B | 同上 | 6.63 / 7.22 / 8.53 | 7.22 / 7.70 / 8.94 | 1.089 / 1.066 / 1.048 | [P2](docs/P2算子报告.md) |
| A2，并发 16 / 24 / 32 | TPOT p50（ms），对比 vLLM | 11.7 / 19.1 / 20.5 | 12.4 / 17.1 / 20.0 | 1.06 / 0.89 / 0.98 | [P2 §3.4](docs/P2算子报告.md) |
| B2 共享长文档，0.1 req/s | TTFT p50（s），各加 30 GB 主机内存层 | 0.95 | 2.82 | 2.97 | [P3 §8](docs/P3分层缓存报告.md) |
| B1 多轮对话，192 会话 | TTFT p50（ms）/ SLO 达标 | 127 / 99.0%（第二次 126 / 99.3%） | 156 / 96.3% | 1.22–1.24 | [P3 §8](docs/P3分层缓存报告.md) |
| 无复用负载 | 开主机层后的吞吐变化 | −0.6% | −1.1% | — | [P3 §8](docs/P3分层缓存报告.md) |

没达到的：8B 对齐矩阵用最终代码重跑是 17/19，比 90% 的线差一格（ShareGPT 并发 8 的 TPOT p99 是推迟调度的代价；1024/256 并发 16 的 TTFT p50 是闭环请求的中位数跳台阶，均值没变；见 P1 对齐报告 §8）；A1 0.6B 的目标是 1.2 倍，1.7B 的目标是 1.05 倍（只有 batch 4 压线）；A2 batch 8 在 1.05 倍的目标线上下（几次复测 1.048–1.051 倍）；AWQ 4096 token 提示词的 TTFT 是对手的 0.79–0.94 倍；AWQ 并发 24 比 vLLM 慢 11%。原因和下一步见各报告的"没完成的"一节。B1 的 TTFT 目标（≤130 ms）在 P3 结束时差 3 ms，收尾回归把推迟调度的等待改为自旋后，最终代码两次复测是 127.4 / 125.5 ms，达到了。

## 实现了什么

| 模块 | 内容 |
|---|---|
| `scheduler/` | 连续批处理；分块 prefill（每步 2048 token）；prefill 优先；短 prefill 优先、单独成步；KV 不够时淘汰前缀缓存，或把长 prefill 换出到主机内存层；推迟调度（下一步在上一步快结束时才定，刚到的请求赶得上） |
| `cache/` | 分页 KV、按页组织的 radix 前缀树；主机内存层：pinned 页池，写回走拷贝引擎（不占 SM），载入按层流水、与前向重叠 |
| `engine/`、`executor/` | FlashInfer attention；decode 的 CUDA Graph（按 batch 分档，batch ≥8 另有不切分 KV 的 graph）；overlap 主循环（CPU 准备下一步时 GPU 在算这一步）；API 与引擎分两个进程 |
| `layers/`、`csrc/` | 自写内核，按形状查表选配置。Triton：小 batch GEMV、split-KV decode attention、残差加 RMSNorm。CUDA C++（AWQ W4A16）：CUDA 核心 GEMV、tensor core decode 内核（权重直接读进 MMA 片段）、M > 16 的融合反量化 GEMM（cp.async 流水、最后一轮 split-K） |
| `models/` | Qwen3 稠密模型（BF16 / FP16，AWQ 4 bit），参数名与 HF 一致，直接加载 safetensors；Llama、Qwen2、Qwen3-MoE 的参考实现 |
| `entrypoints/` | OpenAI 兼容接口（completions / chat，SSE 流式，用量里报告命中缓存的 token）；CLI：`serve` / `generate` / `plan` / `probe` |
| `hybrid/`、`planner/`、`spec/` | 三路径专家执行、放置规划、n-gram 投机解码：v0.1 的参考实现，冻结到 P4 |

## 快速开始

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -e ".[dev]"
```

运行测试。没有 GPU 也能跑：用随机权重的小模型对齐 HF Transformers，GPU 测试自动跳过。

```bash
.venv/bin/python -m pytest -q
```

GPU 上启动服务（Linux，sm_80 及以上）。我们的环境是 torch 2.14.1+cu126、flashinfer-python 0.6.7.post2、Triton 3.8：

```bash
.venv/bin/tandem serve --model models/Qwen3-8B --kv-cache-gb 4 --port 8000
```

- AWQ 模型（AutoAWQ 的 gemm 格式）加 `--dtype float16`。第一次启动会编译 CUDA 内核，需要 `CUDA_HOME` 和 nvcc（我们用 CUDA 12.4、gcc 11.2）；
- 主机内存 KV 层：加 `--host-kv-gb 30`；
- 每项优化都有关闭开关，方便做对照，例如 `--no-small-batch-gemv`、`--no-small-batch-attention`、`--no-small-batch-norm`、`--no-late-schedule`、环境变量 `TANDEM_W4_GEMM=0`（完整列表见 `tandem serve --help`）。

Windows 上把 `.venv/bin/` 换成 `.venv/Scripts/`。

## 评测与复现

- [benchmarks/README.md](benchmarks/README.md)：对齐矩阵（`run_matrix.py`）、复用场景（`run_reuse.py`）、逐算子剖析（`profile_engine.py`）、内核微基准（`bench_*.py`）的用法，以及服务器上要加的环境变量；
- 每份阶段报告的最后一节是复现命令；原始结果在 `benchmarks/results/`，不进版本库。

## 文档

| 文档 | 内容 |
|---|---|
| [设计方案](docs/设计方案.md) | 目标、对齐判据、架构、调研、路线图 |
| [P0 基线报告](docs/P0基线报告.md) | 对手基线、逐算子剖析、A / B 场景的定义 |
| [P1 对齐报告](docs/P1对齐报告.md) | 对齐结果和三处关键修正 |
| [P2 算子报告](docs/P2算子报告.md) | 小 batch 内核、AWQ W4A16 |
| [P3 实施计划](docs/P3实施计划.md)、[P3 分层缓存报告](docs/P3分层缓存报告.md) | 主机内存 KV 层的设计和结果 |
| [面试复习](docs/面试复习.md) | 每项优化做了什么、为什么、证据在哪 |

## 路线图（设计方案 §8）

- **P0 基线**：完成（2026-11-13）。
- **P1 对齐**：完成（2026-11-12）。最终代码重跑：8B 17/19，差一格；0.6B 24/25（P1 对齐报告 §8）。
- **P2 算子**：部分达成。A2 达标；A1 未达标，0.6B 到 1.2 倍要靠持久化内核，列为研究线。
- **P3 多级存储**：部分达成。主机内存层完成，B2 和 B1 的 SLO 达标，B1 的 TTFT 差 3 ms（P3 结束时）；最终代码复测时 B1 的 TTFT 也达到了（127.4 / 125.5 ms，P3 分层缓存报告 §8）；L3（NVMe）在这台机器上做不了。
- **P4 扩展**：未开始。Qwen3-30B-A3B 与三路径专家执行、专用草稿的投机解码、在其他 GPU 上复测。

## 许可证

未定（见设计方案 §10 待确认事项）。
