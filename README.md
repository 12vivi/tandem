# Tandem

面向特定模型的 LLM 推理框架原型：先在选定模型（Qwen3 稠密系列，以 Qwen3-8B 为准）上与 vLLM / SGLang 性能对齐，再在特定场景做专项优化，包括低 batch decode 的算子（A 场景）和显存 / 主机内存 / SSD 多级 KV 存储（B 场景）。设计方案、调研结论和每个组件的出处见 [docs/设计方案.md](docs/设计方案.md)。

> 状态：v0.1 原型。引擎主干（调度、分页 KV、radix 缓存）和参考模型实现可以在 CPU / GPU 上端到端运行，数值已与 HF Transformers 对齐。GPU 快速路径（FlashInfer、CUDA Graph、融合算子）还没有做（见下方"路线图"）。与 vLLM / SGLang 的同机对比已用过渡版本跑完 Qwen3-0.6B 和 Qwen3-8B 的对齐矩阵。v0.1 在 0.6B 上只有较强对手的 0.06–0.12，见 [docs/P0基线报告.md](docs/P0基线报告.md)。

## 已实现

| 模块 | 内容 | 出处（详见设计方案 §4.0） |
|---|---|---|
| `scheduler/` | 连续批处理；token 预算下的分块 prefill，decode 优先；KV 不够时先淘汰前缀缓存，再按重算方式抢占 | Orca (OSDI'22)、Sarathi-Serve (OSDI'24)、vLLM V1 |
| `cache/` | 引用计数的分页 KV 块；按页粒度组织的 radix 前缀树（LRU 淘汰、路径加锁） | PagedAttention (SOSP'23)、RadixAttention (SGLang, NeurIPS'24) |
| `layers/`, `models/` | Llama / Qwen2 / Qwen3 / Qwen3-MoE，参数名与 HF 一致，可直接加载 safetensors；参考 attention 后端（CPU/GPU 通用） | HF Transformers 参考实现 |
| `hybrid/` | **三路径专家执行**：每一步、每个专家在"GPU 常驻 / 换入 GPU / CPU 计算"中选择，使 PCIe 路径与 CPU 路径尽量同时结束；GPU 专家槽用 LRU 管理。冻结至 P4 | KTransformers (SOSP'25)、SGLang Paged Experts、自适应专家分配 (arXiv 2610.23006) |
| `planner/` | 硬件画像（预设或实测）、屋顶线代价模型、显存放置计划、流式交叉点 T*。冻结至 P4，硬件画像复用到多级存储 | FlexGen (ICML'23)、MoE-Lightning (ASPLOS'25) |
| `spec/`, `layers/sampler.py` | n-gram 草稿 + 投机采样验证（贪心模式下与普通解码逐 token 一致）；temperature / top-k / top-p / min-p | Leviathan et al. (ICML'23)、Prompt Lookup Decoding |
| `entrypoints/` | OpenAI 兼容接口（completions / chat，SSE 流式，`include_usage`）；CLI：`serve` / `generate` / `plan` / `probe` | OpenAI API |

## 快速开始

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -e ".[dev]"
```

运行测试（不需要下载模型，用随机权重的小模型对齐 HF Transformers）：

```bash
.venv/Scripts/python -m pytest -q
```

查看放置计划（只需模型配置，不需要权重）：

```bash
.venv/Scripts/tandem plan --model Qwen3-30B-A3B --hw-profile rtx4090-desktop --weight-bits 4
```

用本地模型生成或启动服务（模型目录需包含 config.json、safetensors 和 tokenizer）：

```bash
.venv/Scripts/tandem generate --model models/Qwen3-0.6B --prompt "你好" --chat --max-tokens 64
```

```bash
.venv/Scripts/tandem serve --model models/Qwen3-0.6B --port 8000
```

Linux 上把 `.venv/Scripts/` 换成 `.venv/bin/`。

## 路线图（对应设计方案 §8）

- **P0 基线**（进行中）：解决服务器环境，跑 vLLM / SGLang 在 Qwen3-8B 上的对齐矩阵和场景基线，逐算子剖析（`benchmarks/README.md`、`docs/P0基线报告.md`）。对齐矩阵已用过渡版本跑完（0.6B、8B），场景基线和逐算子剖析还没做。
- **P1 对齐**：FlashInfer attention、CUDA Graph、融合小算子、批量采样、overlap 主循环、API 与引擎进程拆分；Qwen3-0.6B 和 Qwen3-8B BF16 达到 vLLM / SGLang 较好者的 0.95 倍以上。
- **P2 算子**：W4A16 / W8A8 先对齐，再做模型专用融合、形状专用 GEMV、低 batch decode 持久化内核（研究线）。
- **P3 多级存储**：L2 主机内存 / L3 NVMe KV 缓存、按层流水载入、载入 / 重算拆分、会话与 Agent 感知的保留和预取。P2 与 P3 可并行。
- **P4 扩展**：Qwen3-30B-A3B 与三路径专家执行、专用草稿的投机解码、其他 GPU 上复测。

## 许可证

未定（见设计方案 §10 待确认事项）。
