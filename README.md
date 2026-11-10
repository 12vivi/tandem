# Tandem

面向"单 GPU + CPU"的 LLM 推理框架原型。设计方案、调研结论和每个组件的出处见 [docs/设计方案.md](docs/设计方案.md)。

> 状态：v0.1 原型。引擎主干、规划器和三路径专家执行可以在 CPU 上端到端运行，数值已与 HF Transformers 对齐。GPU 快速路径（FlashInfer、CUDA Graph）、C++ CPU 内核和性能对比还没有做，要在 GPU 机器上完成（见下方"路线图"）。

## 已实现

| 模块 | 内容 | 出处（详见设计方案 §4.0） |
|---|---|---|
| `scheduler/` | 连续批处理；token 预算下的分块 prefill，decode 优先；KV 不够时先淘汰前缀缓存，再按重算方式抢占 | Orca (OSDI'22)、Sarathi-Serve (OSDI'24)、vLLM V1 |
| `cache/` | 引用计数的分页 KV 块；按页粒度组织的 radix 前缀树（LRU 淘汰、路径加锁） | PagedAttention (SOSP'23)、RadixAttention (SGLang, NeurIPS'24) |
| `layers/`, `models/` | Llama / Qwen2 / Qwen3 / Qwen3-MoE，参数名与 HF 一致，可直接加载 safetensors；参考 attention 后端（CPU/GPU 通用） | HF Transformers 参考实现 |
| `hybrid/` | **三路径专家执行**：每一步、每个专家在"GPU 常驻 / 换入 GPU / CPU 计算"中选择，使 PCIe 路径与 CPU 路径尽量同时结束；GPU 专家槽用 LRU 管理 | KTransformers (SOSP'25)、SGLang Paged Experts、自适应专家分配 (arXiv 2610.23006) |
| `planner/` | 硬件画像（预设或实测）、屋顶线代价模型、显存放置计划、流式交叉点 T* | FlexGen (ICML'23)、MoE-Lightning (ASPLOS'25) |
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

- **P0 基线**：在租用的 GPU 机器上复现 vLLM / SGLang / KTransformers / llama.cpp 的基线（`benchmarks/README.md`）。
- **P1 GPU 快速路径**：FlashInfer attention 后端、CUDA Graph、Marlin / FP8 GEMM，主循环改为 GPU 常驻、无同步，引擎拆成独立进程。
- **P2 CPU 与异构**：C++ CPU 执行器（kt-kernel / ggml / IQK 内核、按 NUMA 分组的线程池），CUDA Graph 内用 host 回调同步。
- **P3 差异化**：带宽优先的 CPU GEMV、流式 prefill、节点内 PD 分离、卸载感知投机解码、三级 KV。

## 许可证

未定（见设计方案 §10 待确认事项）。
