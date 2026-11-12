# 基准测试

对比规则以 [设计方案 §6](../docs/设计方案.md) 为准。这里只记录可直接执行的步骤。

## 1. 引擎离线吞吐（本机即可）

```bash
python benchmarks/bench_offline.py --model models/Qwen3-0.6B --num-prompts 16 --input-len 128 --output-len 64
```

加 `--save benchmarks/results/xxx.jsonl` 可以归档结果（该目录已被 git 忽略）。

## 2. 对齐矩阵（设计方案 §1.2，在 GPU 服务器上）

所有引擎都用同一个压测客户端（`bench_serving.py`）访问 OpenAI 兼容接口。`run_matrix.py` 一次测一个引擎：启动服务、等就绪、从日志读出 KV 容量、跑完所有格子、关停并等显存释放。结果按行追加到 `--save` 指定的 JSONL，每行带引擎版本、KV 容量等元数据。

1. 准备对手环境（P0 过渡方案，见 `p0/setup_engines.sh` 的说明）：

   ```bash
   bash benchmarks/p0/setup_engines.sh
   ```

2. 在本机生成 ShareGPT 子集（固定种子，规则与 `vllm bench serve` 相同），再拷到服务器的 `benchmarks/data/`：

   ```bash
   python benchmarks/make_sharegpt_subset.py --sharegpt benchmarks/data/ShareGPT_V3_unfiltered_cleaned_split.json --tokenizer models/Qwen3-8B --out benchmarks/data/sharegpt_qwen3.jsonl
   ```

3. 每个引擎各跑一次。`--kv-budget` 取各引擎 KV 容量中最小的那个，保证大家跑同一组格子；先用 `--quick` 冒烟：

   ```bash
   python benchmarks/run_matrix.py --engine vllm --python ~/venvs/vllm-0.32/bin/python --model ~/models/Qwen3-8B --served-name qwen3-8b --sharegpt benchmarks/data/sharegpt_qwen3.jsonl --save benchmarks/results/p0-parity.jsonl
   ```

   `--engine` 可选 `vllm`、`sglang`、`tandem`。不改代码的绕行办法放在 `--extra-args`（服务参数）或 `--env`（环境变量）里，会一起记进结果。

   `--kv-budget` 只决定跑哪些格子，不改变引擎实际分到的 KV。KV 容量会限制并发时（例如 24 GB 卡上的 8B），还要把各引擎的 KV 池调成同样大小，否则离线吞吐比的是 KV 容量而不是引擎：
   1. 先用 `--quick` 冒烟，读出各引擎默认的 KV 容量，取最小值 T。
   2. vLLM 加 `--extra-args "--kv-cache-memory-bytes <T × 每 token 的 KV 字节数>"`。Qwen3-8B 每 token 147,456 字节（36 层 × 8 个 KV 头 × 128 维 × K/V 两份 × 2 字节）。T 取 16 的倍数，正好是整数个块。
   3. SGLang 加 `--extra-args "--max-total-tokens T"`。
   4. 所有引擎都加 `--kv-budget T`，并检查结果里的 `kv_capacity_tokens` 都等于 T。

   3090 服务器（驱动 550，系统 nvcc 12.4，PATH 里的默认编译器是 `/usr/local/gcc-11.2`）上需要的设置：

   | 引擎 | `--env` | 原因 |
   |---|---|---|
   | 全部 | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DO_NOT_TRACK=1` | 服务器不能上网，避免联网超时 |
   | vLLM | `VLLM_NO_USAGE_STATS=1 VLLM_USE_FLASHINFER_SAMPLER=0` | FlashInfer 0.7 现场编译采样内核要用 CUDA ≥12.8 的 nvcc（`--compress-mode`）。压测全是贪心解码，只取 argmax，不受影响 |
   | SGLang | `LD_LIBRARY_PATH=/usr/local/gcc-11.2/lib64` | FlashInfer 用 GCC 11.2 现场编译的 attention 内核需要它的 libstdc++，系统自带的版本太旧 |

   SGLang 还要加 `--extra-args "--cuda-graph-max-bs 64"`：24 GB 的卡默认只把 CUDA graph 捕获到 batch 24，更大的 batch 退回逐个启动内核，小模型会明显掉速（P0 基线报告 §3.3）。

4. 汇总成表格：每格取各次的中位数并给出波动；比值按同一请求切片逐次配对后取中位数。加 `--ours <tag>` 时按设计方案 §1.2 判定是否对齐：

   ```bash
   PYTHONIOENCODING=utf-8 python benchmarks/summarize.py benchmarks/results/p0-0.6b.jsonl --model qwen3-0.6b
   ```

## 3. 公平性检查清单

- 固定版本：vLLM 0.33.0、SGLang 0.5.23、KTransformers v0.7.2（配 `sglang-kt`）、llama.cpp 固定 build；记录驱动、CUDA、PyTorch 版本。
- 使用相同的权重，量化按字节对齐；KV dtype 和容量、max-model-len、最大并发保持一致。
- 先预热再测；前缀缓存要么各引擎都关闭，要么每轮之间清空。
- 投机解码要么全部关闭，要么全部开启并使用同一种草稿，同时报告接受率。
- 对手调到最佳配置，例如扫描 llama.cpp 的 `--n-cpu-moe` 并尝试 `--fit`。
- 记录运行环境是原生 Linux 还是 WSL2。
- 每组至少跑 3 次，报告中位数和波动（`run_matrix.py` 默认 `--repeat 3`）。
