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

4. 汇总成表格：每格取各次的中位数并给出波动；比值按同一请求切片逐次配对后取中位数。加 `--ours <tag>` 时按设计方案 §1.2 判定是否对齐：吞吐不低于最强对手的 0.95 倍，TTFT、TPOT 的 p50 不高于 1.05 倍、p99 不高于 1.10 倍，≥90% 的格子达标，且吞吐没有低于 0.85 倍的格子。

   ```bash
   PYTHONIOENCODING=utf-8 python benchmarks/summarize.py benchmarks/results/p0-0.6b.jsonl --model qwen3-0.6b
   ```

## 3. A 场景：低 batch decode（设计方案 §1.3、§6.3）

还是用 `run_matrix.py`，只换格子：输入 512 / 输出 512，batch 1、4、8；另加 4K 提示词看 TTFT。`--scenario` 写进结果，用来和对齐矩阵区分：

```bash
python benchmarks/run_matrix.py --engine sglang --python ~/venvs/sglang-0.5.10/bin/python \
    --model ~/models/Qwen3-0.6B --served-name qwen3-0.6b --shapes 512x512,4096x16 --levels 1,4,8 \
    --scenario A1 --save benchmarks/results/p0-a.jsonl
```

对手要调到最佳配置。SGLang 另测一组 `--enable-torch-compile --torch-compile-max-bs 8`（加 `--tag` 区分）。vLLM 0.32 默认已开 torch.compile、完整 CUDA graph 和异步调度。A2 用 AWQ 权重时，两边都加 `--dtype float16`。

## 4. B 场景：KV 复用（设计方案 §1.3、§6.3）

- **客户端** `bench_reuse.py`，提示词直接发 token ID，每个引擎收到的请求逐 token 相同：
  - `multiturn`（B1）：回放 ShareGPT 里至少 4 轮的对话，轮间有指数分布的思考时间。下一轮拼进历史的是数据集里的回复，不是引擎生成的文本，所以生成部分的 KV 不会被复用，复用率比真实对话低。
  - `docs`（B2）：一组 8K–24K token 的随机 token 文档，请求按 Zipf 分布选文档，泊松到达，每次附一个不同的短问题。
  - 每条结果都给出"理想复用率"（缓存无限大时能复用的 token 占比）和引擎报告的命中率。
- **多轮子集**在本机生成一次，再拷到服务器：

  ```bash
  python benchmarks/bench_reuse.py subset --sharegpt benchmarks/data/ShareGPT_V3_unfiltered_cleaned_split.json \
      --tokenizer models/Qwen3-8B --out benchmarks/data/sharegpt_multiturn_qwen3.jsonl
  ```

- **运行器** `run_reuse.py`：
  - 打开前缀缓存，用 `--kv-tokens` 把各引擎的显存 KV（L1）固定成同样大小；`--host-gb` 加一层主机内存（L2），vLLM 用原生 KV 卸载，SGLang 用 HiCache。
  - 依次跑多轮（各会话数档位）、长文档（各到达率档位）和一组无复用的对照，每次运行前都重启服务，互不共享缓存。

  ```bash
  python benchmarks/run_reuse.py --engine sglang --python ~/venvs/sglang-0.5.10/bin/python \
      --model ~/models/Qwen3-8B --served-name qwen3-8b --kv-tokens 32992 --host-gb 32 \
      --extra-args "--cuda-graph-max-bs 64" --save benchmarks/results/p0-b.jsonl
  ```

## 5. 逐算子剖析

`profile_engine.py` 用 `nsys launch` 启动服务（参数与对齐矩阵相同），分别抓 batch 1、batch 16 的稳态 decode 和一次 4K 预填充，按算子类别统计每步的 GPU 时间。需要 PATH 里有 nsys（CUDA 12.4 自带 2023.4）。8B 在 24 GB 卡上要加 `--margin-tokens 512`，保证 batch 16 的请求在采集期间都留在 KV 里：

```bash
PATH=/usr/local/cuda-12.4/bin:$PATH python benchmarks/profile_engine.py --engine vllm \
    --python ~/venvs/vllm-0.32/bin/python --model ~/models/Qwen3-8B --served-name qwen3-8b \
    --margin-tokens 512 --save benchmarks/results/p0-profile.jsonl
```

改了分类规则后，`--reanalyze <结果文件>` 可以直接用保存的报告重新统计。`summarize.py` 认得剖析结果和 B 场景结果，会输出相应的表格。

tandem 另有两个诊断开关，默认关闭，通过 `--env` 传给服务：
- `TANDEM_STEP_TIMING=N`：每 N 步打印一次每步主机耗时的分布（调度、构造输入、plan、graph replay、等 GPU 等），用来区分瓶颈在 CPU 还是 GPU。
- `TANDEM_TRACE=<文件>`：引擎核心按时间写事件，每行一条 JSON：请求进入引擎、每步的组成（prefill 了哪些请求、带几个 decode）、首 token 发出。
  - 客户端设 `BENCH_DUMP_DIR=<目录>` 后，`bench_serving.py` 会把每个请求的发送时间、TTFT、各 token 间隔写进这个目录。
  - 两边都用 `perf_counter`（Linux 上是 CLOCK_MONOTONIC），时间可以直接对齐。P1 对齐报告 §4 就是这样查清突发请求的处理过程的。

## 6. 公平性检查清单

- 固定版本：vLLM 0.33.0、SGLang 0.5.23、KTransformers v0.7.2（配 `sglang-kt`）、llama.cpp 固定 build；记录驱动、CUDA、PyTorch 版本。
- 使用相同的权重，量化按字节对齐；KV dtype 和容量、max-model-len、最大并发保持一致。
- 先预热再测；前缀缓存要么各引擎都关闭，要么每轮之间清空。
- 投机解码要么全部关闭，要么全部开启并使用同一种草稿，同时报告接受率。
- 对手调到最佳配置，例如扫描 llama.cpp 的 `--n-cpu-moe` 并尝试 `--fit`。
- 记录运行环境是原生 Linux 还是 WSL2。
- 每组至少跑 3 次，报告中位数和波动（`run_matrix.py` 默认 `--repeat 3`）。
