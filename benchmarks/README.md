# 基准测试

对比规则以 [设计方案 §6](../docs/设计方案.md) 为准。这里只记录可直接执行的步骤。

## 1. 引擎离线吞吐（本机即可）

```bash
python benchmarks/bench_offline.py --model models/Qwen3-0.6B --num-prompts 16 --input-len 128 --output-len 64
```

加 `--save benchmarks/results/xxx.jsonl` 可以归档结果（该目录已被 git 忽略）。

## 2. 服务级对比（P0，在租用的 Linux GPU 机器上）

所有引擎都用同一个压测客户端访问 OpenAI 兼容接口。

1. 启动被测服务（每次只启动一个）：

   ```bash
   tandem serve --model /data/models/Qwen3-8B --port 8000
   ```

   ```bash
   vllm serve /data/models/Qwen3-8B --port 8000
   ```

   ```bash
   python -m sglang.launch_server --model-path /data/models/Qwen3-8B --port 8000
   ```

2. 用同一个客户端压测（示例：随机 1k 输入、256 输出，并发扫描）：

   ```bash
   vllm bench serve --backend openai --base-url http://127.0.0.1:8000 --endpoint /v1/completions --model Qwen3-8B --dataset-name random --random-input-len 1024 --random-output-len 256 --num-prompts 200 --max-concurrency 16 --ignore-eos
   ```

## 3. 公平性检查清单

- 固定版本：vLLM 0.33.0、SGLang 0.5.23、KTransformers v0.7.2（配 `sglang-kt`）、llama.cpp 固定 build；记录驱动、CUDA、PyTorch 版本。
- 使用相同的权重，量化按字节对齐；KV dtype 和容量、max-model-len、最大并发保持一致。
- 先预热再测；前缀缓存要么各引擎都关闭，要么每轮之间清空。
- 投机解码要么全部关闭，要么全部开启并使用同一种草稿，同时报告接受率。
- 对手调到最佳配置，例如扫描 llama.cpp 的 `--n-cpu-moe` 并尝试 `--fit`。
- 记录运行环境是原生 Linux 还是 WSL2。
- 每组至少跑 2 次，报告中位数和波动。
