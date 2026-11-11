#!/bin/bash
# P0 CPU baseline: llama.cpp vs tandem C-mode on the same CPU, with the same client.
# Layout on the server: ~/models (HF dir + GGUF files), ~/llama.cpp (CPU build), ~/tandem (repo + .venv).
# GGUF files: convert_hf_to_gguf.py --outtype f32|q8_0, then llama-quantize ... Q4_K_M.
set -u
# llama.cpp was built with /usr/local/gcc-11.2; its libstdc++ is newer than Ubuntu 20.04's.
export LD_LIBRARY_PATH=/usr/local/gcc-11.2/lib64
M=$HOME/models; L=$HOME/llama.cpp/build/bin; R=$HOME/p0-results; T=$HOME/tandem
echo "=== llama-bench (8 threads, CPU only; pp512 = prefill, tg128 = decode) ==="
for q in f32 q8_0 q4_k_m; do
  $L/llama-bench -m $M/qwen3-0.6b-$q.gguf -t 8 -ngl 0 -p 512 -n 128 -r 2 2>/dev/null | grep -E "^\|.*(pp|tg)[0-9]+"
done
bench() {  # $1 port, $2 tag
  $T/.venv/bin/python $T/benchmarks/bench_serving.py --base-url http://127.0.0.1:$1 --model qwen3-0.6b \
    --tokenizer $M/Qwen3-0.6B --input-len 256 --output-len 64 --num-prompts 16 --concurrency 1,4 \
    --tag $2 --save $R/results.jsonl 2>&1 | grep "^{" | python3 -c "
import sys, json
for l in sys.stdin:
    d = json.loads(l); print('%-28s c=%-2d  %7.1f out tok/s  TTFT %7.0f ms  TPOT %6.1f ms  failed=%d' % (d['tag'], d['concurrency'], d['output_tok_per_s'], d['ttft_ms']['mean'] or -1, d['tpot_ms']['mean'] or -1, d['failed']))"
}
for q in f32 q4_k_m; do
  echo "=== llama-server ($q, 8 threads, 4 slots) ==="
  $L/llama-server -m $M/qwen3-0.6b-$q.gguf -t 8 -ngl 0 -c 8192 -np 4 --port 8200 > /tmp/llama-server-$q.log 2>&1 &
  pid=$!; curl -s --retry 60 --retry-connrefused --retry-delay 2 -o /dev/null http://127.0.0.1:8200/health
  bench 8200 llama.cpp-cpu-$q; kill $pid; wait $pid 2>/dev/null
done
echo "=== tandem C-mode (fp32 reference, 8 threads) ==="
$T/.venv/bin/tandem serve --model $M/Qwen3-0.6B --device cpu --dtype float32 --cpu-threads 8 --max-num-seqs 4 --kv-cache-gb 2 --port 8100 --served-model-name qwen3-0.6b > /tmp/tandem-cpu.log 2>&1 &
pid=$!; curl -s --retry 60 --retry-connrefused --retry-delay 2 -o /dev/null http://127.0.0.1:8100/health
bench 8100 tandem-v0.1-cpu-fp32; kill $pid; wait $pid 2>/dev/null
echo DONE
