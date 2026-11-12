#!/bin/bash
# P0 interim competitor environments, installed offline from a local wheelhouse.
#
# The pinned versions (vLLM 0.33.0, SGLang 0.5.23; design doc §6.1) need driver >= 580, and
# SGLang 0.5.11+ also needs glibc >= 2.34. On the 3090 server (driver 550 / CUDA 12.4,
# glibc 2.31) the newest versions that install are:
#   vLLM 0.32.0+cu129    torch 2.13.0+cu129   (wheel from the v0.32.0 GitHub release;
#                                               runs on driver 550 via CUDA minor-version compat)
#   SGLang 0.5.10        torch 2.9.1+cu129    (last pure-Python SGLang wheel, 2026-04-05)
# Results with these are development references only, not parity (design doc §6.1 option 3).
#
# envs/*.in are the resolver inputs; envs/pylock.*.toml and envs/req-*.txt are the locked
# sets, resolved for Linux x86_64 / glibc 2.31 / CPython 3.12 with
#   uv pip compile <env>.in --python-platform x86_64-manylinux_2_31 --python-version 3.12 \
#     --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu129 \
#     --index-strategy unsafe-best-match --prerelease=if-necessary-or-explicit \
#     --exclude-newer <release date + 1>
# (exclude-newer: SGLang 2026-04-06, vLLM 2026-10-19, i.e. dependencies as of each release).
# The exact command is in the header of each pylock file. Three fixes keep the vLLM set on
# CUDA 12 (any CUDA 13 library fails to load on driver 550):
#   - vllm==0.32.0+cu129 is not on PyPI (PyPI serves the CUDA 13 build). It is resolved with
#     --find-links pointing at a directory holding only that wheel (v0.32.0 GitHub release)
#     and --exclude-newer-package vllm=<today>, so its own metadata applies: it needs
#     humming-kernels[cu12], whereas the PyPI build pulls in CUDA 13 runtime packages. The
#     pylock entry is then pointed back at the GitHub release URL.
#   - torchcodec on PyPI is a CUDA 13 build (libcudart.so.13), and the cu129 index stops at
#     0.16, which targets torch 2.12; vllm.in pins the 0.17.0+cpu build by URL (vLLM only uses
#     it to decode audio on the CPU).
#   - cuda-toolkit 12.9.1 (required by torch) is pinned to its PyPI file by URL: the copy on
#     the PyTorch index has no upload time, which --exclude-newer rejects.
#
# Usage on the server:  WHEELS=~/wheels bash benchmarks/p0/setup_engines.sh [vllm|sglang]
set -euo pipefail
PY=${PY:-$HOME/opt/python3.12/bin/python3}
WHEELS=${WHEELS:-$HOME/wheels}
ENVS=$(cd "$(dirname "$0")/envs" && pwd)
declare -A VENV=([vllm]="$HOME/venvs/vllm-0.32" [sglang]="$HOME/venvs/sglang-0.5.10")

for e in "${@:-vllm sglang}"; do
  for name in $e; do
    venv=${VENV[$name]}
    [ -x "$venv/bin/python" ] || "$PY" -m venv "$venv"
    "$venv/bin/python" -m pip install --quiet --no-index --find-links "$WHEELS" -r "$ENVS/req-$name.txt"
    "$venv/bin/python" -c "import torch, importlib; m = importlib.import_module('$name'); \
print('$name', m.__version__, '| torch', torch.__version__, '| cuda ok:', torch.cuda.is_available())"
  done
done
