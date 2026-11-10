"""Load Hugging Face safetensors checkpoints into tandem models."""

from __future__ import annotations

import glob
import logging
import os
import re

import torch
from safetensors import safe_open
from torch import nn

logger = logging.getLogger(__name__)

_IGNORED = re.compile(r"(rotary_emb\.inv_freq|\.attn\.bias|\.masked_bias)$")
# transformers >= 5 may save fused expert tensors: mlp.experts.gate_up_proj / mlp.experts.down_proj
_FUSED_EXPERTS = re.compile(r"^(model\.layers\.\d+\.mlp\.experts)\.(gate_up_proj|down_proj)$")


def cast_parameters(model: nn.Module, dtype: torch.dtype) -> None:
    """Cast parameters only; float32 buffers such as RoPE tables keep their precision."""
    for p in model.parameters():
        p.data = p.data.to(dtype)


def load_weights(model: nn.Module, model_path: str) -> None:
    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no *.safetensors files in {model_path}")
    params = dict(model.named_parameters())
    loaded: set[str] = set()
    unexpected: list[str] = []
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if _IGNORED.search(name):
                    continue
                fused = _FUSED_EXPERTS.match(name)
                if fused:
                    loaded |= _load_fused_experts(params, fused.group(1), fused.group(2), f.get_tensor(name))
                    continue
                if name not in params:
                    if name != "lm_head.weight":  # tied-embedding checkpoints may still store lm_head
                        unexpected.append(name)
                    continue
                tensor = f.get_tensor(name)
                param = params[name]
                if tuple(param.shape) != tuple(tensor.shape):
                    raise ValueError(f"shape mismatch for {name}: checkpoint {tuple(tensor.shape)} vs model {tuple(param.shape)}")
                with torch.no_grad():
                    param.copy_(tensor.to(param.dtype))
                loaded.add(name)
    if unexpected:
        logger.warning("ignored %d unexpected checkpoint tensors, e.g. %s", len(unexpected), unexpected[:3])
    missing = sorted(set(params) - loaded)
    if missing:
        raise RuntimeError(f"{len(missing)} parameters missing from checkpoint, e.g. {missing[:5]}")


def _load_fused_experts(params: dict, prefix: str, kind: str, tensor: torch.Tensor) -> set[str]:
    """Split fused [E, ...] expert tensors into per-expert Linear weights."""
    loaded = set()
    for e in range(tensor.shape[0]):
        if kind == "gate_up_proj":
            gate_name, up_name = f"{prefix}.{e}.gate_proj.weight", f"{prefix}.{e}.up_proj.weight"
            gate, up = params[gate_name], params[up_name]
            w = tensor[e]
            if w.shape[0] != gate.shape[0] + up.shape[0]:  # stored as [hidden, 2 * inter]
                w = w.t()
            with torch.no_grad():
                gate.copy_(w[: gate.shape[0]].to(gate.dtype))
                up.copy_(w[gate.shape[0] :].to(up.dtype))
            loaded |= {gate_name, up_name}
        else:
            down_name = f"{prefix}.{e}.down_proj.weight"
            down = params[down_name]
            w = tensor[e]
            if tuple(w.shape) != tuple(down.shape):
                w = w.t()
            with torch.no_grad():
                down.copy_(w.to(down.dtype))
            loaded.add(down_name)
    return loaded
