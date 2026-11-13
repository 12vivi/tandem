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
# checkpoint tensors that go into a slice of a fused model parameter:
# (pattern, fused name, shards in row order). Dense-MLP names only: MoE experts are
# "mlp.experts.E.gate_proj" and do not match.
_STACKED = [
    (re.compile(r"^(.*\.self_attn)\.(q|k|v)_proj\.(weight|bias|qweight|qzeros|scales)$"), "qkv_proj", ("q", "k", "v")),
    (re.compile(r"^(model\.layers\.\d+\.mlp)\.(gate|up)_proj\.(weight|qweight|qzeros|scales)$"), "gate_up_proj",
     ("gate", "up")),
]
# AWQ tensors ([K, N/8] / [K/G, N/8] / [K/G, N]) have the output dimension second: shards are
# columns, not rows
_AWQ_SUFFIXES = (".qweight", ".qzeros", ".scales")


def _stacked_target(name: str):
    """-> (fused parameter name, shard, all shards) or None."""
    for pattern, fused, shards in _STACKED:
        m = pattern.match(name)
        if m:
            return f"{m.group(1)}.{fused}.{m.group(3)}", m.group(2), shards
    return None


def _copy_shard(param: torch.Tensor, tensor: torch.Tensor, shard: str, shards: tuple, name: str) -> None:
    """Rows of the fused parameter, in shard order. qkv: q has its own size and k, v share
    one; gate_up: both halves are equal. Offsets are derived from the fused size. AWQ tensors
    are fused along their columns (the output dimension)."""
    if name.endswith(_AWQ_SUFFIXES):
        _copy_shard(param.t(), tensor.t(), shard, shards, name[: name.rindex(".")] + ".weight")
        return
    rows = tensor.shape[0]
    total = param.shape[0]
    if shards == ("q", "k", "v"):
        start = {"q": 0, "k": total - 2 * rows, "v": total - rows}[shard]
    else:
        start = {"gate": 0, "up": rows}[shard]
    if start < 0 or start + rows > total or tuple(tensor.shape[1:]) != tuple(param.shape[1:]):
        raise ValueError(f"shape mismatch for {name}: checkpoint {tuple(tensor.shape)} vs fused {tuple(param.shape)}")
    with torch.no_grad():
        param[start : start + rows].copy_(tensor.to(param.dtype))


def cast_parameters(model: nn.Module, dtype: torch.dtype) -> None:
    """Cast parameters only; float32 buffers such as RoPE tables keep their precision."""
    for p in model.parameters():
        p.data = p.data.to(dtype)


def load_weights(model: nn.Module, model_path: str) -> None:
    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no *.safetensors files in {model_path}")
    params = dict(model.named_parameters())
    # quantized layers keep their checkpoint tensors as buffers (tandem.layers.awq.W4Linear)
    params.update({n: b for n, b in model.named_buffers() if n.endswith(_AWQ_SUFFIXES)})
    loaded: set[str] = set()
    unexpected: list[str] = []
    shards_seen: dict[str, set[str]] = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if _IGNORED.search(name):
                    continue
                fused = _FUSED_EXPERTS.match(name)
                if fused:
                    loaded |= _load_fused_experts(params, fused.group(1), fused.group(2), f.get_tensor(name))
                    continue
                stacked = _stacked_target(name)
                if stacked is not None and stacked[0] in params:
                    target, shard, shards = stacked
                    _copy_shard(params[target], f.get_tensor(name), shard, shards, name)
                    seen = shards_seen.setdefault(target, set())
                    seen.add(shard)
                    if seen == set(shards):
                        loaded.add(target)
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
