"""Projection layers: nn.Linear, or a quantized layer when the checkpoint is quantized."""

from __future__ import annotations

from typing import Any, Optional

from torch import nn


def make_linear(in_features: int, out_features: int, bias: bool = False, quant: Optional[Any] = None) -> nn.Module:
    """``quant``: tandem.model_config.QuantConfig or None. AWQ layers hold the checkpoint tensors
    (qweight / qzeros / scales, no dense weight is ever allocated) until ``finalize_quantized``."""
    if quant is None:
        return nn.Linear(in_features, out_features, bias=bias)
    if bias:
        raise NotImplementedError("quantized projections with a bias")
    from tandem.layers.awq import W4Linear  # Triton: only imported for quantized models

    return W4Linear(in_features, out_features, quant.group_size)


def finalize_quantized(model: nn.Module, device_name: Optional[str] = None) -> int:
    """After the weights are loaded and on the GPU: repack every quantized layer into its kernel
    layout and attach this GPU's kernel configs. Returns how many layers."""
    n = 0
    for mod in model.modules():
        finalize = getattr(mod, "finalize", None)
        if finalize is not None and getattr(mod, "awq_layout", False):
            finalize(device_name)
            n += 1
    return n
