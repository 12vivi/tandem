from __future__ import annotations

from typing import Optional

import torch

_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float64": torch.float64,  # numerics tests only
}


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
    return dev


def resolve_dtype(dtype: str, device: torch.device, config_dtype: Optional[str] = None) -> torch.dtype:
    if dtype != "auto":
        if dtype not in _DTYPES:
            raise ValueError(f"unknown dtype {dtype!r}; choose from {sorted(_DTYPES)}")
        return _DTYPES[dtype]
    if device.type == "cpu":
        # Most desktop CPUs lack native BF16 GEMM, so FP32 is both faster and exact there.
        return torch.float32
    return _DTYPES.get(str(config_dtype), torch.bfloat16) if config_dtype else torch.bfloat16


def cdiv(a: int, b: int) -> int:
    return -(-a // b)
