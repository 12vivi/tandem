"""Sampling and speculative verification.

``filtered_probs`` applies temperature, top-k, top-p and min-p per row. ``verify_draft``
implements speculative sampling (Leviathan et al., ICML'23) specialised to deterministic
drafts (n-gram / lookup drafters): accept draft token d with probability p(d), otherwise
resample from p with d removed. The output distribution equals the target distribution;
in greedy mode the output equals plain greedy decoding.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from tandem.sampling_params import SamplingParams


def filtered_probs(logits: torch.Tensor, params: Sequence[SamplingParams]) -> torch.Tensor:
    """logits: [N, V] float32 -> probabilities [N, V] after all sampling filters.

    Which filters apply is decided from the parameters on the host, so nothing here waits
    for the GPU (the overlap scheduler relies on that)."""
    device = logits.device
    temp = torch.tensor([max(p.temperature, 1e-5) for p in params], device=device, dtype=logits.dtype)
    logits = logits / temp[:, None]
    vocab = logits.shape[-1]

    if any(0 < p.top_k < vocab for p in params):
        top_k = torch.tensor([p.top_k if 0 < p.top_k < vocab else vocab for p in params], device=device)
        kth = torch.gather(logits.sort(dim=-1, descending=True).values, 1, (top_k - 1)[:, None])
        logits = logits.masked_fill(logits < kth, float("-inf"))

    probs = torch.softmax(logits, dim=-1)

    if any(p.top_p < 1.0 for p in params):
        top_p = torch.tensor([p.top_p for p in params], device=device, dtype=probs.dtype)
        sorted_p, idx = probs.sort(dim=-1, descending=True)
        cum = sorted_p.cumsum(dim=-1)
        remove = (cum - sorted_p) > top_p[:, None]  # always keeps the top token
        sorted_p = sorted_p.masked_fill(remove, 0.0)
        probs = torch.zeros_like(probs).scatter_(1, idx, sorted_p)

    if any(p.min_p > 0 for p in params):
        min_p = torch.tensor([p.min_p for p in params], device=device, dtype=probs.dtype)
        thresh = min_p[:, None] * probs.max(dim=-1, keepdim=True).values
        probs = probs.masked_fill(probs < thresh, 0.0)

    return probs / probs.sum(dim=-1, keepdim=True)


def sample_tokens(logits: torch.Tensor, params: Sequence[SamplingParams], generators: Sequence) -> torch.Tensor:
    """One token per row, as an int64 tensor on the logits' device, without host syncs.
    All-greedy batches are one argmax; other rows sample one by one (each may carry its own
    seeded generator)."""
    if all(p.greedy for p in params):
        return logits.argmax(dim=-1)
    out = []
    for i, (p, g) in enumerate(zip(params, generators)):
        if p.greedy:
            out.append(logits[i].argmax().view(1))
        else:
            out.append(torch.multinomial(filtered_probs(logits[i : i + 1], [p])[0], 1, generator=g))
    return torch.cat(out)


def sample_token(
    logits: torch.Tensor, params: SamplingParams, generator: Optional[torch.Generator]
) -> int:
    """Sample one token from a single row of logits [V]."""
    if params.greedy:
        return int(logits.argmax())
    probs = filtered_probs(logits[None], [params])[0]
    return int(torch.multinomial(probs, 1, generator=generator))


def verify_draft(
    logits: torch.Tensor,
    draft: Sequence[int],
    params: SamplingParams,
    generator: Optional[torch.Generator],
) -> list[int]:
    """logits: [len(draft) + 1, V]; row j scores the token after position j.

    Returns the accepted draft prefix plus one corrected or bonus token.
    """
    out: list[int] = []
    if params.greedy:
        target = logits.argmax(dim=-1).tolist()
        for j, d in enumerate(draft):
            if target[j] != d:
                out.append(target[j])
                return out
            out.append(d)
        out.append(target[len(draft)])
        return out

    probs = filtered_probs(logits, [params] * logits.shape[0])
    for j, d in enumerate(draft):
        p = probs[j]
        u = torch.rand((), generator=generator, device=p.device)
        if u < p[d]:
            out.append(d)
            continue
        residual = p.clone()
        residual[d] = 0.0
        total = residual.sum()
        residual = residual / total if total > 0 else p
        out.append(int(torch.multinomial(residual, 1, generator=generator)))
        return out
    out.append(int(torch.multinomial(probs[len(draft)], 1, generator=generator)))
    return out
