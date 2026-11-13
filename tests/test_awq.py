"""AWQ W4A16 (tandem.layers.awq): repack from the AutoAWQ "gemm" layout, dequantize, decode kernels.

The AWQ packer below follows AutoAWQ's convention (nibble i of an int32 holds output column
ORDER[i] of its group of 8); the real-checkpoint check is in tests/test_real_model.py-style runs
on the server (Qwen3-8B-AWQ).
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
pytest.importorskip("triton")

from tandem.layers.awq import W4Linear, interleave_gate_up, pack_rows, repack_awq, w4_dequant  # noqa: E402

AWQ_PACK_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)


def pack_awq(values: torch.Tensor) -> torch.Tensor:
    """[R, 8 C] values 0..15 -> [R, C] int32 in AutoAWQ order."""
    r, c8 = values.shape
    v = values.view(r, c8 // 8, 8).to(torch.int64)
    out = torch.zeros(r, c8 // 8, dtype=torch.int64, device=values.device)
    for i, col in enumerate(AWQ_PACK_ORDER):
        out |= v[:, :, col] << (4 * i)
    return out.to(torch.int32)


def random_awq(k: int, n: int, group: int = 128, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randint(0, 16, (k, n), device="cuda", generator=g, dtype=torch.int32)
    z = torch.randint(0, 16, (k // group, n), device="cuda", generator=g, dtype=torch.int32)
    s = (torch.rand(k // group, n, device="cuda", generator=g) * 0.02 + 0.001).to(torch.float16)
    w = ((q - z.repeat_interleave(group, dim=0)).float() * s.float().repeat_interleave(group, dim=0)).t()  # [N, K]
    return pack_awq(q), pack_awq(z), s, w


@pytest.mark.parametrize("k,n", [(256, 64), (1024, 384), (4096, 1024)])
def test_repack_and_dequant(k, n):
    qweight, qzeros, scales, w_ref = random_awq(k, n)
    packed, s, z = repack_awq(qweight, qzeros, scales, 128)
    assert packed.shape == (n, k // 8) and s.shape == (n, k // 128) and z.shape == (n, k // 128)
    w = w4_dequant(packed, s, z, 128)
    assert torch.allclose(w.float(), w_ref, atol=1e-3, rtol=1e-3)


def test_dequant_deinterleave():
    """w4_dequant(deinterleave=True) on interleave_gate_up rows == w4_dequant on the original rows."""
    qweight, qzeros, scales, _ = random_awq(256, 96, seed=7)
    packed, s, z = repack_awq(qweight, qzeros, scales, 128)
    rows = (interleave_gate_up(t) for t in (packed, s, z))
    assert torch.equal(w4_dequant(*rows, 128, deinterleave=True), w4_dequant(packed, s, z, 128))


def test_pack_rows_layout():
    """int32 (j, cc) of row n and group g holds k = 128 g + 32 cc + 8 j + 2 q + e in nibble q + 4 e."""
    n, k = 24, 384
    q = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.int32)
    packed = pack_rows(q).view(n, k // 128, 4, 4)  # [n, g, j, cc]
    for g, j, cc, qq, e in ((0, 0, 0, 0, 0), (1, 2, 3, 1, 1), (2, 3, 1, 3, 0), (2, 1, 2, 2, 1)):
        nib = (packed[:, g, j, cc] >> (4 * (qq + 4 * e))) & 0xF
        assert torch.equal(nib, q[:, 128 * g + 32 * cc + 8 * j + 2 * qq + e]), (g, j, cc, qq, e)


def _layer(k, n, seed, silu_mul=False):
    qweight, qzeros, scales, w_ref = random_awq(k, n, seed=seed)
    layer = W4Linear(k, n, 128).cuda()
    layer.qweight.copy_(qweight)
    layer.qzeros.copy_(qzeros)
    layer.scales.copy_(scales)
    if silu_mul:
        assert layer.enable_silu_mul()
    layer.finalize()
    return layer, w_ref


def _swiglu_ref(x, w_ref):
    """float32 SwiGLU on the float16-rounded weights the kernels use ((q - z) * s rounded once):
    the product silu(gate) * up would otherwise amplify that rounding beyond any kernel error."""
    gate, up = (x.float() @ w_ref.half().float().t()).chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


def test_w4linear_small_and_large_batch():
    k, n = 512, 256
    layer, w_ref = _layer(k, n, seed=2)
    assert not layer.awq_layout and not hasattr(layer, "qweight") and layer.packed.shape == (n, k // 8)
    for m in (1, 3, 8, 12, 16, 37):  # CUDA kernels for m <= 16, dequantize + cuBLAS above
        x = torch.randn(m, k, device="cuda", dtype=torch.float16)
        y = layer(x).float()
        assert torch.allclose(y, x.float() @ w_ref.t(), atol=0.05, rtol=0.02), m
    x = torch.empty(3 * k + 1, device="cuda", dtype=torch.float16)[1:].view(3, k).normal_()  # not 16-byte aligned
    assert torch.allclose(layer(x).float(), x.float() @ w_ref.t(), atol=0.05, rtol=0.02)


def test_w4linear_silu_mul():
    """gate_up with SwiGLU: rows interleaved at finalize, epilogue at decode (m <= 16), dequantize +
    cuBLAS + de-interleave for prefill and for unaligned inputs."""
    k, n = 512, 384
    layer, w_ref = _layer(k, n, seed=5, silu_mul=True)
    assert not W4Linear(k, n + 8, 128).enable_silu_mul()  # N % 16 != 0: not fusable
    for m in (1, 3, 8, 12, 16, 37):
        x = torch.randn(m, k, device="cuda", dtype=torch.float16)
        y = layer(x).float()
        assert y.shape == (m, n // 2)
        assert torch.allclose(y, _swiglu_ref(x, w_ref), atol=0.03, rtol=0.02), (m, (y - _swiglu_ref(x, w_ref)).abs().max())
    x = torch.empty(3 * k + 1, device="cuda", dtype=torch.float16)[1:].view(3, k).normal_()
    assert torch.allclose(layer(x).float(), _swiglu_ref(x, w_ref), atol=0.03, rtol=0.02)


# ------------------------------------------------------------------ end to end: tiny AWQ checkpoint
def _quantize_awq(w: torch.Tensor, group: int = 128):
    """w [N, K] -> AWQ (qweight [K, N/8], qzeros [K/G, N/8], scales [K/G, N]) and the dequantized [N, K]."""
    n, k = w.shape
    wg = w.t().float().reshape(k // group, group, n)
    lo, hi = wg.amin(1), wg.amax(1)
    s = ((hi - lo).clamp(min=1e-5) / 15).half().float()
    z = torch.round(-lo / s).clamp(0, 15)
    q = torch.clamp(torch.round(wg / s[:, None, :]) + z[:, None, :], 0, 15)
    w_deq = ((q - z[:, None, :]) * s[:, None, :]).reshape(k, n).t()
    return pack_awq(q.reshape(k, n).to(torch.int32)), pack_awq(z.to(torch.int32)), s.half(), w_deq


@pytest.fixture(scope="module")
def tiny_awq(tmp_path_factory):
    """A tiny Qwen3 saved twice: AWQ-quantized projections, and the same weights dequantized."""
    pytest.importorskip("transformers")
    import json

    from safetensors.torch import load_file, save_file
    from test_gpu_fastpath import _save_tiny_qwen3

    dense = str(tmp_path_factory.mktemp("tiny_dense"))
    _save_tiny_qwen3(dense, 4, 2)
    tensors = load_file(f"{dense}/model.safetensors")
    quant, deq = {}, {}
    for name, t in tensors.items():
        proj = name.endswith(".weight") and any(p in name for p in ("_proj.weight",))
        if proj and ".layers." in name:
            qw, qz, sc, w_deq = _quantize_awq(t.cuda())
            base = name[: -len(".weight")]
            quant.update({f"{base}.qweight": qw.cpu(), f"{base}.qzeros": qz.cpu(), f"{base}.scales": sc.cpu()})
            deq[name] = w_deq.half().cpu()
        else:
            quant[name] = t.half()
            deq[name] = t.half()
    paths = {}
    for kind, tens in (("awq", quant), ("deq", deq)):
        path = str(tmp_path_factory.mktemp(f"tiny_{kind}"))
        save_file(tens, f"{path}/model.safetensors")
        cfg = json.load(open(f"{dense}/config.json"))
        cfg["torch_dtype"] = "float16"
        cfg.pop("dtype", None)
        if kind == "awq":
            cfg["quantization_config"] = {"quant_method": "awq", "bits": 4, "group_size": 128, "zero_point": True,
                                          "version": "gemm"}
        json.dump(cfg, open(f"{path}/config.json", "w"))
        paths[kind] = path
    return paths


def test_awq_checkpoint_matches_dequantized_model(tiny_awq):
    """Loader (q/k/v and gate/up fused along the output dimension), repack, GEMV / dequant paths
    and CUDA graphs: greedy outputs equal those of the dequantized dense model, within near-ties
    of the float32 reference."""
    from test_gpu_fastpath import GREEDY, VOCAB, assert_same_or_near_tie, fp32_reference, make, prompts

    ps = prompts(12, [5, 40, 300, 9, 17], vocab=VOCAB)
    awq_eng = make(tiny_awq["awq"], dtype="float16")
    assert sum(1 for m in awq_eng.model.modules() if type(m).__name__ == "W4Linear") == 4 * 2
    out = awq_eng.generate(ps, GREEDY)
    ref = make(tiny_awq["deq"], dtype="float16").generate(ps, GREEDY)
    hf = fp32_reference(tiny_awq["deq"])
    assert awq_eng.runner.num_graph_steps > 0
    for p, a, b in zip(ps, ref, out):
        assert_same_or_near_tie(hf, p, a.output_token_ids, b.output_token_ids, tol=0.3)


SHAPES = ((128, 24), (256, 520), (1024, 64), (4096, 1000))  # N % 16 != 0 exercises the row clamp
MMA_CONFIGS = ((1, 2, 1), (1, 4, 1), (2, 2, 1), (2, 4, 1), (4, 1, 1), (4, 2, 1), (4, 4, 1), (8, 1, 1), (8, 2, 1),
               (2, 2, 2), (4, 1, 2), (4, 2, 2), (8, 1, 2), (8, 2, 2))  # (warps_k, unroll, ntile)


def _ext():
    from tandem.layers import w4_cuda

    ext = w4_cuda.load()
    if ext is None:
        pytest.skip("W4A16 CUDA kernels did not build (no nvcc?)")
    return ext


def test_cuda_w4_gemv_matches_dequantized():
    """CUDA-core kernel (LOP3 dequant) against the dequantized matmul, every compiled config."""
    ext = _ext()
    for k, n in SHAPES:
        qweight, qzeros, scales, w_ref = random_awq(k, n, seed=3)
        packed, s, z = repack_awq(qweight, qzeros, scales, 128)
        for m in range(1, 5):
            x = torch.randn(m, k, device="cuda", dtype=torch.float16)
            ref = x.float() @ w_ref.t()
            for rows, warps in ((1, 4), (2, 2), (2, 4), (2, 8), (4, 2), (4, 4), (4, 8), (8, 2), (8, 4)):
                y = ext.w4_gemv(x, packed, s, z, rows, warps).float()
                assert torch.allclose(y, ref, atol=0.05, rtol=0.02), (k, n, m, rows, warps, (y - ref).abs().max())


def test_cuda_w4_mma_matches_dequantized():
    """Tensor-core kernel (weights as the A operand, batch tiles of 8, 1 or 2 row tiles per warp)
    against the dequantized matmul, M = 1..16 and every compiled config."""
    ext = _ext()
    for k, n in SHAPES:
        qweight, qzeros, scales, w_ref = random_awq(k, n, seed=4)
        packed, s, z = repack_awq(qweight, qzeros, scales, 128)
        for m in range(1, 17):
            x = torch.randn(m, k, device="cuda", dtype=torch.float16)
            ref = x.float() @ w_ref.t()
            for cfg in MMA_CONFIGS:
                y = ext.w4_mma(x, packed, s, z, *cfg, 0).float()
                assert torch.allclose(y, ref, atol=0.02, rtol=0.01), (k, n, m, cfg, (y - ref).abs().max())


def test_cuda_w4_mma_silu_mul():
    """SwiGLU epilogue (act = 1): rows interleaved by interleave_gate_up, y = silu(gate) * up."""
    ext = _ext()
    for k, n in ((128, 32), (1024, 512), (4096, 1024)):
        qweight, qzeros, scales, w_ref = random_awq(k, n, seed=6)
        packed, s, z = (interleave_gate_up(t) for t in repack_awq(qweight, qzeros, scales, 128))
        for m in range(1, 17):
            x = torch.randn(m, k, device="cuda", dtype=torch.float16)
            ref = _swiglu_ref(x, w_ref)
            for cfg in MMA_CONFIGS:
                y = ext.w4_mma(x, packed, s, z, *cfg, 1).float()
                assert y.shape == (m, n // 2)
                assert torch.allclose(y, ref, atol=0.02, rtol=0.01), (k, n, m, cfg, (y - ref).abs().max())
