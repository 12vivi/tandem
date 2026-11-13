"""GPU fast path (FlashInfer attention and ops, CUDA graphs) against the PyTorch reference path.

Skipped unless CUDA and flashinfer are available. Both paths run BF16 on the GPU and differ
only in kernel rounding, so the checks are: prefill logits close; greedy outputs agree on
models with well-separated logits; CUDA-graph decode agrees with eager decode, including
padded buckets, chunked prefill, prefix-cache sharing and preemption.

    pytest tests/test_gpu_fastpath.py -q
    TANDEM_TEST_MODEL=~/models/Qwen3-0.6B pytest tests/test_gpu_fastpath.py -q   # adds real weights
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
pytest.importorskip("flashinfer")

from conftest import prompts  # noqa: E402
from tandem.sampling_params import SamplingParams  # noqa: E402
from tandem.utils import cdiv  # noqa: E402

VOCAB = 512
GREEDY = SamplingParams(max_tokens=24, temperature=0.0, ignore_eos=True)


def _save_tiny_qwen3(path: str, num_heads: int, num_kv_heads: int) -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(vocab_size=VOCAB, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                      num_attention_heads=num_heads, num_key_value_heads=num_kv_heads, head_dim=128,
                      max_position_embeddings=2048, rms_norm_eps=1e-6, tie_word_embeddings=True)
    torch.manual_seed(1234)
    model = Qwen3ForCausalLM(cfg)
    with torch.no_grad():  # well-separated logits: no greedy near-ties
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(0.0, 0.1)
    model.save_pretrained(path)


@pytest.fixture(scope="module", params=[(4, 2), (8, 2)], ids=["gqa2", "gqa4"])
def tiny_path(request, tmp_path_factory) -> str:
    heads, kv_heads = request.param
    path = str(tmp_path_factory.mktemp(f"qwen3_gqa{heads // kv_heads}"))
    _save_tiny_qwen3(path, heads, kv_heads)
    return path


def make(path: str, **overrides):
    from tandem.config import EngineConfig
    from tandem.engine.llm_engine import LLMEngine

    kw = dict(model=path, device="cuda", dtype="bfloat16", block_size=16, num_kv_blocks=512,
              max_num_batched_tokens=256, max_model_len=1024, max_num_seqs=16, cuda_graph_max_bs=16)
    kw.update(overrides)
    return LLMEngine(EngineConfig(**kw), load_tokenizer=False)


def common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def fp32_reference(path: str):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).cuda().eval()


@torch.no_grad()
def assert_same_or_near_tie(hf, prompt: list[int], a: list[int], b: list[int], tol: float) -> None:
    """Two BF16 greedy outputs for one prompt agree, or first differ at a near-tie of the FP32
    model: both chosen tokens in its top 5 and their FP32 scores within ``tol``. Different
    kernels round differently, so either may pick either side of a near-tie; after the first
    difference the continuations legitimately part (vLLM's tests use the same logic)."""
    d = common_prefix(a, b)
    if d == min(len(a), len(b)):
        return
    logits = hf(torch.tensor([prompt + a[:d]], device="cuda")).logits[0, -1].float()
    gap = abs(float(logits[a[d]] - logits[b[d]]))
    top = logits.topk(5).indices.tolist()
    assert gap < tol and a[d] in top and b[d] in top, (d, gap, a[d], b[d], top)


@pytest.fixture(scope="module")
def tiny_fp32(tiny_path):
    return fp32_reference(tiny_path)


def test_backend_selection(tiny_path):
    eng = make(tiny_path)
    assert eng.backend_name == "flashinfer" and eng.runner.graph_runner is not None
    assert make(tiny_path, attention_backend="torch").backend_name == "torch"


@torch.inference_mode()
def test_prefill_logits_close(tiny_path, tiny_fp32):
    """The fast path is as close to FP32 as the BF16 reference path is."""
    from tandem.executor.forward_context import ForwardContext

    prompt = prompts(0, [37], vocab=VOCAB)[0]
    exact = tiny_fp32(torch.tensor([prompt], device="cuda")).logits[0].float()
    err = {}
    for name in ("torch", "flashinfer"):
        eng = make(tiny_path, attention_backend=name, cuda_graph=False)
        blocks = eng.allocator.allocate(cdiv(len(prompt), eng.config.block_size))
        runner = eng.runner
        positions, meta = runner.attn_backend.prepare([(blocks, 0, len(prompt))], eng.config.block_size, eng.device)
        ctx = ForwardContext(meta, eng.kv_cache, runner.attn_backend, eng.moe_executor, ops=runner.ops)
        hidden = eng.model(torch.tensor(prompt, device=eng.device), positions, ctx)
        err[name] = float((eng.model.compute_logits(hidden).float() - exact).abs().max())
    assert err["flashinfer"] <= 1.5 * err["torch"] + 0.02, err


def test_greedy_matches_reference_with_graphs(tiny_path, tiny_fp32):
    # 5 requests -> padded into the batch-8 bucket; a 300-token prompt is chunked (budget 256)
    ps = prompts(1, [5, 17, 40, 300, 9], vocab=VOCAB)
    ref = make(tiny_path, attention_backend="torch").generate(ps, GREEDY)
    eager_eng = make(tiny_path, cuda_graph=False)
    eager = eager_eng.generate(ps, GREEDY)
    graph_eng = make(tiny_path)
    graph = graph_eng.generate(ps, GREEDY)
    sync = make(tiny_path, overlap_schedule=False).generate(ps, GREEDY)  # no overlap
    assert graph_eng.runner.num_graph_steps > 0 and eager_eng.runner.num_graph_steps == 0
    for p, r, e, g, s in zip(ps, ref, eager, graph, sync):
        for a, b in ((r, e), (e, g), (s, g)):
            assert_same_or_near_tie(tiny_fp32, p, a.output_token_ids, b.output_token_ids, tol=0.3)


def test_prefix_sharing_and_preemption(tiny_path, tiny_fp32):
    base = prompts(2, [64], vocab=VOCAB)[0]
    followers = [base + p for p in prompts(3, [7, 21, 33], vocab=VOCAB)]
    plain = make(tiny_path, enable_prefix_caching=False).generate(followers, GREEDY)
    eng = make(tiny_path)
    eng.generate([base], GREEDY)
    shared = eng.generate(followers, GREEDY)  # several decode rows list the same cached blocks
    assert eng.radix.hit_rate > 0
    for p, a, b in zip(followers, plain, shared):
        assert_same_or_near_tie(tiny_fp32, p, a.output_token_ids, b.output_token_ids, tol=0.3)
    # 14 blocks x 16 tokens cannot hold six requests of 40 + 24 tokens at once
    ps = prompts(4, [40] * 6, vocab=VOCAB)
    tight = make(tiny_path, num_kv_blocks=14)
    outs = tight.generate(ps, GREEDY)
    loose = make(tiny_path).generate(ps, GREEDY)
    assert tight.scheduler.num_preemptions > 0
    for p, a, b in zip(ps, outs, loose):
        assert_same_or_near_tie(tiny_fp32, p, a.output_token_ids, b.output_token_ids, tol=0.3)


@pytest.mark.parametrize("graph", [True, False], ids=["graph", "eager"])
def test_steps_do_not_wait_for_queued_gpu_work(tiny_path, graph):
    """With overlap scheduling the host prepares step N+1 while the GPU runs step N and waits
    only for step N-1's sampled tokens. Any other host-device sync in a step (a blocking copy,
    .item(), ...) leaves the GPU idle once per step: FlashInfer's plan() made one for host
    kv_indices on a CUDA-graph wrapper (0.6B decode: 11.9% GPU idle, 2575 instead of 2323 us
    per step). Queue a long GPU sleep, run a step: the sleep must still run when it returns."""
    eng = make(tiny_path, cuda_graph=graph)
    params = SamplingParams(max_tokens=64, temperature=0.0, ignore_eos=True)
    stream = torch.cuda.current_stream()

    def step_returns_while_gpu_busy() -> bool:
        torch.cuda.synchronize()
        torch.cuda._sleep(200_000_000)  # ~0.1 s of GPU work queued ahead of the step
        eng.step()
        busy = not stream.query()
        torch.cuda.synchronize()
        return busy

    for p in prompts(5, [20, 33, 7], vocab=VOCAB):
        eng.add_request(prompt_token_ids=p, sampling_params=params)
    for _ in range(6):  # a prefill and decode steps; lazy initialisation may synchronise once
        eng.step()
    assert all(step_returns_while_gpu_busy() for _ in range(3)), "a decode step waited for the GPU"
    first, second = prompts(6, [40, 40], vocab=VOCAB)
    eng.add_request(prompt_token_ids=first, sampling_params=params)
    for _ in range(3):  # warm up the 40-token prefill shape (new GEMM / attention kernels)
        eng.step()
    eng.add_request(prompt_token_ids=second, sampling_params=params)
    assert step_returns_while_gpu_busy(), "a prefill step waited for the GPU"  # prefill_first: alone


@pytest.mark.parametrize("has_norm", [True, False], ids=["qk_norm", "no_norm"])
@torch.inference_mode()
def test_fused_prologue_matches_float32_math(has_norm):
    pytest.importorskip("triton")
    from tandem.layers.ops import _rotate_half
    from tandem.layers.rotary import RotaryEmbedding
    from tandem.layers.triton_ops import qkv_prologue

    torch.manual_seed(0)
    t, nh, nkv, hd, eps = 37, 8, 2, 128, 1e-6
    dev, bf16 = torch.device("cuda"), torch.bfloat16
    qkv = torch.randn(t, (nh + 2 * nkv) * hd, device=dev, dtype=bf16)
    pos = torch.randint(0, 2000, (t,), device=dev)
    slots = torch.randperm(64 * 16, device=dev)[:t]
    rope = RotaryEmbedding(hd, 4096, 1e6, None).to(dev)
    qw = (1 + 0.1 * torch.randn(hd, device=dev)).to(bf16)
    kw = (1 + 0.1 * torch.randn(hd, device=dev)).to(bf16)
    kc = torch.zeros(64 * 16, nkv, hd, device=dev, dtype=bf16)
    vc = torch.zeros_like(kc)
    q = qkv_prologue(qkv, pos, slots, kc, vc, rope.cos_sin_cache, nh, nkv, hd,
                     qw if has_norm else None, kw if has_norm else None, eps)

    x = qkv.float().view(t, nh + 2 * nkv, hd)
    qh, kh, vh = x[:, :nh], x[:, nh:nh + nkv], x[:, nh + nkv:]
    if has_norm:
        qh = qh * torch.rsqrt(qh.pow(2).mean(-1, keepdim=True) + eps) * qw.float()
        kh = kh * torch.rsqrt(kh.pow(2).mean(-1, keepdim=True) + eps) * kw.float()
    cos, sin = rope.cos_cached[pos].unsqueeze(1), rope.sin_cached[pos].unsqueeze(1)
    rot = lambda h: h * cos + _rotate_half(h) * sin  # noqa: E731
    torch.testing.assert_close(q.float(), rot(qh), atol=3e-2, rtol=1e-2)
    torch.testing.assert_close(kc[slots].float(), rot(kh), atol=3e-2, rtol=1e-2)
    torch.testing.assert_close(vc[slots].float(), vh, atol=0, rtol=0)


def test_greedy_with_fused_prologue(tiny_path, tiny_fp32):
    pytest.importorskip("triton")
    ps = prompts(5, [5, 17, 40, 300, 9], vocab=VOCAB)
    plain = make(tiny_path, fused_prologue=False).generate(ps, GREEDY)
    fused_eng = make(tiny_path, fused_prologue=True)
    fused = fused_eng.generate(ps, GREEDY)
    assert fused_eng.runner.ops.fused_prologue and fused_eng.runner.num_graph_steps > 0
    for p, a, b in zip(ps, plain, fused):
        assert_same_or_near_tie(tiny_fp32, p, a.output_token_ids, b.output_token_ids, tol=0.3)


@pytest.mark.parametrize("defer", [False, True], ids=["reduce_each", "defer"])
def test_small_batch_gemv_matches_linear(defer):
    pytest.importorskip("triton")
    from tandem.layers.triton_gemv import gemv

    torch.manual_seed(7)
    for n, k in ((1024, 2048), (4096, 1024), (6, 512)):  # odd N exercises the row mask
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
        for m in range(1, 9):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            ref = (x.float() @ w.float().t())
            for bn, bk, nw in ((1, 512, 1), (2, 256, 2), (4, 512, 4)):
                out = gemv(x, w, bn, bk, nw, 3, defer=defer).float()
                assert torch.allclose(out, ref, atol=0.02, rtol=0.01), (n, k, m, bn, bk, nw)


def test_gemv_prologue_matches_unfused():
    pytest.importorskip("triton")
    from tandem.layers.triton_gemv import gemv_prologue

    torch.manual_seed(3)
    n, k = 768, 512
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
    g = torch.rand(k, device="cuda", dtype=torch.bfloat16) + 0.5
    for m in (1, 2, 3, 4, 8):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        r = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        v = x.float() + r.float()
        normed = (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6) * g.float()).to(torch.bfloat16).float()
        y, r_new = gemv_prologue("add_rmsnorm", x, w, residual=r, norm_weight=g, eps=1e-6, block_n=2, block_k=256)
        assert torch.allclose(y.float(), normed @ w.float().t(), atol=0.03, rtol=0.02)
        assert torch.equal(r_new, v.to(torch.bfloat16))
        y = gemv_prologue("rmsnorm", x, w, norm_weight=g, eps=1e-6, block_n=4, block_k=128, num_warps=2)
        xn = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * g.float()).to(torch.bfloat16)
        assert torch.allclose(y.float(), xn.float() @ w.float().t(), atol=0.03, rtol=0.02)
        h = torch.randn(m, 2 * k, device="cuda", dtype=torch.bfloat16)
        act = (torch.nn.functional.silu(h[:, :k].float()) * h[:, k:].float()).to(torch.bfloat16).float()
        y = gemv_prologue("silu_mul", h, w, block_n=1, block_k=512)
        assert torch.allclose(y.float(), act @ w.float().t(), atol=0.03, rtol=0.02)


@pytest.mark.parametrize("force", ["1", "2"], ids=["gemv", "gemv_fused"])
def test_greedy_with_small_batch_gemv(tiny_path, tiny_fp32, monkeypatch, force):
    """Every decoder projection through the Triton GEMV (forced; "2" also folds the norms and
    SiLU-and-mul into the GEMV input): same greedy outputs as cuBLAS, within near-ties, in
    CUDA-graph decode (batch buckets 1-8) and eager prefill."""
    pytest.importorskip("triton")
    ps = prompts(6, [5, 17, 40, 300, 9], vocab=VOCAB)
    plain_eng = make(tiny_path, small_batch_gemv=False)
    plain = plain_eng.generate(ps, GREEDY)
    monkeypatch.setenv("TANDEM_GEMV_FORCE", force)
    gemv_eng = make(tiny_path)
    out = gemv_eng.generate(ps, GREEDY)
    assert plain_eng.num_gemv_layers == 0 and gemv_eng.num_gemv_layers == 4 * 2  # 4 projections x 2 layers
    assert gemv_eng.num_fused_layers == (2 if force == "2" else 0)
    assert gemv_eng.runner.num_graph_steps > 0
    for p, a, b in zip(ps, plain, out):
        assert_same_or_near_tie(tiny_fp32, p, a.output_token_ids, b.output_token_ids, tol=0.3)


@pytest.mark.parametrize("gqa", [2, 4])
def test_triton_decode_attention_matches_reference(gqa):
    """Split-KV decode attention against float32 math over the same paged cache: shuffled pages,
    contexts that end mid-page, empty splits (more splits than pages)."""
    pytest.importorskip("triton")
    from tandem.layers.triton_decode_attn import decode_attention

    torch.manual_seed(11)
    hkv, d, page = 4, 128, 16
    hq = hkv * gqa
    num_pages = 64
    k_cache = torch.randn(num_pages * page, hkv, d, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn(num_pages * page, hkv, d, device="cuda", dtype=torch.bfloat16)
    for lens in ([1], [17, 300, 33], [5, 16, 64, 129, 200, 7, 31, 90]):
        bsz = len(lens)
        perm = torch.randperm(num_pages).tolist()
        indptr, indices, last, pos = [0], [], [], 0
        for n in lens:
            npg = (n + page - 1) // page
            indices += perm[pos:pos + npg]
            pos += npg
            indptr.append(indptr[-1] + npg)
            last.append(n - (npg - 1) * page)
        q = torch.randn(bsz, hq, d, device="cuda", dtype=torch.bfloat16)
        args = (torch.tensor(indptr, dtype=torch.int32, device="cuda"),
                torch.tensor(indices, dtype=torch.int32, device="cuda"),
                torch.tensor(last, dtype=torch.int32, device="cuda"))
        ref = torch.empty(bsz, hq, d, device="cuda")
        for b, n in enumerate(lens):
            slots = torch.cat([torch.arange(p * page, p * page + page) for p in indices[indptr[b]:indptr[b + 1]]])[:n]
            k = k_cache[slots.cuda()].float().repeat_interleave(gqa, dim=1)  # [n, hq, d]
            v = v_cache[slots.cuda()].float().repeat_interleave(gqa, dim=1)
            w = torch.softmax(torch.einsum("hd,nhd->hn", q[b].float(), k) * d ** -0.5, dim=-1)
            ref[b] = torch.einsum("hn,nhd->hd", w, v)
        for nsplit, ppi, nw in ((1, 1, 2), (4, 2, 2), (16, 1, 1), (32, 4, 4)):
            for fused_merge in (True, False):
                for _ in range(2):  # twice: the in-kernel merge must leave its counters at zero
                    out = decode_attention(q, k_cache, v_cache, *args, page, d ** -0.5, nsplit=nsplit,
                                           pages_per_iter=ppi, num_warps=nw, fused_merge=fused_merge)
                    assert torch.allclose(out.float(), ref, atol=2e-2, rtol=2e-2), (
                        lens, nsplit, ppi, fused_merge, (out.float() - ref).abs().max())


def test_greedy_with_triton_decode_attention(tiny_path, tiny_fp32):
    """Decode attention at batch 1-2 through the Triton kernel (GQA 2) vs FlashInfer: same greedy
    outputs within near-ties, single request and a pair (CUDA-graph buckets 1 and 2)."""
    pytest.importorskip("triton")
    for ps in (prompts(8, [23], vocab=VOCAB), prompts(9, [11, 70], vocab=VOCAB)):
        base = make(tiny_path, small_batch_attention=False).generate(ps, GREEDY)
        eng = make(tiny_path)
        out = eng.generate(ps, GREEDY)
        assert eng.runner.attn_backend.triton_decode_nsplit  # on by default (used only for GQA 2)
        for p, a, b in zip(ps, base, out):
            assert_same_or_near_tie(tiny_fp32, p, a.output_token_ids, b.output_token_ids, tol=0.3)


MODEL = os.environ.get("TANDEM_TEST_MODEL")


@pytest.mark.skipif(not MODEL, reason="set TANDEM_TEST_MODEL to a local HF model directory")
def test_real_model_fast_path_matches_reference():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    texts = ["The capital of France is", "def fibonacci(n):", "请用一句话介绍你自己。"]
    ids = [tok.encode(t) for t in texts]
    sp = SamplingParams(max_tokens=32, temperature=0.0, ignore_eos=True)
    kw = dict(num_kv_blocks=None, kv_cache_memory_gb=1.0, max_model_len=2048)
    ref = make(MODEL, attention_backend="torch", **kw).generate(ids, sp)
    fast = make(MODEL, **kw).generate(ids, sp)
    hf = fp32_reference(MODEL)
    for p, r, f in zip(ids, ref, fast):
        assert_same_or_near_tie(hf, p, r.output_token_ids, f.output_token_ids, tol=1.0)
