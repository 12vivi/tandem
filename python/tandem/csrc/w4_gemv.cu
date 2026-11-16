// W4A16 kernels (design doc P2 / A2): y[M, N] = x[M, K] @ dequant(W)[N, K]^T.
//
// Weight layout (tandem/layers/awq.py pack_rows): packed int32 [N][K/128][4 j][4 cc]. For output
// row n and quant group G (128 k), int32 (j, cc) holds the 8 consecutive weights
//     k = 128 G + 32 cc + 8 j + 2 q + e   (q = 0..3, e = 0..1)   in nibble q + 4 e,
// and scales / zeros are float16 [N][K/128], w = (q - z) * s. The LOP3 trick (FasterTransformer,
// Marlin) turns nibbles (q, q + 4) into half2(1024 + w_2q, 1024 + w_2q+1) without int-to-float
// conversions, so one int32 dequantizes to the 4 half2 of its 8 consecutive k. A lane loads the
// 16 bytes (j, cc = 0..3) of one row and group: one load covers a quant group, and for each cc the
// 4 lanes j = 0..3 of a row together hold 32 consecutive k (one 64-byte segment of x).
//
//   w4_gemv  M <= 4, CUDA cores: the 8 lane quads of a warp take quant groups quad, quad + 8, ...
//            of ROWS rows; float16 products over 8 k, float32 accumulation, warp-shuffle reduction.
//   w4_mma   M <= 16, tensor cores: mma.m16n8k16 with float32 accumulation and the weights as the
//            A operand ("swap AB"): one MMA covers 16 output rows x 16 k and the batch is its
//            n = 8 side, so M <= 8 needs one MMA per 16 x 16 weight tile instead of a batch padded
//            to 16. The MMA may sum its 16 k in any order as long as A and B use the same one, so
//            lane (g, c) feeds the 8 consecutive k of int32 (c, cc) to both: logical k 2c, 2c+1
//            and 2c+8, 2c+9 of step s are physical 8c + 4s + {0, 1} and 8c + 4s + {2, 3}. The
//            weights go from global memory straight into A fragments (no shared memory, no
//            ldmatrix), x straight into B fragments (one 16-byte load per lane and 32 k). A CTA
//            computes NT x 16 output rows; its WARPS_K warps split K by quant groups (UNROLL groups
//            in flight per lane) and add up their partial sums in shared memory. For gate_up it
//            can also apply SwiGLU in its epilogue (act = 1, see w4_mma_kernel).
//   w4_gemm  M > 16 (prefill, decode batches over 16), tensor cores: x is the A operand (16 tokens
//            per MMA), the weights are dequantized in registers into B fragments with the same k
//            permutation, x and weight tiles go through a cp.async ring in shared memory, optional
//            split-K across CTAs, and the same SwiGLU epilogue (see w4_gemm_kernel). Replaces
//            "dequantize to float16 + cuBLAS", which wrote and re-read the whole float16 weights.

#include <algorithm>

#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

namespace {

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {  // i is a constant after unrolling
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// Weights are read exactly once: do not allocate them in L1, which then keeps x (re-read by every CTA)
__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
    uint4 v;
    asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ uint32_t h2u(half2 h) { return *reinterpret_cast<uint32_t*>(&h); }
__device__ __forceinline__ half2 u2h(uint32_t u) { return *reinterpret_cast<half2*>(&u); }

__device__ __forceinline__ void dequant8(uint32_t i4s, half2 (&h)[4]) {
    uint32_t* hh = reinterpret_cast<uint32_t*>(h);
    constexpr uint32_t kLut = (0xf0 & 0xcc) | 0xaa;  // (a & b) | c
    constexpr uint32_t kBottom = 0x000f000f, kTop = 0x00f000f0, kMagic = 0x64006400;  // half2(1024, 1024)
    const uint32_t top = i4s >> 8;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(hh[0]) : "r"(i4s), "n"(kBottom), "n"(kMagic), "n"(kLut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(hh[1]) : "r"(i4s), "n"(kTop), "n"(kMagic), "n"(kLut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(hh[2]) : "r"(top), "n"(kBottom), "n"(kMagic), "n"(kLut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(hh[3]) : "r"(top), "n"(kTop), "n"(kMagic), "n"(kLut));
}

struct GroupParams {
    half2 zz, zt, ss;  // 1024 + z, -(64 + z), scale
};

__device__ __forceinline__ GroupParams group_params(const half* scales, const half* zeros, size_t idx) {
    const half z = __ldg(zeros + idx);
    GroupParams p;
    p.ss = __half2half2(__ldg(scales + idx));
    p.zz = __half2half2(__hadd(z, __float2half(1024.0f)));
    p.zt = __half2half2(__hneg(__hadd(z, __float2half(64.0f))));
    return p;
}

// int32 of 8 consecutive k -> h[q] = (w_2q, w_2q+1): (raw - (1024 + z)) * s for the low nibbles,
// (raw / 16 - (64 + z)) * s for the high ones (raw = 1024 + 16 q); all steps exact before the scale
__device__ __forceinline__ void dequant_group8(uint32_t packed, const GroupParams& p, half2 (&h)[4]) {
    const half2 sixteenth = __float2half2_rn(1.0f / 16.0f);
    dequant8(packed, h);
    h[0] = __hmul2(__hsub2(h[0], p.zz), p.ss);
    h[1] = __hmul2(__hfma2(h[1], sixteenth, p.zt), p.ss);
    h[2] = __hmul2(__hsub2(h[2], p.zz), p.ss);
    h[3] = __hmul2(__hfma2(h[3], sixteenth, p.zt), p.ss);
}

// ------------------------------------------------------------------------------- CUDA cores
// Lane (quad, j) = (lane / 4, lane % 4) takes the 16 bytes (j, cc = 0..3) of quant groups
// quad, quad + 8, ... of each of ROWS rows: a warp load reads 8 groups = 512 contiguous bytes of
// a row, and each x load reads 8 fully used 64-byte segments.
template <int M, int ROWS, int WARPS>
__global__ void __launch_bounds__(WARPS * 32) w4_gemv_kernel(
    const half* __restrict__ x, const uint4* __restrict__ w, const half* __restrict__ scales,
    const half* __restrict__ zeros, half* __restrict__ y, int N, int K) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int quad = lane >> 2, j = lane & 3;
    const int row0 = (blockIdx.x * WARPS + warp) * ROWS;
    if (row0 >= N) return;
    const int groups = K >> 7;
    float acc[M][ROWS];
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[m][r] = 0.0f;

    for (int G = quad; G < groups; G += 8) {
        uint4 wv[ROWS];
        GroupParams gp[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = min(row0 + r, N - 1);  // rows past N recompute the last row; not stored
            const size_t gi = static_cast<size_t>(row) * groups + G;
            wv[r] = ld_stream(w + gi * 4 + j);
            gp[r] = group_params(scales, zeros, gi);
        }
#pragma unroll
        for (int cc = 0; cc < 4; ++cc) {
            uint4 xv[M];  // x at k = 128 G + 32 cc + 8 j .. + 7
#pragma unroll
            for (int m = 0; m < M; ++m)
                xv[m] = __ldg(reinterpret_cast<const uint4*>(
                    x + static_cast<size_t>(m) * K + (G << 7) + (cc << 5) + (j << 3)));
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                half2 h[4];
                dequant_group8(word(wv[r], cc), gp[r], h);
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    half2 p = __hmul2(h[0], u2h(xv[m].x));
                    p = __hfma2(h[1], u2h(xv[m].y), p);
                    p = __hfma2(h[2], u2h(xv[m].z), p);
                    p = __hfma2(h[3], u2h(xv[m].w), p);
                    const float2 f = __half22float2(p);
                    acc[m][r] += f.x + f.y;
                }
            }
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            float v = acc[m][r];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
            if (lane == 0 && row0 + r < N) y[static_cast<size_t>(m) * N + row0 + r] = __float2half_rn(v);
        }
}

// ------------------------------------------------------------------------------ tensor cores
__device__ __forceinline__ float silu(float v) { return v / (1.0f + __expf(-v)); }

__device__ __forceinline__ void mma16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// MT batch tiles of 8 (M <= 8 MT) and NT tiles of 16 output rows per warp (x fragments reused
// NT times). Fragments of lane (g, c) = (lane / 4, lane % 4) for row tile t and batch tile b:
//   A (weights, 16 x 16): a0 / a2 = row n0 + 16 t + g, a1 / a3 = row n0 + 16 t + g + 8
//   B (x^T, 16 x 8): column g = batch row 8 b + g
//   D (16 x 8, float32): d0, d1 = output row n0 + 16 t + g, batch rows 8 b + 2c, 8 b + 2c + 1;
//                        d2, d3 = output row n0 + 16 t + g + 8
// act = 1 (SwiGLU, for gate_up): the rows of every 16-row tile T are gate rows 8T .. 8T+7 then
// the matching up rows (tandem/layers/awq.py interleave_gate_up), so a lane's d0 / d2 hold gate
// and up of the same feature 8T + g, and the kernel writes y[M, N/2] = silu(gate) * up directly
// from the float32 accumulators: no activation kernel, and gate_up never goes to memory.
template <int WARPS_K, int UNROLL, int MT, int NT>
__global__ void __launch_bounds__(WARPS_K * 32) w4_mma_kernel(
    const half* __restrict__ x, const uint4* __restrict__ w, const half* __restrict__ scales,
    const half* __restrict__ zeros, half* __restrict__ y, int M, int N, int K, int act) {
    __shared__ float red[WARPS_K][NT * MT * 4][32];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, c = lane & 3;
    const int n0 = blockIdx.x * 16 * NT;
    const int groups = K >> 7;
    size_t base[NT][2];  // rows past N recompute the last row; not stored
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int h = 0; h < 2; ++h) base[t][h] = static_cast<size_t>(min(n0 + 16 * t + 8 * h + g, N - 1)) * groups;
    const half* xb[MT];
    bool has_x[MT];
#pragma unroll
    for (int b = 0; b < MT; ++b) {
        has_x[b] = 8 * b + g < M;  // batch rows past M are zero columns of B
        xb[b] = x + static_cast<size_t>(has_x[b] ? 8 * b + g : 0) * K + 8 * c;
    }
    float d[NT][MT][4];
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int b = 0; b < MT; ++b)
#pragma unroll
            for (int i = 0; i < 4; ++i) d[t][b][i] = 0.0f;

    for (int G0 = warp; G0 < groups; G0 += WARPS_K * UNROLL) {
        uint4 wv[UNROLL][NT][2];
        GroupParams gp[UNROLL][NT][2];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {  // UNROLL quant groups in flight per lane
            const int G = min(G0 + u * WARPS_K, groups - 1);
#pragma unroll
            for (int t = 0; t < NT; ++t)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    wv[u][t][h] = ld_stream(w + (base[t][h] + G) * 4 + c);
                    gp[u][t][h] = group_params(scales, zeros, base[t][h] + G);
                }
        }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int G = G0 + u * WARPS_K;
            if (G >= groups) break;
#pragma unroll
            for (int cc = 0; cc < 4; ++cc) {
                uint4 xq[MT];  // x at k = 128 G + 32 cc + 8 c .. + 7
#pragma unroll
                for (int b = 0; b < MT; ++b)
                    xq[b] = has_x[b] ? __ldg(reinterpret_cast<const uint4*>(xb[b] + (G << 7) + (cc << 5)))
                                     : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    half2 hl[4], hh[4];
                    dequant_group8(word(wv[u][t][0], cc), gp[u][t][0], hl);
                    dequant_group8(word(wv[u][t][1], cc), gp[u][t][1], hh);
#pragma unroll
                    for (int s = 0; s < 2; ++s)  // k16 step s: physical k 8c + 4s .. 8c + 4s + 3
#pragma unroll
                        for (int b = 0; b < MT; ++b)
                            mma16816(d[t][b], h2u(hl[2 * s]), h2u(hh[2 * s]), h2u(hl[2 * s + 1]), h2u(hh[2 * s + 1]),
                                     word(xq[b], 2 * s), word(xq[b], 2 * s + 1));
                }
            }
        }
    }
    if (WARPS_K > 1) {
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
            for (int b = 0; b < MT; ++b)
#pragma unroll
                for (int i = 0; i < 4; ++i) red[warp][(t * MT + b) * 4 + i][lane] = d[t][b][i];
        __syncthreads();
        if (warp != 0) return;
#pragma unroll
        for (int v = 1; v < WARPS_K; ++v)
#pragma unroll
            for (int t = 0; t < NT; ++t)
#pragma unroll
                for (int b = 0; b < MT; ++b)
#pragma unroll
                    for (int i = 0; i < 4; ++i) d[t][b][i] += red[v][(t * MT + b) * 4 + i][lane];
    }
    if (act) {  // N % 16 == 0 (checked on the host)
        const int half_n = N >> 1;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const int f = ((n0 + 16 * t) >> 1) + g;  // feature 8 T + g of tile T
            if (f >= half_n) continue;
#pragma unroll
            for (int b = 0; b < MT; ++b) {
                const int m = 8 * b + 2 * c;
                if (m < M) y[static_cast<size_t>(m) * half_n + f] = __float2half_rn(silu(d[t][b][0]) * d[t][b][2]);
                if (m + 1 < M)
                    y[static_cast<size_t>(m + 1) * half_n + f] = __float2half_rn(silu(d[t][b][1]) * d[t][b][3]);
            }
        }
        return;
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const int na = n0 + 16 * t + g, nb = na + 8;
#pragma unroll
        for (int b = 0; b < MT; ++b) {
            const int m = 8 * b + 2 * c;
            if (m < M) {
                if (na < N) y[static_cast<size_t>(m) * N + na] = __float2half_rn(d[t][b][0]);
                if (nb < N) y[static_cast<size_t>(m) * N + nb] = __float2half_rn(d[t][b][2]);
            }
            if (m + 1 < M) {
                if (na < N) y[static_cast<size_t>(m + 1) * N + na] = __float2half_rn(d[t][b][1]);
                if (nb < N) y[static_cast<size_t>(m + 1) * N + nb] = __float2half_rn(d[t][b][3]);
            }
        }
    }
}

// ---------------------------------------------------------------- tensor cores, M > 16 (prefill)
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))), "l"(src));
}
// An opaque copy: the compiler cannot recompute v from its inputs (rematerialization), so values
// computed once before a loop stay in registers instead of being recomputed every iteration
__device__ __forceinline__ int opaque(int v) {
    asm volatile("" : "+r"(v));
    return v;
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

// x is the A operand here (16 tokens per MMA) and the weights the B operand (8 output rows), with
// w4_mma's k permutation: lane (g, c) feeds the 8 consecutive k of int32 (c, cc) to both, so
// logical k 2c, 2c+1 / 2c+8, 2c+9 of step s are physical 32 cc + 8c + 4s + {0, 1} / {2, 3}.
// A CTA computes BM = 16 MT tokens x BN = WARPS_N x 8 NT output rows; warp (wn, wk) takes rows
// wn x 8 NT .. + 8 NT - 1 of every token and quant groups wk, wk + WARPS_K, ... (partial sums added
// in shared memory at the end). The x and weight tiles of WARPS_K quant groups per step go through
// a STAGES-deep cp.async ring in shared memory; x rows are XOR swizzled so that the 16-byte
// fragment loads of rows g and g + 1 use different banks. Fragments of lane (g, c):
//   A (x, 16 x 16): token rows 16 i + g (a0, a2) and + 8 (a1, a3): one 16-byte load each per cc
//   B (weights, 16 x 8): row 8 t + g, dequantized from its 16 bytes (j = c, cc = 0..3) per group
//   D: d0, d1 = token 16 i + g, output rows 8 t + 2c, 8 t + 2c + 1; d2, d3 = token 16 i + g + 8
// act = 1 (gate_up, rows in interleave_gate_up order): row tiles 2T and 2T + 1 are the gate and up
// rows of features 8T .. 8T + 7, so a lane holds gate and up of features 8T + 2c, 8T + 2c + 1 and
// writes y[M, N/2] = silu(gate) * up from the float32 accumulators.
// Split-K: CTAs 0 .. dp_tiles - 1 take whole tiles; every later tile is shared by `splits` CTAs,
// each taking a range of quant groups (dp_tiles = 0: all tiles split, for small M; dp_tiles = the
// full waves: only the last, partial wave is split so that it fills the GPU too, see gemm_launch).
// Each warp of a split stores its float32 partial sums to ws (fragment order, coalesced) and bumps
// its counter in locks; the warp that arrives last adds the others' sums (ld.global.cg: L2, never
// a stale L1 line from an earlier call), writes y and resets the counter for the next call.
template <int MT, int NT, int WARPS_N, int WARPS_K, int STAGES>
__global__ void __launch_bounds__(WARPS_N * WARPS_K * 32) w4_gemm_kernel(
    const half* __restrict__ x, const uint4* __restrict__ w, const half* __restrict__ scales,
    const half* __restrict__ zeros, half* __restrict__ y, float* __restrict__ ws, int* __restrict__ locks,
    int M, int N, int K, int dp_tiles, int splits, int act) {
    static_assert(NT % 2 == 0, "the SwiGLU epilogue pairs row tiles");
    constexpr int BM = 16 * MT, WN = 8 * NT, BN = WARPS_N * WN, THREADS = WARPS_N * WARPS_K * 32;
    constexpr int XC = BM * 16, WC = BN * 4, GC = XC + WC;  // 16-byte chunks per quant group: x, weights
    extern __shared__ uint4 smem[];  // ring: STAGES steps x WARPS_K quant groups x GC chunks
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wn = warp % WARPS_N, wk = warp / WARPS_N;
    const int g = lane >> 2, c = lane & 3;
    const int groups = K >> 7, steps = (groups + WARPS_K - 1) / WARPS_K;
    const bool whole = static_cast<int>(blockIdx.x) < dp_tiles;
    const int sk = blockIdx.x - dp_tiles;  // index among the split CTAs
    const int nsplit = whole ? 1 : splits;
    const int tile = whole ? blockIdx.x : dp_tiles + sk / splits, split = whole ? 0 : sk % splits;
    const int step0 = split * steps / nsplit, nsteps = (split + 1) * steps / nsplit - step0;
    // tile order: bands of 8 row blocks, row block fastest, so up to 8 CTAs read a weight tile at
    // about the same time (one DRAM read, then L2 hits) and the band's x stays in L2
    const int mb = (M + BM - 1) / BM, nb = (N + BN - 1) / BN;
    const int band = tile / (8 * nb), band_rows = min(mb - band * 8, 8), r_in = tile - band * 8 * nb;
    const int m0 = (band * 8 + r_in % band_rows) * BM, n0 = (r_in / band_rows) * BN;

    // this thread's cp.async sources, as 32-bit offsets of quant group 0 (computed once; rows past
    // M / N load the last row: computed, never stored). x chunk i = it * THREADS + tid is row
    // i / 16, chunk i % 16; THREADS / 16 is even, so the row parity (the swizzle) is fixed per thread.
    constexpr int XIT = (XC + THREADS - 1) / THREADS, WIT = (WC + THREADS - 1) / THREADS;
    int xsrc[XIT], wsrc[WIT];
#pragma unroll
    for (int it = 0; it < XIT; ++it) {
        const int i = it * THREADS + threadIdx.x;
        xsrc[it] = opaque(min(m0 + (i >> 4), M - 1) * K + ((i & 15) << 3));
    }
#pragma unroll
    for (int it = 0; it < WIT; ++it) {
        const int i = it * THREADS + threadIdx.x;
        wsrc[it] = opaque(min(n0 + (i >> 2), N - 1) * groups * 4 + (i & 3));
    }
    const int xdst = opaque(((threadIdx.x >> 4) << 4) + ((threadIdx.x & 15) ^ (((threadIdx.x >> 4) & 1) << 2)));
    auto load = [&](int ls) {
        const int step = step0 + ls;
        uint4* st = smem + (ls % STAGES) * (WARPS_K * GC);
#pragma unroll
        for (int u = 0; u < WARPS_K; ++u) {
            const int G = step * WARPS_K + u;
            if (G >= groups) break;
            uint4* sx = st + u * GC;
#pragma unroll
            for (int it = 0; it < XIT; ++it)
                if (XC % THREADS == 0 || it * THREADS + threadIdx.x < XC)
                    cp_async16(sx + xdst + it * THREADS, x + (xsrc[it] + (G << 7)));
#pragma unroll
            for (int it = 0; it < WIT; ++it)
                if (WC % THREADS == 0 || it * THREADS + threadIdx.x < WC)
                    cp_async16(sx + XC + it * THREADS + threadIdx.x, w + (wsrc[it] + (G << 2)));
        }
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < nsteps) load(s);
        cp_async_commit();
    }
    int prow[NT];  // this lane's weight rows, for the group parameters
#pragma unroll
    for (int t = 0; t < NT; ++t) prow[t] = opaque(min(n0 + wn * WN + 8 * t + g, N - 1) * groups);
    const int swz = (g & 1) << 2;  // token rows 16 i + 8 h + g: (row & 1) == (g & 1)
    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][t][e] = 0.0f;

    for (int ls = 0; ls < nsteps; ++ls) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();  // this step's tiles are visible, and every warp is done with the slot the next load refills
        if (ls + STAGES - 1 < nsteps) load(ls + STAGES - 1);
        cp_async_commit();
        const int G = (step0 + ls) * WARPS_K + wk;
        if (G >= groups) continue;
        GroupParams gp[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t) gp[t] = group_params(scales, zeros, prow[t] + G);
        const uint4* sx = smem + (ls % STAGES) * (WARPS_K * GC) + wk * GC;
        uint4 wv[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t) wv[t] = sx[XC + (wn * WN + 8 * t + g) * 4 + c];
#pragma unroll
        for (int cc = 0; cc < 4; ++cc) {
            uint4 xa[MT][2];  // x at k = 128 G + 32 cc + 8 c .. + 7 of token rows 16 i + g (+ 8)
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h) xa[i][h] = sx[((16 * i + 8 * h + g) << 4) + ((4 * cc + c) ^ swz)];
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                half2 hb[4];
                dequant_group8(word(wv[t], cc), gp[t], hb);
#pragma unroll
                for (int s = 0; s < 2; ++s)  // k16 step s: physical k 8c + 4s .. 8c + 4s + 3 of this cc
#pragma unroll
                    for (int i = 0; i < MT; ++i)
                        mma16816(acc[i][t], word(xa[i][0], 2 * s), word(xa[i][1], 2 * s), word(xa[i][0], 2 * s + 1),
                                 word(xa[i][1], 2 * s + 1), h2u(hb[2 * s]), h2u(hb[2 * s + 1]));
            }
        }
    }
    if constexpr (WARPS_K > 1) {  // add up the K slices of warps wk > 0 in warp wk = 0
        constexpr int V = MT * NT * 4;
        cp_async_wait<0>();
        __syncthreads();  // the ring is free
        float* red = reinterpret_cast<float*>(smem);  // [WARPS_K - 1][WARPS_N][V][32]
        if (wk > 0) {
            float* dst = red + ((wk - 1) * WARPS_N + wn) * V * 32 + lane;
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int t = 0; t < NT; ++t)
#pragma unroll
                    for (int e = 0; e < 4; ++e) dst[((i * NT + t) * 4 + e) * 32] = acc[i][t][e];
        }
        __syncthreads();
        if (wk > 0) return;
#pragma unroll
        for (int v = 1; v < WARPS_K; ++v) {
            const float* src = red + ((v - 1) * WARPS_N + wn) * V * 32 + lane;
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int t = 0; t < NT; ++t)
#pragma unroll
                    for (int e = 0; e < 4; ++e) acc[i][t][e] += src[((i * NT + t) * 4 + e) * 32];
        }
    }
    if (nsplit > 1) {  // warps wk = 0 only from here on
        constexpr int V = MT * NT * 4;
        const int st = tile - dp_tiles;  // index among the split tiles
        float* mine = ws + (static_cast<size_t>(sk) * WARPS_N + wn) * V * 32 + lane;
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int t = 0; t < NT; ++t)
#pragma unroll
                for (int e = 0; e < 4; ++e) mine[((i * NT + t) * 4 + e) * 32] = acc[i][t][e];
        __threadfence();
        __syncwarp();
        int last = 0;
        if (lane == 0) last = atomicAdd(locks + st * WARPS_N + wn, 1) == splits - 1;
        if (!__shfl_sync(0xffffffffu, last, 0)) return;
        __threadfence();
        for (int v = 0; v < splits; ++v) {
            if (v == split) continue;
            const float* other = ws + ((static_cast<size_t>(st) * splits + v) * WARPS_N + wn) * V * 32 + lane;
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int t = 0; t < NT; ++t)
#pragma unroll
                    for (int e = 0; e < 4; ++e) acc[i][t][e] += __ldcg(other + ((i * NT + t) * 4 + e) * 32);
        }
        if (lane == 0) locks[st * WARPS_N + wn] = 0;  // ready for the next call
    }
    const int nw = n0 + wn * WN;
    if (act) {  // N % 16 == 0 (checked on the host)
        const int half_n = N >> 1;
#pragma unroll
        for (int t = 0; t < NT; t += 2) {
            const int f = ((nw + 8 * t) >> 1) + 2 * c;  // gate in row tile t, up in t + 1
            if (f >= half_n) continue;
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int m = m0 + 16 * i + 8 * h + g;
                    if (m < M)
                        *reinterpret_cast<half2*>(y + static_cast<size_t>(m) * half_n + f) = __floats2half2_rn(
                            silu(acc[i][t][2 * h]) * acc[i][t + 1][2 * h],
                            silu(acc[i][t][2 * h + 1]) * acc[i][t + 1][2 * h + 1]);
                }
        }
        return;
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const int n = nw + 8 * t + 2 * c;  // N % 8 == 0: n < N implies n + 1 < N
        if (n >= N) continue;
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int m = m0 + 16 * i + 8 * h + g;
                if (m < M)
                    *reinterpret_cast<half2*>(y + static_cast<size_t>(m) * N + n) =
                        __floats2half2_rn(acc[i][t][2 * h], acc[i][t][2 * h + 1]);
            }
    }
}

template <int MT, int NT, int WN, int WK, int ST>
void gemm_launch(const half* xp, const uint4* wp, const half* sp, const half* zp, half* yp, torch::Tensor& ws,
                 torch::Tensor& locks, int m, int n, int k, int splits, int act, cudaStream_t stream) {
    constexpr int BM = 16 * MT, BN = WN * 8 * NT;
    constexpr int ring = ST * WK * (BM * 16 + BN * 4) * 16, red = (WK - 1) * WN * MT * NT * 4 * 32 * 4;
    constexpr int bytes = ring > red ? ring : red;
    auto kern = w4_gemm_kernel<MT, NT, WN, WK, ST>;
    static int per_sm = 0;  // resident CTAs per SM; set before the first launch (outside graph capture)
    if (per_sm == 0) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, WN * WK * 32, bytes));
        per_sm = per_sm > 0 ? per_sm : 1;
    }
    const int tiles = ((m + BM - 1) / BM) * ((n + BN - 1) / BN), steps = (k / 128 + WK - 1) / WK;
    const int64_t room = std::min<int64_t>(ws.numel() / (BM * BN), locks.numel() / WN);  // split tiles x splits
    int dp = tiles, s = 1;
    if (splits == 0) {  // whole tiles for the full waves; split the last, partial wave along K
        const int slots = per_sm * at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
        const int rem = tiles % slots;
        double best = 1.0;  // s = 1: the partial wave of whole tiles, in units of one tile's time
        for (int c = 2; c <= 8 && c <= steps && rem > 0; ++c) {
            if (static_cast<int64_t>(rem) * c > room) break;
            const double t = static_cast<double>((rem * c + slots - 1) / slots) / c + 0.01 * c;  // + fix-up cost
            if (t < best) best = t, s = c;
        }
        if (s > 1) dp = tiles - rem;
    } else if (splits > 1) {
        TORCH_CHECK(static_cast<int64_t>(tiles) * splits <= room, "w4_gemm: split-K workspace too small for ",
                    tiles, " tiles x ", splits, " splits");
        dp = 0, s = splits;
    }
    kern<<<dp + (tiles - dp) * s, WN * WK * 32, bytes, stream>>>(xp, wp, sp, zp, yp, ws.data_ptr<float>(),
                                                                 locks.data_ptr<int32_t>(), m, n, k, dp, s, act);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& s, const torch::Tensor& z) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == at::kHalf && x.is_contiguous(),
                "x: contiguous 2-D float16 CUDA tensor");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "x: 16-byte aligned");
    TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kInt && w.is_contiguous() &&
                reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0, "packed: contiguous 16-byte aligned int32");
    TORCH_CHECK(s.scalar_type() == at::kHalf && z.scalar_type() == at::kHalf && s.is_contiguous() && z.is_contiguous(),
                "scales / zeros: contiguous float16");
    TORCH_CHECK(x.size(1) % 128 == 0 && w.dim() == 2 && w.size(1) * 8 == x.size(1), "K % 128 == 0, packed [N, K / 8]");
    TORCH_CHECK(s.dim() == 2 && s.size(0) == w.size(0) && s.size(1) * 128 == x.size(1) && z.sizes() == s.sizes(),
                "scales / zeros [N, K / 128]");
}

template <int M>
void gemv_m(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& s, const torch::Tensor& z,
            torch::Tensor& y, int rows, int warps) {
    const int n = w.size(0), k = x.size(1);
    auto stream = c10::cuda::getCurrentCUDAStream();
    const auto* xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
    const auto* wp = reinterpret_cast<const uint4*>(w.data_ptr<int32_t>());
    const auto* sp = reinterpret_cast<const half*>(s.data_ptr<at::Half>());
    const auto* zp = reinterpret_cast<const half*>(z.data_ptr<at::Half>());
    auto* yp = reinterpret_cast<half*>(y.data_ptr<at::Half>());
#define TANDEM_GEMV_CASE(R, W)                                                                \
    if (rows == R && warps == W) {                                                            \
        w4_gemv_kernel<M, R, W><<<(n + R * W - 1) / (R * W), W * 32, 0, stream>>>(xp, wp, sp, zp, yp, n, k); \
        return;                                                                               \
    }
    TANDEM_GEMV_CASE(1, 4) TANDEM_GEMV_CASE(2, 2) TANDEM_GEMV_CASE(2, 4) TANDEM_GEMV_CASE(2, 8)
    TANDEM_GEMV_CASE(4, 2) TANDEM_GEMV_CASE(4, 4) TANDEM_GEMV_CASE(4, 8) TANDEM_GEMV_CASE(8, 2) TANDEM_GEMV_CASE(8, 4)
#undef TANDEM_GEMV_CASE
    TORCH_CHECK(false, "w4_gemv: unsupported rows / warps ", rows, " / ", warps);
}

}  // namespace

torch::Tensor w4_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor s, torch::Tensor z, int64_t rows, int64_t warps) {
    check(x, w, s, z);
    const at::cuda::OptionalCUDAGuard guard(x.device());
    const int m = x.size(0);
    TORCH_CHECK(m >= 1 && m <= 4, "w4_gemv: M <= 4 (larger M: w4_mma)");
    auto y = torch::empty({m, w.size(0)}, x.options());
    switch (m) {
        case 1: gemv_m<1>(x, w, s, z, y, rows, warps); break;
        case 2: gemv_m<2>(x, w, s, z, y, rows, warps); break;
        case 3: gemv_m<3>(x, w, s, z, y, rows, warps); break;
        default: gemv_m<4>(x, w, s, z, y, rows, warps); break;
    }
    return y;
}

torch::Tensor w4_mma(torch::Tensor x, torch::Tensor w, torch::Tensor s, torch::Tensor z, int64_t warps_k,
                     int64_t unroll, int64_t ntile, int64_t act) {
    check(x, w, s, z);
    const at::cuda::OptionalCUDAGuard guard(x.device());
    const int m = x.size(0), n = w.size(0), k = x.size(1);
    TORCH_CHECK(m >= 1 && m <= 16, "w4_mma: M <= 16");
    TORCH_CHECK(act == 0 || (act == 1 && n % 16 == 0), "w4_mma: act 0, or 1 (SwiGLU) with N % 16 == 0");
    auto y = torch::empty({m, act ? n / 2 : n}, x.options());
    const int a = static_cast<int>(act);
    auto stream = c10::cuda::getCurrentCUDAStream();
    const auto* xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
    const auto* wp = reinterpret_cast<const uint4*>(w.data_ptr<int32_t>());
    const auto* sp = reinterpret_cast<const half*>(s.data_ptr<at::Half>());
    const auto* zp = reinterpret_cast<const half*>(z.data_ptr<at::Half>());
    auto* yp = reinterpret_cast<half*>(y.data_ptr<at::Half>());
#define TANDEM_MMA_CASE(W, U, T)                                                                          \
    if (warps_k == W && unroll == U && ntile == T) {                                                      \
        const int blocks = (n + 16 * T - 1) / (16 * T);                                                   \
        if (m > 8) w4_mma_kernel<W, U, 2, T><<<blocks, W * 32, 0, stream>>>(xp, wp, sp, zp, yp, m, n, k, a); \
        else w4_mma_kernel<W, U, 1, T><<<blocks, W * 32, 0, stream>>>(xp, wp, sp, zp, yp, m, n, k, a);      \
        return y;                                                                                         \
    }
    TANDEM_MMA_CASE(1, 2, 1) TANDEM_MMA_CASE(1, 4, 1) TANDEM_MMA_CASE(2, 2, 1) TANDEM_MMA_CASE(2, 4, 1)
    TANDEM_MMA_CASE(4, 1, 1) TANDEM_MMA_CASE(4, 2, 1) TANDEM_MMA_CASE(4, 4, 1) TANDEM_MMA_CASE(8, 1, 1)
    TANDEM_MMA_CASE(8, 2, 1)
    TANDEM_MMA_CASE(2, 2, 2) TANDEM_MMA_CASE(4, 1, 2) TANDEM_MMA_CASE(4, 2, 2) TANDEM_MMA_CASE(8, 1, 2)
    TANDEM_MMA_CASE(8, 2, 2)
#undef TANDEM_MMA_CASE
    TORCH_CHECK(false, "w4_mma: unsupported warps_k / unroll / ntile ", warps_k, " / ", unroll, " / ", ntile);
}

torch::Tensor w4_gemm(torch::Tensor x, torch::Tensor w, torch::Tensor s, torch::Tensor z, torch::Tensor ws,
                      torch::Tensor locks, int64_t mt, int64_t nt, int64_t warps_n, int64_t warps_k, int64_t stages,
                      int64_t splits, int64_t act) {
    check(x, w, s, z);
    TORCH_CHECK(ws.is_cuda() && ws.scalar_type() == at::kFloat && locks.is_cuda() && locks.scalar_type() == at::kInt,
                "w4_gemm: float32 workspace and int32 locks (zero) on the GPU");
    TORCH_CHECK(splits >= 0 && splits <= 16, "w4_gemm: splits 0 (split the last wave) or 1 .. 16");
    const at::cuda::OptionalCUDAGuard guard(x.device());
    const int m = x.size(0), n = w.size(0), k = x.size(1);
    TORCH_CHECK(n % 8 == 0, "w4_gemm: N % 8 == 0");
    TORCH_CHECK(act == 0 || (act == 1 && n % 16 == 0), "w4_gemm: act 0, or 1 (SwiGLU) with N % 16 == 0");
    TORCH_CHECK(static_cast<int64_t>(m) * k < (int64_t{1} << 31), "w4_gemm: M x K < 2^31 (32-bit offsets)");
    auto y = torch::empty({m, act ? n / 2 : n}, x.options());
    if (m == 0) return y;
    const int a = static_cast<int>(act);
    auto stream = c10::cuda::getCurrentCUDAStream();
    const auto* xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
    const auto* wp = reinterpret_cast<const uint4*>(w.data_ptr<int32_t>());
    const auto* sp = reinterpret_cast<const half*>(s.data_ptr<at::Half>());
    const auto* zp = reinterpret_cast<const half*>(z.data_ptr<at::Half>());
    auto* yp = reinterpret_cast<half*>(y.data_ptr<at::Half>());
#define TANDEM_GEMM_CASE(MT, NT, WN, WK, ST)                                                     \
    if (mt == MT && nt == NT && warps_n == WN && warps_k == WK && stages == ST) {               \
        gemm_launch<MT, NT, WN, WK, ST>(xp, wp, sp, zp, yp, ws, locks, m, n, k, splits, a, stream); \
        return y;                                                                                \
    }
    TANDEM_GEMM_CASE(4, 4, 8, 1, 3) TANDEM_GEMM_CASE(4, 4, 4, 1, 4) TANDEM_GEMM_CASE(4, 4, 4, 1, 2)
    TANDEM_GEMM_CASE(4, 4, 4, 2, 2) TANDEM_GEMM_CASE(4, 2, 8, 1, 3)
    TANDEM_GEMM_CASE(2, 4, 8, 1, 4) TANDEM_GEMM_CASE(2, 4, 4, 2, 3) TANDEM_GEMM_CASE(2, 4, 4, 1, 3)
    TANDEM_GEMM_CASE(2, 2, 4, 2, 4) TANDEM_GEMM_CASE(2, 2, 8, 1, 4)
#undef TANDEM_GEMM_CASE
    TORCH_CHECK(false, "w4_gemm: unsupported mt / nt / warps_n / warps_k / stages ", mt, " / ", nt, " / ", warps_n,
                " / ", warps_k, " / ", stages);
}
