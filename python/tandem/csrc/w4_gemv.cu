// W4A16 decode kernels (design doc P2 / A2): y[M, N] = x[M, K] @ dequant(W)[N, K]^T for M <= 16.
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

#include <cuda_fp16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

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
