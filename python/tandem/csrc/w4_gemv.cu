// W4A16 small-batch GEMV for decode (design doc P2 / A2): y[M, N] = x[M, K] @ dequant(W)[N, K]^T, M <= 8.
//
// Weights in tandem's repacked AWQ layout (tandem/layers/awq.py): packed [N, K/8] int32, 8
// consecutive k of one output row per int32 in AWQ's nibble order (k 0..7 in nibbles 0,4,1,5,2,6,
// 3,7), scales / zeros [N, K/128] float16, w = (q - z) * s.
//
// Each warp computes ROWS output rows; each lane takes 32 consecutive k per step (one 16-byte load
// per row), so a warp streams 512 contiguous bytes of a row per step. Dequantization uses the
// LOP3 trick of FasterTransformer / Marlin: OR-ing a nibble pair into the bits of half2(1024, 1024)
// gives (1024 + q0, 1024 + q1) with no int-to-float conversion; AWQ's nibble order makes every
// half2 hold two consecutive k. Products of 8 weights are summed in float16, then accumulated in
// float32; a warp shuffle reduction finishes each output.

#include <cuda_fp16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace {

__device__ __forceinline__ void dequant8(uint32_t i4s, half2 (&h)[4]) {
    uint32_t* hh = reinterpret_cast<uint32_t*>(h);
    constexpr uint32_t kLut = (0xf0 & 0xcc) | 0xaa;  // (a & b) | c
    constexpr uint32_t kBottom = 0x000f000f, kTop = 0x00f000f0, kMagic = 0x64006400;  // half2(1024, 1024)
    const uint32_t top = i4s >> 8;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(hh[0]) : "r"(i4s), "n"(kBottom), "n"(kMagic), "n"(kLut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(hh[1]) : "r"(i4s), "n"(kTop), "n"(kMagic), "n"(kLut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(hh[2]) : "r"(top), "n"(kBottom), "n"(kMagic), "n"(kLut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(hh[3]) : "r"(top), "n"(kTop), "n"(kMagic), "n"(kLut));
    // h[0], h[2] = 1024 + q; h[1], h[3] = 1024 + 16 q (the caller folds the offsets with the zero point)
}

template <int M, int ROWS, int WARPS>
__global__ void __launch_bounds__(WARPS * 32) w4_gemv_kernel(
    const half* __restrict__ x, const uint32_t* __restrict__ w, const half* __restrict__ scales,
    const half* __restrict__ zeros, half* __restrict__ y, int N, int K) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * WARPS + warp) * ROWS;
    if (row0 >= N) return;
    const int groups = K >> 7;  // group size 128
    const half2 kSixteenth = __float2half2_rn(1.0f / 16.0f);
    float acc[M][ROWS];
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[m][r] = 0.0f;

    for (int kb = lane * 32; kb < K; kb += 32 * 32) {
        uint4 wv[ROWS];
        half2 zz[ROWS], zt[ROWS], ss[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = min(row0 + r, N - 1);  // rows past N recompute the last row; not stored
            wv[r] = __ldg(reinterpret_cast<const uint4*>(w + static_cast<size_t>(row) * (K >> 3) + (kb >> 3)));
            const int g = row * groups + (kb >> 7);
            const half z = __ldg(zeros + g);
            ss[r] = __half2half2(__ldg(scales + g));
            zz[r] = __half2half2(__hadd(z, __float2half(1024.0f)));  // 1024 + z
            zt[r] = __half2half2(__hneg(__hadd(z, __float2half(64.0f))));  // -(64 + z)
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {  // 4 int32 = 32 weights per row
            half2 xv[M][4];
#pragma unroll
            for (int m = 0; m < M; ++m) {
                const uint4 v = __ldg(reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * K + kb + 8 * i));
                const half2* hv = reinterpret_cast<const half2*>(&v);
                xv[m][0] = hv[0]; xv[m][1] = hv[1]; xv[m][2] = hv[2]; xv[m][3] = hv[3];
            }
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const uint32_t packed = reinterpret_cast<const uint32_t*>(&wv[r])[i];
                half2 h[4];
                dequant8(packed, h);
                h[0] = __hmul2(__hsub2(h[0], zz[r]), ss[r]);
                h[1] = __hmul2(__hfma2(h[1], kSixteenth, zt[r]), ss[r]);
                h[2] = __hmul2(__hsub2(h[2], zz[r]), ss[r]);
                h[3] = __hmul2(__hfma2(h[3], kSixteenth, zt[r]), ss[r]);
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    half2 p = __hmul2(h[0], xv[m][0]);
                    p = __hfma2(h[1], xv[m][1], p);
                    p = __hfma2(h[2], xv[m][2], p);
                    p = __hfma2(h[3], xv[m][3], p);
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

template <int M>
void launch(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& s, const torch::Tensor& z,
            torch::Tensor& y, int rows, int warps) {
    const int n = w.size(0), k = x.size(1);
    auto stream = c10::cuda::getCurrentCUDAStream();
    const auto* xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
    const auto* wp = reinterpret_cast<const uint32_t*>(w.data_ptr<int32_t>());
    const auto* sp = reinterpret_cast<const half*>(s.data_ptr<at::Half>());
    const auto* zp = reinterpret_cast<const half*>(z.data_ptr<at::Half>());
    auto* yp = reinterpret_cast<half*>(y.data_ptr<at::Half>());
#define TANDEM_W4_CASE(R, W)                                                                    \
    if (rows == R && warps == W) {                                                              \
        const int blocks = (n + R * W - 1) / (R * W);                                           \
        w4_gemv_kernel<M, R, W><<<blocks, W * 32, 0, stream>>>(xp, wp, sp, zp, yp, n, k);       \
        return;                                                                                 \
    }
    TANDEM_W4_CASE(1, 4) TANDEM_W4_CASE(2, 4) TANDEM_W4_CASE(4, 4)
    TANDEM_W4_CASE(1, 8) TANDEM_W4_CASE(2, 8) TANDEM_W4_CASE(1, 2) TANDEM_W4_CASE(2, 2)
#undef TANDEM_W4_CASE
    TORCH_CHECK(false, "w4_gemv: unsupported rows/warps ", rows, "/", warps);
}

}  // namespace

torch::Tensor w4_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor s, torch::Tensor z, int64_t rows, int64_t warps) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.is_contiguous(), "x: contiguous float16 CUDA");
    TORCH_CHECK(w.scalar_type() == at::kInt && s.scalar_type() == at::kHalf && z.scalar_type() == at::kHalf, "dtypes");
    const int m = x.size(0), k = x.size(1);
    TORCH_CHECK(m >= 1 && m <= 8 && k % 128 == 0 && w.size(1) * 8 == k, "shapes: M <= 8, K % 128 == 0");
    auto y = torch::empty({m, w.size(0)}, x.options());
    switch (m) {
        case 1: launch<1>(x, w, s, z, y, rows, warps); break;
        case 2: launch<2>(x, w, s, z, y, rows, warps); break;
        case 3: launch<3>(x, w, s, z, y, rows, warps); break;
        case 4: launch<4>(x, w, s, z, y, rows, warps); break;
        case 5: launch<5>(x, w, s, z, y, rows, warps); break;
        case 6: launch<6>(x, w, s, z, y, rows, warps); break;
        case 7: launch<7>(x, w, s, z, y, rows, warps); break;
        default: launch<8>(x, w, s, z, y, rows, warps); break;
    }
    return y;
}
