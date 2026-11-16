// Whole-page KV copies between the GPU cache and a pinned host pool (design doc §4.4; the
// GPU-assisted I/O of SGLang's HiCache).
//
// GPU cache: layer-first, one K and one V tensor per layer; the tokens of block b occupy bytes
// [b * page_bytes, (b + 1) * page_bytes) of each (tandem/cache/kv_cache.py).
// Host pool: page-first, [pages][layers][K, V][page_bytes], so a page holds every layer and is
// read or written as one unit.
//
// The kernel reads or writes the mapped pinned memory directly: one launch per batch of pages
// (or per layer, for pipelined loads) instead of a cudaMemcpy per page, layer and K/V (72 per
// page for Qwen3-8B, 32 KB each). Its grid is small and fixed (`ctas`): a copy is bound by PCIe
// (13 GB/s), which a few SMs saturate, while one CTA per chunk would hold every SM's thread
// slots for the whole transfer and slow down the model running beside it.

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <mutex>
#include <vector>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

namespace {

constexpr int kThreads = 256, kUnroll = 4;

// Pinned buffers whose tensors were freed. cudaFreeHost is not allowed while a CUDA graph is
// being captured (global capture mode), and a pool can be garbage-collected at any time,
// including during another engine's capture; so frees wait for the next pinned_empty call.
std::mutex g_pending_mu;
std::vector<void*> g_pending;

void free_pending() {
    std::vector<void*> ptrs;
    {
        std::lock_guard<std::mutex> lock(g_pending_mu);
        ptrs.swap(g_pending);
    }
    for (void* p : ptrs) cudaFreeHost(p);
}

// Chunks are (page in the batch, layer-K/V index lk0 .. lk0 + lk_count - 1), page_bytes each.
__global__ void __launch_bounds__(kThreads) kv_page_copy_kernel(
    const int64_t* __restrict__ layer_ptrs, const int32_t* __restrict__ gpu_blocks,
    const int32_t* __restrict__ host_pages, char* __restrict__ host, int64_t page_bytes, int layers2, int lk0,
    int lk_count, int num_pages, bool to_host) {
    const int64_t n16 = page_bytes / 16;
    const int64_t chunks = static_cast<int64_t>(num_pages) * lk_count;
    for (int64_t c = blockIdx.x; c < chunks; c += gridDim.x) {
        const int p = static_cast<int>(c / lk_count), lk = lk0 + static_cast<int>(c % lk_count);
        char* g = reinterpret_cast<char*>(layer_ptrs[lk]) + static_cast<int64_t>(gpu_blocks[p]) * page_bytes;
        char* h = host + (static_cast<int64_t>(host_pages[p]) * layers2 + lk) * page_bytes;
        const uint4* src = reinterpret_cast<const uint4*>(to_host ? g : h);
        uint4* dst = reinterpret_cast<uint4*>(to_host ? h : g);
        for (int64_t base = threadIdx.x; base < n16; base += kThreads * kUnroll) {
            uint4 v[kUnroll];  // kUnroll loads in flight per thread before the stores: PCIe latency
#pragma unroll
            for (int u = 0; u < kUnroll; ++u) {
                const int64_t i = base + u * kThreads;
                if (i < n16) v[u] = src[i];
            }
#pragma unroll
            for (int u = 0; u < kUnroll; ++u) {
                const int64_t i = base + u * kThreads;
                if (i < n16) dst[i] = v[u];
            }
        }
    }
}

}  // namespace

// Exact-size mapped pinned memory (torch's pinned allocator rounds up to a power of two: a 30 GB
// pool would take 34 GB). Returns a uint8 CPU tensor; its memory is freed by the next call.
torch::Tensor pinned_empty(int64_t nbytes) {
    free_pending();  // a safe point: pools are created at engine start, outside graph capture
    void* p = nullptr;
    const cudaError_t err = cudaHostAlloc(&p, static_cast<size_t>(nbytes), cudaHostAllocMapped | cudaHostAllocPortable);
    TORCH_CHECK(err == cudaSuccess, "cudaHostAlloc(", nbytes, "): ", cudaGetErrorString(err));
    return torch::from_blob(p, {nbytes}, [](void* q) {
                                std::lock_guard<std::mutex> lock(g_pending_mu);
                                g_pending.push_back(q);
                            },
                            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true));
}

// Device-side address of mapped pinned memory.
int64_t device_ptr(torch::Tensor pinned) {
    void* d = nullptr;
    const cudaError_t err = cudaHostGetDevicePointer(&d, pinned.data_ptr(), 0);
    TORCH_CHECK(err == cudaSuccess, "cudaHostGetDevicePointer: ", cudaGetErrorString(err));
    return reinterpret_cast<int64_t>(d);
}

// Copy pages gpu_blocks[i] <-> host_pages[i] on the current stream: layer-K/V tensors lk0 ..
// lk0 + lk_count - 1 (lk = 2 * layer + (0: K, 1: V); lk_count = -1: all of them).
void kv_page_copy(torch::Tensor layer_ptrs, torch::Tensor gpu_blocks, torch::Tensor host_pages, int64_t host_dev_ptr,
                  int64_t page_bytes, bool to_host, int64_t lk0, int64_t lk_count, int64_t ctas) {
    TORCH_CHECK(layer_ptrs.is_cuda() && layer_ptrs.scalar_type() == at::kLong, "layer_ptrs: int64 CUDA tensor");
    TORCH_CHECK(gpu_blocks.is_cuda() && host_pages.is_cuda() && gpu_blocks.scalar_type() == at::kInt &&
                host_pages.scalar_type() == at::kInt && gpu_blocks.numel() == host_pages.numel(),
                "gpu_blocks / host_pages: int32 CUDA tensors of the same length");
    TORCH_CHECK(page_bytes > 0 && page_bytes % 16 == 0, "page_bytes: a positive multiple of 16");
    const int layers2 = static_cast<int>(layer_ptrs.numel());
    if (lk_count < 0) lk_count = layers2 - lk0;
    TORCH_CHECK(lk0 >= 0 && lk_count >= 1 && lk0 + lk_count <= layers2 && ctas >= 1, "kv_page_copy: bad range");
    const int n = static_cast<int>(gpu_blocks.numel());
    if (n == 0) return;
    const at::cuda::OptionalCUDAGuard guard(gpu_blocks.device());
    const int64_t chunks = static_cast<int64_t>(n) * lk_count;
    const int grid = static_cast<int>(std::min<int64_t>(chunks, ctas));
    kv_page_copy_kernel<<<grid, kThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        layer_ptrs.data_ptr<int64_t>(), gpu_blocks.data_ptr<int32_t>(), host_pages.data_ptr<int32_t>(),
        reinterpret_cast<char*>(host_dev_ptr), page_bytes, layers2, static_cast<int>(lk0), static_cast<int>(lk_count),
        n, to_host);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
