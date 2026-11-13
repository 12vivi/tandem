// Whole-page KV copies between the GPU cache and a pinned host pool (design doc §4.4; the
// GPU-assisted I/O of SGLang's HiCache).
//
// GPU cache: layer-first, one K and one V tensor per layer; the tokens of block b occupy bytes
// [b * page_bytes, (b + 1) * page_bytes) of each (tandem/cache/kv_cache.py).
// Host pool: page-first, [pages][layers][K, V][page_bytes], so a page holds every layer and is
// read or written as one unit.
//
// One CTA per (page, layer, K/V) moves page_bytes with 16-byte accesses, reading or writing the
// mapped pinned memory directly: one launch per batch of pages instead of 2 x layers x pages
// cudaMemcpy calls (72 per page for Qwen3-8B, 32 KB each).

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

namespace {

__global__ void kv_page_copy_kernel(const int64_t* __restrict__ layer_ptrs, const int32_t* __restrict__ gpu_blocks,
                                    const int32_t* __restrict__ host_pages, char* __restrict__ host, int64_t page_bytes,
                                    int layers2, bool to_host) {
    const int p = blockIdx.x, lk = blockIdx.y;  // page in the batch, layer * 2 + (0: K, 1: V)
    char* g = reinterpret_cast<char*>(layer_ptrs[lk]) + static_cast<int64_t>(gpu_blocks[p]) * page_bytes;
    char* h = host + (static_cast<int64_t>(host_pages[p]) * layers2 + lk) * page_bytes;
    const uint4* src = reinterpret_cast<const uint4*>(to_host ? g : h);
    uint4* dst = reinterpret_cast<uint4*>(to_host ? h : g);
    const int64_t n = page_bytes / 16;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}

}  // namespace

// Exact-size mapped pinned memory (torch's pinned allocator rounds up to a power of two: a 30 GB
// pool would take 34 GB). Returns a uint8 CPU tensor that frees the memory with it.
torch::Tensor pinned_empty(int64_t nbytes) {
    void* p = nullptr;
    const cudaError_t err = cudaHostAlloc(&p, static_cast<size_t>(nbytes), cudaHostAllocMapped | cudaHostAllocPortable);
    TORCH_CHECK(err == cudaSuccess, "cudaHostAlloc(", nbytes, "): ", cudaGetErrorString(err));
    return torch::from_blob(p, {nbytes}, [](void* q) { cudaFreeHost(q); },
                            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true));
}

// Device-side address of mapped pinned memory.
int64_t device_ptr(torch::Tensor pinned) {
    void* d = nullptr;
    const cudaError_t err = cudaHostGetDevicePointer(&d, pinned.data_ptr(), 0);
    TORCH_CHECK(err == cudaSuccess, "cudaHostGetDevicePointer: ", cudaGetErrorString(err));
    return reinterpret_cast<int64_t>(d);
}

// Copy the pages gpu_blocks[i] <-> host_pages[i] of every layer on the current stream.
void kv_page_copy(torch::Tensor layer_ptrs, torch::Tensor gpu_blocks, torch::Tensor host_pages, int64_t host_dev_ptr,
                  int64_t page_bytes, bool to_host) {
    TORCH_CHECK(layer_ptrs.is_cuda() && layer_ptrs.scalar_type() == at::kLong, "layer_ptrs: int64 CUDA tensor");
    TORCH_CHECK(gpu_blocks.is_cuda() && host_pages.is_cuda() && gpu_blocks.scalar_type() == at::kInt &&
                host_pages.scalar_type() == at::kInt && gpu_blocks.numel() == host_pages.numel(),
                "gpu_blocks / host_pages: int32 CUDA tensors of the same length");
    TORCH_CHECK(page_bytes > 0 && page_bytes % 16 == 0, "page_bytes: a positive multiple of 16");
    const int n = static_cast<int>(gpu_blocks.numel());
    if (n == 0) return;
    const at::cuda::OptionalCUDAGuard guard(gpu_blocks.device());
    const dim3 grid(n, static_cast<unsigned>(layer_ptrs.numel()));
    kv_page_copy_kernel<<<grid, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        layer_ptrs.data_ptr<int64_t>(), gpu_blocks.data_ptr<int32_t>(), host_pages.data_ptr<int32_t>(),
        reinterpret_cast<char*>(host_dev_ptr), page_bytes, static_cast<int>(layer_ptrs.numel()), to_host);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
