#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <stdexcept>

// 包含 CUDA 运行时头文件
#include <cuda_runtime.h>
#include <cuda_fp16.h> // For potential half-precision tests

// --- CUDA 错误检查宏 ---
#define CUDA_CHECK(err) { \
    cudaError_t err_code = err; \
    if (err_code != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err_code) \
                  << " in file " << __FILE__ \
                  << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// 定义指令级并行度 (Instruction-Level Parallelism)
constexpr int kILP = 4;

// --- 内存对齐检查函数 ---
// 对应于 init_args 中的对齐检查逻辑
template<typename scalar_t>
__device__ __forceinline__ bool is_aligned(
    scalar_t* p1, scalar_t* p2, scalar_t* p3, scalar_t* p4, scalar_t* p5) {
    // kILP=4, scalar_t=float -> 16 bytes. float4 requires 16-byte alignment.
    constexpr int alignment_bytes = kILP * sizeof(scalar_t);
    return (reinterpret_cast<uintptr_t>(p1) % alignment_bytes == 0) &&
           (reinterpret_cast<uintptr_t>(p2) % alignment_bytes == 0) &&
           (reinterpret_cast<uintptr_t>(p3) % alignment_bytes == 0) &&
           (reinterpret_cast<uintptr_t>(p4) % alignment_bytes == 0) &&
           (reinterpret_cast<uintptr_t>(p5) % alignment_bytes == 0);
}



template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
};

template <typename T>
__device__ __forceinline__ void load_store(
    T* dst,
    T* src,
    int64_t dst_offset,
    int64_t src_offset) {
  using LT = aligned_vector<T, kILP>;
  ((LT*)dst)[dst_offset] = ((LT*)src)[src_offset];
}

// --- Adamom CUDA 内核 ---
// 精确复现了原始 kernel 的 fast/slow path 逻辑
// 每个块处理一个大的、连续的数据块
template <typename scalar_t>
__global__ void adamom_kernel(
    scalar_t* params,
    scalar_t* grads,
    scalar_t* exp_avgs,      // m (一阶动量)
    scalar_t* exp_avg_sqs,   // v (二阶动量)
    scalar_t* v_bias_corrections, // v 的偏差修正项
    size_t elements_per_block, // 每个块要处理的元素数量
    double lr, double beta1, double beta2, double eps, double weight_decay,
    const float* grad_scale_ptr, const float* found_inf_ptr) {

    if (found_inf_ptr && *found_inf_ptr == 1.0f) {
        return;
    }

    // 计算当前块负责处理的数据块的起始和结束索引
    const size_t block_start_idx = (size_t)blockIdx.x * elements_per_block;

    using opmath_t = float;
    const bool all_aligned = is_aligned(
        params + block_start_idx, grads + block_start_idx, exp_avgs + block_start_idx,
        exp_avg_sqs + block_start_idx, v_bias_corrections + block_start_idx);

    constexpr int depth = 5;

    float* args[depth];
    float r_args[depth][kILP];
    args[0] = params + block_start_idx;
    args[1] = grads + block_start_idx;
    args[2] = exp_avgs + block_start_idx;
    args[3] = exp_avg_sqs + block_start_idx;
    args[4] = v_bias_corrections + block_start_idx;

    if (all_aligned && (elements_per_block % kILP == 0) && sizeof(scalar_t) == 4) {
        for (int64_t i_start = threadIdx.x; i_start * kILP < elements_per_block; i_start += blockDim.x) {
            #pragma unroll
            for (int i = 0; i < depth; ++i) {
              load_store(r_args[i], args[i], 0, i_start);
            }

            // 2. Adamom Update
            #pragma unroll
            for (int ii = 0; ii < kILP; ++ii) {
                opmath_t grad = static_cast<opmath_t>(r_args[1][ii]);
                opmath_t m = static_cast<opmath_t>(r_args[2][ii]);
                opmath_t v = static_cast<opmath_t>(r_args[3][ii]);
                opmath_t v_bias_correction = static_cast<opmath_t>(r_args[4][ii]);
                opmath_t weight = static_cast<opmath_t>(r_args[0][ii]);

                if (grad_scale_ptr) {
                    grad /= (static_cast<double>(*grad_scale_ptr));
                }
                const opmath_t grad_to_store = grad;

                opmath_t dx = grad + weight_decay * weight;

                v = beta2 * v + dx * dx;
                v_bias_correction = beta2 * v_bias_correction + 1.0;

                m = beta1 * m + (1.0 - beta1) * dx;

                opmath_t denom = v / v_bias_correction + eps;
                opmath_t eta = lr * rsqrt(denom);

                weight -= eta * m;

                // Write back
                if (grad_scale_ptr) {
                    r_args[1][ii] = grad_to_store;
                }
                r_args[2][ii] = static_cast<scalar_t>(m);
                r_args[3][ii] = static_cast<scalar_t>(v);
                r_args[4][ii] = static_cast<scalar_t>(v_bias_correction);
                r_args[0][ii] = static_cast<scalar_t>(weight);
            }

            #pragma unroll
            for (int i = 0; i < depth; ++i) {
              load_store(args[i], r_args[i], i_start, 0);
            }
        }

    } else {
        if(blockIdx.x == 0) {
            printf("Using slow path due to alignment or ILP constraints.\n");
            return;
        }
    }
}


// 辅助函数，用于打印 vector 内容
template <typename T>
void print_vector(const std::string& name, const std::vector<T>& vec, int count = 10) {
    std::cout << name << ": [";
    int n = std::min((int)vec.size(), count);
    for (int i = 0; i < n; ++i) {
        std::cout << vec[i] << (i == n - 1 ? "" : ", ");
    }
    if (vec.size() > n) std::cout << ", ...";
    std::cout << "]" << std::endl;
}

// --- Main 函数 ---
int main() {
    // --- 1. 设置测试环境 (Host 端) ---
    // 根据您的要求进行大规模配置
    const int blockSize = 512;
    const int gridSize = 320;
    const size_t elements_per_block =  65536;
    // 注意：这是一个非常大的内存分配（~40 GB），请确保您的 GPU 有足够显存。
    const size_t num_elements = elements_per_block * gridSize;
    using DType = float;

    std::cout << "Configuring large-scale test:" << std::endl;
    std::cout << "  - Total Blocks (Grid Size): " << gridSize << std::endl;
    std::cout << "  - Threads per Block: " << blockSize << std::endl;
    std::cout << "  - Elements per Block: " << elements_per_block << std::endl;
    std::cout << "  - Total Elements: " << num_elements << " (" << (double)num_elements / 1e9 << " billion)" << std::endl;
    std::cout << "  - Estimated Memory: " << (double)(num_elements * sizeof(DType) * 5) / (1024*1024*1024) << " GB" << std::endl;


    std::vector<DType> h_params(num_elements);
    std::vector<DType> h_grads(num_elements);
    std::vector<DType> h_exp_avgs(num_elements, 0.0f);
    std::vector<DType> h_exp_avg_sqs(num_elements, 0.0f);
    std::vector<DType> h_v_bias_corrections(num_elements, 0.0f);

    std::iota(h_params.begin(), h_params.end(), 1.0f);
    for(size_t i = 0; i < num_elements; ++i) {
        h_grads[i] = static_cast<DType>(i % 100) * 0.1f;
    }

    const double lr = 0.01;
    const double beta1 = 0.9;
    const double beta2 = 0.999;
    const double eps = 1e-8;
    const double weight_decay = 0.01;

    // --- 2. 分配 GPU 内存 (Device 端) ---
    DType *d_params, *d_grads, *d_exp_avgs, *d_exp_avg_sqs, *d_v_bias_corrections;
    size_t data_size = num_elements * sizeof(DType);

    std::cout << "\nAllocating GPU memory..." << std::endl;
    CUDA_CHECK(cudaMalloc(&d_params, data_size));
    CUDA_CHECK(cudaMalloc(&d_grads, data_size));
    CUDA_CHECK(cudaMalloc(&d_exp_avgs, data_size));
    CUDA_CHECK(cudaMalloc(&d_exp_avg_sqs, data_size));
    CUDA_CHECK(cudaMalloc(&d_v_bias_corrections, data_size));
    std::cout << "Memory allocated successfully." << std::endl;

    // --- 3. 将数据从 Host 拷贝到 Device ---
    std::cout << "Copying data from Host to Device..." << std::endl;
    CUDA_CHECK(cudaMemcpy(d_params, h_params.data(), data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grads, h_grads.data(), data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_exp_avgs, h_exp_avgs.data(), data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_exp_avg_sqs, h_exp_avg_sqs.data(), data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_bias_corrections, h_v_bias_corrections.data(), data_size, cudaMemcpyHostToDevice));
    std::cout << "Data copied successfully." << std::endl;

    // --- 4. 设置内核启动参数并启动内核 ---
    std::cout << "\n--- Before Optimizer Step ---" << std::endl;
    print_vector("Params", h_params);
    print_vector("Grads", h_grads);

    std::cout << "\n--- Running CUDA Kernel ---" << std::endl;
    adamom_kernel<DType><<<gridSize, blockSize>>>(
        d_params, d_grads, d_exp_avgs, d_exp_avg_sqs, d_v_bias_corrections,
        elements_per_block,
        lr, beta1, beta2, eps, weight_decay,
        nullptr, nullptr
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Kernel execution finished." << std::endl;

    // --- 5. 将结果从 Device 拷贝回 Host ---
    CUDA_CHECK(cudaMemcpy(h_params.data(), d_params, data_size, cudaMemcpyDeviceToHost));

    std::cout << "\n--- After Optimizer Step ---" << std::endl;
    print_vector("Updated Params", h_params);

    // --- 6. 释放 GPU 内存 ---
    CUDA_CHECK(cudaFree(d_params));
    CUDA_CHECK(cudaFree(d_grads));
    CUDA_CHECK(cudaFree(d_exp_avgs));
    CUDA_CHECK(cudaFree(d_exp_avg_sqs));
    CUDA_CHECK(cudaFree(d_v_bias_corrections));

    std::cout << "\nCUDA execution finished successfully." << std::endl;

    return 0;
}

