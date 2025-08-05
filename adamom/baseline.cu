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
    const size_t block_end_idx = block_start_idx + elements_per_block;

    using opmath_t = double;
    const bool all_aligned = is_aligned(
        params + block_start_idx, grads + block_start_idx, exp_avgs + block_start_idx,
        exp_avg_sqs + block_start_idx, v_bias_corrections + block_start_idx);

    if (all_aligned && (elements_per_block % kILP == 0) && sizeof(scalar_t) == 4) {
        // --- 快速路径 (Fast Path) ---
        // 块内的线程协同处理分配给该块的连续数据
        using VecType = float4;
        const size_t block_start_vec_idx = block_start_idx / kILP;
        const size_t num_vec_per_block = elements_per_block / kILP;
        
        for (int i = threadIdx.x; i < num_vec_per_block; i += blockDim.x) {
            const int vec_idx = block_start_vec_idx + i;

            // 1. Vectorized Load
            VecType r_params_vec = ((VecType*)params)[vec_idx];
            VecType r_grads_vec = ((VecType*)grads)[vec_idx];
            VecType r_m_vec = ((VecType*)exp_avgs)[vec_idx];
            VecType r_v_vec = ((VecType*)exp_avg_sqs)[vec_idx];
            VecType r_v_corr_vec = ((VecType*)v_bias_corrections)[vec_idx];

            opmath_t r_params[kILP] = { (opmath_t)r_params_vec.x, (opmath_t)r_params_vec.y, (opmath_t)r_params_vec.z, (opmath_t)r_params_vec.w };
            opmath_t r_grads[kILP] = { (opmath_t)r_grads_vec.x, (opmath_t)r_grads_vec.y, (opmath_t)r_grads_vec.z, (opmath_t)r_grads_vec.w };
            opmath_t r_m[kILP] = { (opmath_t)r_m_vec.x, (opmath_t)r_m_vec.y, (opmath_t)r_m_vec.z, (opmath_t)r_m_vec.w };
            opmath_t r_v[kILP] = { (opmath_t)r_v_vec.x, (opmath_t)r_v_vec.y, (opmath_t)r_v_vec.z, (opmath_t)r_v_vec.w };
            opmath_t r_v_corr[kILP] = { (opmath_t)r_v_corr_vec.x, (opmath_t)r_v_corr_vec.y, (opmath_t)r_v_corr_vec.z, (opmath_t)r_v_corr_vec.w };

            // 2. Adamom Update
            #pragma unroll
            for (int ii = 0; ii < kILP; ++ii) {
                opmath_t grad = r_grads[ii];
                if (grad_scale_ptr) grad /= static_cast<opmath_t>(*grad_scale_ptr);
                
                opmath_t dx = grad + weight_decay * r_params[ii];
                r_v[ii] = beta2 * r_v[ii] + dx * dx;
                r_v_corr[ii] = beta2 * r_v_corr[ii] + 1.0;
                r_m[ii] = beta1 * r_m[ii] + (1.0 - beta1) * dx;
                opmath_t denom = r_v[ii] / r_v_corr[ii] + eps;
                opmath_t eta = lr * rsqrt(denom);
                r_params[ii] -= eta * r_m[ii];

                if (grad_scale_ptr) r_grads[ii] = grad;
            }

            // 3. Vectorized Store
            r_params_vec = make_float4((float)r_params[0], (float)r_params[1], (float)r_params[2], (float)r_params[3]);
            r_m_vec = make_float4((float)r_m[0], (float)r_m[1], (float)r_m[2], (float)r_m[3]);
            r_v_vec = make_float4((float)r_v[0], (float)r_v[1], (float)r_v[2], (float)r_v[3]);
            r_v_corr_vec = make_float4((float)r_v_corr[0], (float)r_v_corr[1], (float)r_v_corr[2], (float)r_v_corr[3]);
            
            ((VecType*)params)[vec_idx] = r_params_vec;
            ((VecType*)exp_avgs)[vec_idx] = r_m_vec;
            ((VecType*)exp_avg_sqs)[vec_idx] = r_v_vec;
            ((VecType*)v_bias_corrections)[vec_idx] = r_v_corr_vec;
            if (grad_scale_ptr) {
                r_grads_vec = make_float4((float)r_grads[0], (float)r_grads[1], (float)r_grads[2], (float)r_grads[3]);
                ((VecType*)grads)[vec_idx] = r_grads_vec;
            }
        }

    } else {
        // --- 慢速路径 (Slow Path) ---
        // 块内的线程协同处理分配给该块的连续数据
        opmath_t r_params[kILP], r_grads[kILP], r_m[kILP], r_v[kILP], r_v_corr[kILP];

        for (size_t i_chunk_start = 0; i_chunk_start < elements_per_block; i_chunk_start += blockDim.x * kILP) {
            const size_t i_start = block_start_idx + i_chunk_start;
            
            // 1. Strided Load (模拟 load_args)
            #pragma unroll
            for (int ii = 0; ii < kILP; ++ii) {
                const size_t i = i_start + threadIdx.x + ii * blockDim.x;
                if (i < block_end_idx) {
                    r_params[ii] = static_cast<opmath_t>(params[i]);
                    r_grads[ii] = static_cast<opmath_t>(grads[i]);
                    r_m[ii] = static_cast<opmath_t>(exp_avgs[i]);
                    r_v[ii] = static_cast<opmath_t>(exp_avg_sqs[i]);
                    r_v_corr[ii] = static_cast<opmath_t>(v_bias_corrections[i]);
                } else {
                    r_params[ii] = 0; r_grads[ii] = 0; r_m[ii] = 0; r_v[ii] = 0; r_v_corr[ii] = 0;
                }
            }

            // 2. Adamom Update
            #pragma unroll
            for (int ii = 0; ii < kILP; ++ii) {
                const size_t i = i_start + threadIdx.x + ii * blockDim.x;
                if (i < block_end_idx) {
                    opmath_t grad = r_grads[ii];
                    if (grad_scale_ptr) grad /= static_cast<opmath_t>(*grad_scale_ptr);
                    
                    opmath_t dx = grad + weight_decay * r_params[ii];
                    r_v[ii] = beta2 * r_v[ii] + dx * dx;
                    r_v_corr[ii] = beta2 * r_v_corr[ii] + 1.0;
                    r_m[ii] = beta1 * r_m[ii] + (1.0 - beta1) * dx;
                    opmath_t denom = r_v[ii] / r_v_corr[ii] + eps;
                    opmath_t eta = lr * rsqrt(denom);
                    r_params[ii] -= eta * r_m[ii];

                    if (grad_scale_ptr) r_grads[ii] = grad;
                }
            }

            // 3. Strided Store
            #pragma unroll
            for (int ii = 0; ii < kILP; ++ii) {
                const size_t i = i_start + threadIdx.x + ii * blockDim.x;
                if (i < block_end_idx) {
                    params[i] = static_cast<scalar_t>(r_params[ii]);
                    exp_avgs[i] = static_cast<scalar_t>(r_m[ii]);
                    exp_avg_sqs[i] = static_cast<scalar_t>(r_v[ii]);
                    v_bias_corrections[i] = static_cast<scalar_t>(r_v_corr[ii]);
                    if (grad_scale_ptr) grads[i] = static_cast<scalar_t>(r_grads[ii]);
                }
            }
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
    const size_t elements_per_block = (size_t)blockSize * 65536;
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

/*
--- 如何编译和运行 ---

1. 确保您已经安装了 NVIDIA CUDA Toolkit。
2. 将以上代码保存为 `adamom_cuda.cu` 文件。
3. 打开终端，使用 nvcc 编译器进行编译：
   nvcc adamom_cuda.cu -o adamom_cuda -std=c++17

4. 运行生成的可执行文件：
   ./adamom_cuda
*/
