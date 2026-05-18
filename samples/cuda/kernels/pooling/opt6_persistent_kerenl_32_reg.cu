#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime_api.h>
#include <fstream>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

// ==========================================================================
// 1. 缺失的定义补充
// ==========================================================================

/**
 * @brief 计算向上取整的整数除法 (ceil(a/b))。
 * @param a 分子
 * @param b 分母
 */
#define ITER(x, y) (x + y - 1) / y

template <typename DATA_TYPE,
          typename std::enable_if_t<
              !std::is_same<__half, DATA_TYPE>::value &&
              !std::is_same<__nv_bfloat16, DATA_TYPE>::value> * = nullptr>
__device__ __forceinline__ void fastSpecializedAtomicAdd(DATA_TYPE *base,
                                                         int64_t offset,
                                                         const int64_t length,
                                                         DATA_TYPE value) {
  atomicAdd(base + offset, value);
}

template <class DATA_TYPE>
__device__ __forceinline__ void fastAtomicAdd(DATA_TYPE *base, int64_t offset,
                                              const int64_t length,
                                              DATA_TYPE value) {
  fastSpecializedAtomicAdd(base, offset, length, value);
}

// struct SharedMemory {
//   int edge_in_tile[TILE_K_PER_BLOCK];
//   int edge_out_tile[TILE_K_PER_BLOCK];
//   // 双缓冲，用于隐藏内存延迟
//   // 布局: [buffer_idx][warp_idx][element_idx]
//   DATA_TYPE emb_buffer[2][NUM_WARPS * EMB_DIM_STATIC];
// };

template <typename DATA_TYPE, int TILE_K_PER_BLOCK, int BLOCK_THREADS>
__global__ void __launch_bounds__(BLOCK_THREADS)
    gpu_pooling_forward_async_kernel(
        const DATA_TYPE *__restrict__ emb_table,
        const int *__restrict__ edge_in, const int *__restrict__ edge_out,
        DATA_TYPE *__restrict__ pooling_table, const int64_t emb_dim,
        const int edge_length,
        int *work_tile_idx /* 新增：全局工作块原子计数器 */) {
  constexpr int WARP_SIZE = 32;
  constexpr int NUM_WARPS = BLOCK_THREADS / WARP_SIZE;
  const int thread_id = threadIdx.x;
  const int warp_id = thread_id / WARP_SIZE;
  const int lane_id = thread_id % WARP_SIZE;

  // 动态共享内存布局
  extern __shared__ char smem_storage[];
  int *smem_edge_in = reinterpret_cast<int *>(smem_storage);
  int *smem_edge_out = smem_edge_in + TILE_K_PER_BLOCK;
  int *smem_work_tile_idx =
      reinterpret_cast<int *>(smem_edge_out + TILE_K_PER_BLOCK);
  while (true) {
    // 1. 原子地获取一个工作块索引 (work-stealing)
    if (thread_id == 0) {
      *smem_work_tile_idx = atomicAdd(work_tile_idx, 1);
    }
    __syncthreads();
    const int tile_idx = *smem_work_tile_idx;
    // 2. 根据领取的索引计算该块处理的边范围
    const int block_tile_start = tile_idx * TILE_K_PER_BLOCK;

    // 3. 检查是否所有工作都已完成，若是则退出循环
    if (block_tile_start >= edge_length) {
      break;
    }

    const int block_tile_end =
        min(block_tile_start + TILE_K_PER_BLOCK, edge_length);
    const int block_tile_size = block_tile_end - block_tile_start;

    // 预加载边索引到共享内存
    __syncthreads();  // 确保上一轮迭代的共享内存使用已完成
    for (int i = thread_id; i < block_tile_size; i += BLOCK_THREADS) {
      smem_edge_in[i] = edge_in[block_tile_start + i];
      smem_edge_out[i] = edge_out[block_tile_start + i];
    }
    int k_base = 0;
    __syncthreads();

    // 主处理循环 (处理一个tile内部的边)
    while (true) {
      if (k_base >= block_tile_size) break;

      // 每个warp处理一个边
      const int k_warp = k_base + warp_id;
      if (k_warp < block_tile_size) {
        const int in_offset = smem_edge_in[k_warp] * emb_dim;
        const int out_offset = smem_edge_out[k_warp] * emb_dim;

        // 处理embedding维度
        for (int j = lane_id; j < emb_dim; j += WARP_SIZE) {
          atomicAdd(&pooling_table[out_offset + j], emb_table[in_offset + j]);
        }
      }
      k_base += NUM_WARPS;
    }
  }  // 结束主 while 循环
}

// 宏定义用于检查CUDA API调用的返回状态
#define CUDA_CHECK(call)                                               \
  do {                                                                 \
    cudaError_t status = call;                                         \
    if (status != cudaSuccess) {                                       \
      fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status));                             \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

// ==========================================================================
// 3. Main函数：用于数据构造、内核调用和性能测试
// ==========================================================================

int main() {
  // ---- 1. 参数定义 ----
  // (此部分保持不变)

  using DataType = float;
  const int TILE_INDICES_VAL = 512;
  const int BLOCK_SIZE = 512;
  std::ifstream inFile("binary_data.bin", std::ios::binary);

  int edge_length, emb_table_length, pooling_table_length;
  int64_t emb_dim;
  inFile.read(reinterpret_cast<char *>(&edge_length), sizeof(int));
  inFile.read(reinterpret_cast<char *>(&emb_table_length), sizeof(int));
  inFile.read(reinterpret_cast<char *>(&pooling_table_length), sizeof(int));
  inFile.read(reinterpret_cast<char *>(&emb_dim), sizeof(int64_t));

  int *edge_in_cpu = reinterpret_cast<int *>(malloc(edge_length * sizeof(int)));
  int *edge_out_cpu =
      reinterpret_cast<int *>(malloc(edge_length * sizeof(int)));
  DataType *emb_table_cpu =
      reinterpret_cast<DataType *>(malloc(emb_table_length * sizeof(DataType)));
  DataType *pooling_table_cpu = reinterpret_cast<DataType *>(
      malloc(pooling_table_length * sizeof(DataType)));
  inFile.read(reinterpret_cast<char *>(edge_in_cpu), edge_length * sizeof(int));
  inFile.read(reinterpret_cast<char *>(edge_out_cpu),
              edge_length * sizeof(int));
  inFile.read(reinterpret_cast<char *>(emb_table_cpu),
              emb_table_length * sizeof(float));

  std::cout << "===== KERNEL PERFORMANCE TEST (CORRECTED) =====" << std::endl;
  std::cout << "Data Type: float" << std::endl;
  std::cout << "Embedding Dim: " << emb_dim << std::endl;
  std::cout << "Edge Length: " << edge_length << std::endl;
  std::cout << "===============================================" << std::endl;

  DataType *d_emb_table, *d_pooling_table;
  int *d_edge_in, *d_edge_out;

  CUDA_CHECK(cudaMalloc(&d_emb_table, emb_table_length * sizeof(DataType)));
  CUDA_CHECK(cudaMalloc(&d_edge_in, edge_length * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_edge_out, edge_length * sizeof(int)));
  CUDA_CHECK(
      cudaMalloc(&d_pooling_table, pooling_table_length * sizeof(DataType)));

  CUDA_CHECK(cudaMemcpy(d_emb_table, emb_table_cpu,
                        emb_table_length * sizeof(DataType),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_edge_in, edge_in_cpu, edge_length * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_edge_out, edge_out_cpu, edge_length * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemset(d_pooling_table, 0, pooling_table_length * sizeof(DataType)));

  const dim3 blockDim(BLOCK_SIZE);
  const dim3 gridDim(78 * 4);
  std::cout << "Grid Dim: " << gridDim.x << ", Block Dim: " << blockDim.x
            << std::endl;
  size_t smem_size = 2 * TILE_INDICES_VAL * sizeof(int) + 1 * sizeof(int);

  int *d_work_tile_idx;
  cudaMalloc(&d_work_tile_idx, sizeof(int));
  cudaMemset(d_work_tile_idx, 0, sizeof(int));
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  gpu_pooling_forward_async_kernel<DataType, TILE_INDICES_VAL, BLOCK_SIZE>
      <<<gridDim, blockDim, smem_size>>>(d_emb_table, d_edge_in, d_edge_out,
                                         d_pooling_table, emb_dim, edge_length,
                                         d_work_tile_idx);

  CUDA_CHECK(cudaDeviceSynchronize());

  // int num_runs = 100;

  // CUDA_CHECK(cudaEventRecord(start));
  // for (int i = 0; i < num_runs; ++i) {
  //   cudaMemset(d_work_tile_idx, 0, sizeof(int));
  //   gpu_pooling_forward_async_kernel<DataType, TILE_INDICES_VAL, BLOCK_SIZE>
  //       <<<gridDim, blockDim, smem_size>>>(d_emb_table, d_edge_in,
  //       d_edge_out,
  //                                          d_pooling_table, emb_dim,
  //                                          edge_length, d_work_tile_idx);
  // }
  // CUDA_CHECK(cudaEventRecord(stop));

  // CUDA_CHECK(cudaEventSynchronize(stop));
  // float total_time = 0;
  // CUDA_CHECK(cudaEventElapsedTime(&total_time, start, stop));

  // float average_time_ms = total_time / num_runs;
  // std::cout << "\n--- Performance Results ---" << std::endl;
  // std::cout << "Number of test runs: " << num_runs << std::endl;
  // std::cout << "Average kernel execution time: " << average_time_ms << " ms"
  //           << std::endl;

  // std::cout << "\n--- Verification ---" << std::endl;

  // std::cout
  //     << "Resetting GPU buffer and running kernel once for verification..."
  //     << std::endl;
  // CUDA_CHECK(
  //     cudaMemset(d_pooling_table, 0, pooling_table_length *
  //     sizeof(DataType)));

  // cudaMemset(d_work_tile_idx, 0, sizeof(int));
  // gpu_pooling_forward_async_kernel<DataType, TILE_INDICES_VAL, BLOCK_SIZE>
  //     <<<gridDim, blockDim, smem_size>>>(d_emb_table, d_edge_in, d_edge_out,
  //                                        d_pooling_table, emb_dim,
  //                                        edge_length, d_work_tile_idx);
  // CUDA_CHECK(cudaDeviceSynchronize());  // 确保内核执行完毕

  // std::vector<DataType> h_gpu_result(pooling_table_length);
  // CUDA_CHECK(cudaMemcpy(h_gpu_result.data(), d_pooling_table,
  //                       h_gpu_result.size() * sizeof(DataType),
  //                       cudaMemcpyDeviceToHost));
  // memset(pooling_table_cpu, 0, pooling_table_length * sizeof(DataType));

  // std::cout << "Calculating reference result on CPU..." << std::endl;
  // for (int i = 0; i < edge_length; ++i) {
  //   int in_node = edge_in_cpu[i];
  //   int out_node = edge_out_cpu[i];
  //   for (int64_t d = 0; d < emb_dim; ++d) {
  //     pooling_table_cpu[out_node * emb_dim + d] +=
  //         emb_table_cpu[in_node * emb_dim + d];
  //   }
  // }

  // double total_absolute_error = 0.0;
  // for (size_t i = 0; i < pooling_table_length; ++i) {
  //   total_absolute_error += std::abs(pooling_table_cpu[i] - h_gpu_result[i]);
  // }

  // std::cout << "Total absolute error between CPU and GPU: "
  //           << total_absolute_error << std::endl;
  // if (total_absolute_error < 1e-1) {
  //   std::cout << "Result verification PASSED." << std::endl;
  // } else {
  //   std::cout << "Result verification FAILED." << std::endl;
  // }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_emb_table));
  CUDA_CHECK(cudaFree(d_edge_in));
  CUDA_CHECK(cudaFree(d_edge_out));
  CUDA_CHECK(cudaFree(d_pooling_table));

  return 0;
}
