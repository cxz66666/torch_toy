#include <iostream>
#include <vector>
#include <numeric>  // For std::iota
#include <cstdint>
#include <stdexcept>

#include <cuda_runtime.h>

// --- CUDA Error Checking Macro ---
// 这是一个标准的宏，用于捕获并报告CUDA API调用中的任何错误。
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// --- Your Kernel (Optimized Version) ---
// 已更名为 gpu_block_copy 并添加了 __restrict__ 关键字以获得最佳性能
template <typename DATA_TYPE>
__global__ void gpu_block_copy(const DATA_TYPE *A,
                               const int32_t *src_block_offset,
                               const int32_t *dst_block_offset,
                               const int32_t *block_copy_size, DATA_TYPE *B) {
  const int task_idx = blockIdx.x;  // 每个CUDA块处理一个拷贝任务

  // 从全局内存中读取此任务的元数据
  const int src_offset = src_block_offset[task_idx];
  const int dst_offset = dst_block_offset[task_idx];
  const int copy_size = block_copy_size[task_idx];

  // 如果任务大小为0，则什么也不做
  if (copy_size <= 0) {
    return;
  }

  // 为当前任务创建基地址指针
  const DATA_TYPE *p_src = A + src_offset;
  DATA_TYPE *p_dst = B + dst_offset;

  // 使用Grid-Stride循环，让块内所有线程协作完成拷贝
  const int tid = threadIdx.x;
  const int stride = blockDim.x;

  for (int i = tid; i < copy_size; i += stride) {
    p_dst[i] = __ldg(p_src + i);
  }
}

// --- Main Test Function ---
int main() {
  // 1. --- Test Configuration ---
  const int num_tasks = 2554;  // 我们模拟8个独立的拷贝任务
  using DataType = float;      // 我们以float类型为例进行测试

  std::cout << "Starting test with " << num_tasks << " copy tasks."
            << std::endl;

  // 2. --- Host Data Generation ---
  // 在CPU上创建元数据向量
  std::vector<int32_t> h_src_block_offset(num_tasks);
  std::vector<int32_t> h_dst_block_offset(num_tasks);
  std::vector<int32_t> h_block_copy_size(num_tasks);

  int64_t total_elements = 0;

  std::cout << "Generating task metadata..." << std::endl;
  // 生成每个任务的元数据，并计算总共需要多少元素
  for (int i = 0; i < num_tasks; ++i) {
    // 让每个任务的拷贝大小不同，以增加测试的复杂性
    int32_t current_copy_size = 32 * 1024 + (i % 5) * 123;

    h_block_copy_size[i] = current_copy_size;
    h_src_block_offset[i] = total_elements;
    h_dst_block_offset[i] = total_elements;

    total_elements += current_copy_size;
  }
  std::cout << "Total elements to allocate: " << total_elements << std::endl;

  // 根据总元素数量，创建源数据和用于接收结果的向量
  std::vector<DataType> h_A(total_elements);
  std::vector<DataType> h_B_result(total_elements, 0.0f);  // 初始化为0

  // 使用 0, 1, 2, 3, ... 填充源数据，便于验证
  std::iota(h_A.begin(), h_A.end(), 0.0f);
  std::cout << "Host data generated." << std::endl;

  // 3. --- Device Memory Allocation ---
  DataType *d_A, *d_B;
  int32_t *d_src_block_offset, *d_dst_block_offset, *d_block_copy_size;

  CUDA_CHECK(cudaMalloc(&d_A, total_elements * sizeof(DataType)));
  CUDA_CHECK(cudaMalloc(&d_B, total_elements * sizeof(DataType)));
  CUDA_CHECK(cudaMalloc(&d_src_block_offset, num_tasks * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_dst_block_offset, num_tasks * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_block_copy_size, num_tasks * sizeof(int32_t)));
  std::cout << "Device memory allocated." << std::endl;

  // 4. --- Host to Device Data Transfer ---
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), total_elements * sizeof(DataType),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_src_block_offset, h_src_block_offset.data(),
                        num_tasks * sizeof(int32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dst_block_offset, h_dst_block_offset.data(),
                        num_tasks * sizeof(int32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_block_copy_size, h_block_copy_size.data(),
                        num_tasks * sizeof(int32_t), cudaMemcpyHostToDevice));
  std::cout << "Data transferred from Host to Device." << std::endl;

  // 5. --- Kernel Launch ---
  const int block_dim = 256;       // 每个块使用256个线程，这是一个常用的大小
  const int grid_dim = num_tasks;  // 网格大小等于任务数量，每个块处理一个任务

  std::cout << "Launching kernel with gridDim=" << grid_dim
            << ", blockDim=" << block_dim << "..." << std::endl;
  gpu_block_copy<DataType><<<grid_dim, block_dim>>>(
      d_A, d_src_block_offset, d_dst_block_offset, d_block_copy_size, d_B);

  // 等待Kernel执行完毕
  CUDA_CHECK(cudaDeviceSynchronize());
  std::cout << "Kernel execution finished." << std::endl;

  // 6. --- Device to Host Data Transfer ---
  CUDA_CHECK(cudaMemcpy(h_B_result.data(), d_B,
                        total_elements * sizeof(DataType),
                        cudaMemcpyDeviceToHost));
  std::cout << "Results transferred from Device to Host." << std::endl;

  // 7. --- Verification ---
  std::cout << "Verifying results..." << std::endl;
  bool success = true;
  for (int64_t i = 0; i < total_elements; ++i) {
    // 由于我们设置的源和目标布局相同，所以可以直接比较 h_A 和 h_B_result
    if (std::abs(h_A[i] - h_B_result[i]) > 1e-6) {
      std::cerr << "Verification FAILED at index " << i << "! "
                << "Expected: " << h_A[i] << ", Got: " << h_B_result[i]
                << std::endl;
      success = false;
      break;
    }
  }

  if (success) {
    std::cout << "Verification PASSED! The result is correct." << std::endl;
  }

  // 8. --- Cleanup ---
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_src_block_offset));
  CUDA_CHECK(cudaFree(d_dst_block_offset));
  CUDA_CHECK(cudaFree(d_block_copy_size));
  std::cout << "Device memory freed." << std::endl;

  return 0;
}