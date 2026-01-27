#include <stdlib.h>
#include <iostream>
#include <cuda/atomic> // 引入 CUDA 标准库实现的 atomic

// -------------------------------------------------------------------------
// 使用 cuda::atomic_ref 实现 DeviceSyncer
// -------------------------------------------------------------------------
namespace mscclpp {
struct DeviceSyncer {
  unsigned int count;
  unsigned int turn;

  __device__ __forceinline__ void sync(int num_blocks) {
    __syncthreads(); // 块内同步
    if (threadIdx.x == 0) {
      // 创建 atomic_ref，指定作用域为 Device (跨 Block 可见)
      // 这等价于 C++20 的 std::atomic_ref 但增加了 GPU 作用域控制
      cuda::atomic_ref<unsigned int, cuda::thread_scope_device> ref_count(count);
      cuda::atomic_ref<unsigned int, cuda::thread_scope_device> ref_turn(turn);

      unsigned int prev_turn = ref_turn.load(cuda::std::memory_order_relaxed);
      unsigned int old_count = ref_count.fetch_add(1, cuda::std::memory_order_release);

      if (old_count == num_blocks - 1) {
        // 最后一个到达的 Block
        ref_count.store(0, cuda::std::memory_order_relaxed);
        // 更新轮次，Release 语义确保前面的 store 对其他人可见
        ref_turn.fetch_add(1, cuda::std::memory_order_release);
      }
      // 注意 这里要每个block都至少load一次，因为需要执行一下acquire语义的invalid L1操作
      while (ref_turn.load(cuda::std::memory_order_acquire) == prev_turn);

    }
    
    __syncthreads(); // 块内同步
  }

  __device__ __forceinline__ void sync2(int num_blocks) {
    __syncthreads(); // 块内同步
    if (threadIdx.x == 0) {
      __threadfence();
    //   cuda::atomic_thread_fence(cuda::memory_order_relaxed, cuda::thread_scope_device);

      cuda::atomic_ref<unsigned int, cuda::thread_scope_device> ref_count(count);
      cuda::atomic_ref<unsigned int, cuda::thread_scope_device> ref_turn(turn);

      unsigned int prev_turn = ref_turn.load(cuda::std::memory_order_relaxed);
      unsigned int old_count = ref_count.fetch_add(1, cuda::std::memory_order_relaxed);

      if (old_count == num_blocks - 1) {
        // 最后一个到达的 Block
        ref_count.store(0, cuda::std::memory_order_relaxed);
        // 更新轮次，Release 语义确保前面的 store 对其他人可见
        ref_turn.fetch_add(1, cuda::std::memory_order_relaxed);
      }
      // 注意 这里要每个block都至少load一次，因为需要执行一下acquire语义的invalid L1操作
      while (ref_turn.load(cuda::std::memory_order_relaxed) == prev_turn);

    }
    
    __syncthreads(); // 块内同步
  }

};
} // namespace mscclpp

// -------------------------------------------------------------------------

#define HIDVA_CUDATHROW(cmd)                                                                                       \
  do {                                                                                                               \
    cudaError_t err = cmd;                                                                                           \
    if (err != cudaSuccess) {                                                                                        \
      std::cerr << #cmd << ":" << cudaGetErrorString(err) << std::endl;                                             \
      abort();                                                                                                       \
    }                                                                                                                \
  } while (false)

__device__ unsigned get_smid(void) {
  unsigned ret = 0;
  asm("mov.u32 %0, %smid;" : "=r"(ret));
  return ret;
}

__managed__ unsigned int bad_apple_cnt = 0;
#define SM2_VAR_EXPECTED 77
#define SM1_VAR_EXPECTED 33
unsigned int sm1_var_expected = SM1_VAR_EXPECTED;
unsigned int sm2_var_expected = SM2_VAR_EXPECTED;

__device__ mscclpp::DeviceSyncer dev_syncer;

__device__ unsigned int sm1_var = SM1_VAR_EXPECTED;
__device__ void store_sm1_var(unsigned int val) {
  asm("st.weak.global.wb.u32  [sm1_var], %0;" :: "r"(val));
}

__device__ unsigned int sm2_var = SM2_VAR_EXPECTED;
__device__ void store_sm2_var(unsigned int val) {
  asm("st.weak.global.wb.u32  [sm2_var], %0;" :: "r"(val));
}

__device__ unsigned load_sm1_var() {
  unsigned ret = 0;
  asm("ld.weak.global.ca.u32 %0, [sm1_var];" : "=r"(ret));
  return ret;
}

__device__ unsigned load_sm2_var() {
  unsigned ret = 0;
  asm("ld.weak.global.ca.u32 %0, [sm2_var];" : "=r"(ret));
  return ret;
}

__managed__ unsigned int smids[2] = {0, 0};

__global__ void zy_sync_test(){
  if (load_sm1_var() != SM1_VAR_EXPECTED) {
    __brkpt();
  }
  if (load_sm2_var() != SM2_VAR_EXPECTED) {
    __brkpt();
  }

  if (threadIdx.x == 0) {
    atomicExch(&smids[blockIdx.x], get_smid());
    if (blockIdx.x == 0) {
      store_sm1_var(SM2_VAR_EXPECTED);
    } else {
      store_sm2_var(SM1_VAR_EXPECTED);
    }
  }

  dev_syncer.sync2(gridDim.x);

  if (threadIdx.x == 0) {
    if (blockIdx.x == 0) {
      if (load_sm2_var() == SM2_VAR_EXPECTED) {
        atomicAdd(&bad_apple_cnt, 1);
      }
    } else {
      if (load_sm1_var() == SM1_VAR_EXPECTED) {
        atomicAdd(&bad_apple_cnt, 1);
      }
    }
  }
  __syncthreads();
}

int main(int argc, char** argv){
  int device;
  cudaDeviceProp prop;
  cudaSetDevice(0);
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);

  int max_sm = 0;
  int num_blks = 0;
  HIDVA_CUDATHROW(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blks,
      zy_sync_test,
      1024,
      max_sm));
  std::cout << "device=" << device << ",maxThreadsPerMultiProcessor=" << prop.maxThreadsPerMultiProcessor
            << ",sharedMemPerMultiprocessor=" << prop.sharedMemPerMultiprocessor
            << ",max_sm=" << max_sm
            << ",num_blks=" << num_blks << std::endl;

  mscclpp::DeviceSyncer syncer = {};
  HIDVA_CUDATHROW(cudaMemcpyToSymbol(dev_syncer, &syncer, sizeof(mscclpp::DeviceSyncer)));

  unsigned long i = 0;
  unsigned long same_sm = 0;
  while (true) {
    HIDVA_CUDATHROW(cudaMemcpyToSymbol(sm1_var, &sm1_var_expected, sizeof(sm1_var_expected)));
    HIDVA_CUDATHROW(cudaMemcpyToSymbol(sm2_var, &sm2_var_expected, sizeof(sm2_var_expected)));
    zy_sync_test<<<2, 1024, max_sm>>>();
    HIDVA_CUDATHROW(cudaDeviceSynchronize());
    HIDVA_CUDATHROW(cudaGetLastError());

    if (smids[0] == smids[1]) {
      same_sm += 1;
    }
    ++i;
    if (i % 10000 == 0) {
      std::cout << "Do it again!" << i << ",bad_apple=" << bad_apple_cnt << ",same_sm=" << same_sm << std::endl;
    }
  }

  return 0;
}