#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <type_traits>

// /usr/local/cuda/bin/nvcc   -gencode arch=compute_80,code=sm_80  --extended-lambda copy_reduce.cu -lineinfo
// sudo /usr/local/cuda/bin/ncu --print-source cuda,sass --import-source 1 --page source -o ncu_report%i --set full ./a.out

// ==========================================
// Part 1: NCCL 基础定义与 PTX 封装 (Mock)
// ==========================================

#define WARP_SIZE 32

// 模拟 NCCL 的 BytePack，用于强制 128-bit 访问
template<int Bytes>
struct alignas(Bytes) BytePack {
    uint64_t storage[Bytes / 8];
};

// 特化：处理 Bytes < 8 的情况 (虽然 float 是 4 字节，但 NCCL 这里的 BytePack 主要是为了 128bit)
template<>
struct alignas(4) BytePack<4> {
    uint32_t storage[1];
};

// 获取 Global 地址的 PTX
__device__ __forceinline__ uintptr_t cvta_to_global(void* ptr) {
    return  (uintptr_t)__cvta_generic_to_global(ptr);
}

// Volatile Load (Bypass L1)
template<int Bytes, typename T>
__device__ __forceinline__ T ld_volatile_global(uintptr_t addr) {
    T val;
    if constexpr (Bytes == 16) {
        uint4* p = reinterpret_cast<uint4*>(&val);
        asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
            : "=r"(p->x), "=r"(p->y), "=r"(p->z), "=r"(p->w) : "l"(addr));
    } else if constexpr (Bytes == 4) {
        uint32_t* p = reinterpret_cast<uint32_t*>(&val);
        asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(*p) : "l"(addr));
    }
    return val;
}

// Regular Store
template<int Bytes, typename T>
__device__ __forceinline__ void st_global(uintptr_t addr, T val) {
    if constexpr (Bytes == 16) {
        uint4* p = reinterpret_cast<uint4*>(&val);
        asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};"
            :: "l"(addr), "r"(p->x), "r"(p->y), "r"(p->z), "r"(p->w));
    } else if constexpr (Bytes == 4) {
        uint32_t* p = reinterpret_cast<uint32_t*>(&val);
        asm volatile("st.global.u32 [%0], %1;" :: "l"(addr), "r"(*p));
    }
}

// 占位符：模拟 Multimem 加载 (本地复现暂不使用 Multimem 硬件特性)
template<typename RedFn, int BytePerPack>
__device__ __forceinline__ BytePack<BytePerPack> applyLoadMultimem(RedFn fn, uintptr_t addr) {
    return ld_volatile_global<BytePerPack, BytePack<BytePerPack>>(addr);
}

// 占位符：模拟 Multimem 存储
template<typename T>
__device__ __forceinline__ void multimem_st_global(uintptr_t addr, T val) {
    // Fallback to normal store
    st_global<sizeof(T)>(addr, val);
}

// ==========================================
// Part 2: 算子定义 (Float Sum)
// ==========================================

// 基础加法算子
struct FuncSum {
    __device__ __forceinline__ float operator()(float a, float b) const {
        return a + b;
    }
};

// 模拟 NCCL 的 applyPreOp (空操作)
template<typename T>
__device__ __forceinline__ T applyPreOp(FuncSum fn, T val) {
    return val;
}

// 模拟 NCCL 的 applyPostOp (空操作)
template<typename T>
__device__ __forceinline__ T applyPostOp(FuncSum fn, T val) {
    return val;
}

// 核心：applyReduce 针对 BytePack<16> 的特化 (Type Punning)
// 这里展示了如何把 128-bit 寄存器当做 4 个 float 进行计算
template<typename RedFn>
__device__ __forceinline__ BytePack<16> applyReduce(RedFn fn, BytePack<16> acc, BytePack<16> val) {
    BytePack<16> res;
    float* acc_f = reinterpret_cast<float*>(&acc);
    float* val_f = reinterpret_cast<float*>(&val);
    float* res_f = reinterpret_cast<float*>(&res);

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        res_f[i] = fn(acc_f[i], val_f[i]);
    }
    return res;
}

// 针对 BytePack<4> 的特化
template<typename RedFn>
__device__ __forceinline__ BytePack<4> applyReduce(RedFn fn, BytePack<4> acc, BytePack<4> val) {
    BytePack<4> res;
    float* acc_f = reinterpret_cast<float*>(&acc);
    float* val_f = reinterpret_cast<float*>(&val);
    float* res_f = reinterpret_cast<float*>(&res);
    res_f[0] = fn(acc_f[0], val_f[0]);
    return res;
}

// Helper for template logic
template<typename RedFn> struct LoadMultimem_BigPackSize { static constexpr int BigPackSize = 16; };


// ==========================================
// Part 3: 用户提供的 Kernel 源码
// (完全保留原逻辑，仅添加必要的模板适配)
// ==========================================

template<typename RedFn, typename T, int Unroll, int BytePerPack,
         int MultimemSrcs, int MinSrcs, int MaxSrcs,
         int MultimemDsts, int MinDsts, int MaxDsts, int PreOpSrcs,
         typename IntBytes, typename SrcPtrFn, typename DstPtrFn>
__device__ __forceinline__ void reduceCopyPacks(
    int nThreads, int &thread,
    uint64_t redArg, uint64_t *preOpArgs, bool postOp,
    int nSrcs, SrcPtrFn const &srcPtrFn, int nDsts, DstPtrFn const &dstPtrFn,
    IntBytes &nBytesBehind, IntBytes &nBytesAhead
  ) {
  static_assert(std::is_signed<IntBytes>::value, "IntBytes must be a signed integral type.");
  if (BytePerPack == 0) asm("trap;");

  constexpr int BytePerHunk = Unroll*WARP_SIZE*BytePerPack;
  int nWarps = nThreads/WARP_SIZE;
  int warp = thread/WARP_SIZE;
  int lane = thread%WARP_SIZE;

  IntBytes threadBytesBehind = nBytesBehind + (warp*BytePerHunk + lane*BytePerPack);
  IntBytes threadBytesAhead = nBytesAhead - (warp*BytePerHunk + lane*BytePerPack);
  IntBytes nHunksAhead = nBytesAhead/(BytePerHunk + !BytePerHunk);
  nBytesBehind += nHunksAhead*BytePerHunk;
  nBytesAhead -= nHunksAhead*BytePerHunk;
  if (Unroll==1 && BytePerPack <= nBytesAhead) {
    nHunksAhead += 1;
    nBytesBehind += nBytesAhead - (nBytesAhead%(BytePerPack + !BytePerPack));
    nBytesAhead = nBytesAhead%(BytePerPack + !BytePerPack);
  }
  nHunksAhead -= warp;

  RedFn redFn; // redArg is dummy for FuncSum
  uintptr_t minSrcs[MinSrcs + !MinSrcs];
  uintptr_t minDsts[MinDsts + !MinDsts];
  #pragma unroll
  for (int s=0; s < MinSrcs; s++) {
    minSrcs[s] = cvta_to_global(srcPtrFn(s)) + threadBytesBehind;
  }
  #pragma unroll
  for (int d=0; d < MinDsts; d++) {
    minDsts[d] = cvta_to_global(dstPtrFn(d)) + threadBytesBehind;
  }

  while (Unroll==1 ? (BytePerPack <= threadBytesAhead) : (0 < nHunksAhead)) {
    BytePack<BytePerPack> acc[Unroll];

    { RedFn preFn; 
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        if (0 < MultimemSrcs) {
           // Placeholder
        } else {
          acc[u] = ld_volatile_global<BytePerPack, BytePack<BytePerPack>>(minSrcs[0]);
          if (0 < PreOpSrcs) acc[u] = applyPreOp(preFn, acc[u]);
        }
        minSrcs[0] += WARP_SIZE*BytePerPack;
      }
    }

    #pragma unroll (MinSrcs-1 + !(MinSrcs-1))
    for (int s=1; s < MinSrcs; s++) {
      BytePack<BytePerPack> tmp[Unroll];
      RedFn preFn;
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        tmp[u] = ld_volatile_global<BytePerPack, BytePack<BytePerPack>>(minSrcs[s]);
        minSrcs[s] += WARP_SIZE*BytePerPack;
      }
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        if (s < PreOpSrcs) tmp[u] = applyPreOp(preFn, tmp[u]);
        acc[u] = applyReduce(redFn, acc[u], tmp[u]);
      }
    }

    // Dynamic remainder loop
    for (int s=MinSrcs; (MinSrcs < MaxSrcs) && (s < MaxSrcs) && (s < nSrcs); s++) {
      uintptr_t src = cvta_to_global(srcPtrFn(s)) + threadBytesBehind;
      BytePack<BytePerPack> tmp[Unroll];
      RedFn preFn;
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        tmp[u] = ld_volatile_global<BytePerPack, BytePack<BytePerPack>>(src);
        src += WARP_SIZE*BytePerPack;
      }
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        if (s < PreOpSrcs) tmp[u] = applyPreOp(preFn, tmp[u]);
        acc[u] = applyReduce(redFn, acc[u], tmp[u]);
      }
    }

    if (postOp) {
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++)
        acc[u] = applyPostOp(redFn, acc[u]);
    }

    #pragma unroll (MinDsts + !MinDsts)
    for (int d=0; d < MinDsts; d++) {
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
         st_global<BytePerPack>(minDsts[d], acc[u]);
         minDsts[d] += WARP_SIZE*BytePerPack;
      }
    }
    // Dynamic dst loop omitted for brevity in repro, assuming MinDsts covers it or nDsts is small
    
    nWarps = nThreads/WARP_SIZE;
    #pragma unroll
    for (int s=0; s < MinSrcs; s++) minSrcs[s] += (nWarps-1)*BytePerHunk;
    #pragma unroll
    for (int d=0; d < MinDsts; d++) minDsts[d] += (nWarps-1)*BytePerHunk;
    threadBytesBehind += nWarps*BytePerHunk;
    threadBytesAhead -= nWarps*BytePerHunk;
    nHunksAhead -= nWarps;
  }

  nWarps = nThreads/WARP_SIZE;
  warp = thread/WARP_SIZE;
  lane = thread%WARP_SIZE;
  if (Unroll==1 && nHunksAhead > 0) nHunksAhead -= nWarps;
  warp = -nHunksAhead;
  thread = warp*WARP_SIZE + lane;
}

template<int Unroll, typename RedFn, typename T,
         int MultimemSrcs, int MinSrcs, int MaxSrcs,
         int MultimemDsts, int MinDsts, int MaxDsts, int PreOpSrcs,
         typename IntBytes, typename SrcPtrFn, typename DstPtrFn>
__device__ __forceinline__ void reduceCopy(
    int thread, int nThreads,
    uint64_t redArg, uint64_t *preOpArgs, bool postOp,
    int nSrcs, SrcPtrFn const &srcPtrFn, int nDsts, DstPtrFn const &dstPtrFn,
    IntBytes nElts
  ) {
  int lane = thread%WARP_SIZE;
  constexpr int BigPackSize = (MultimemSrcs == 0) ? 16 : LoadMultimem_BigPackSize<RedFn>::BigPackSize;

  if (MaxDsts==0) return;
  
  IntBytes nBytesBehind = 0;
  IntBytes nBytesAhead = nElts*sizeof(T);

  if constexpr (BigPackSize > sizeof(T)) {
    bool aligned = true;
    if (lane < nSrcs) aligned &= 0 == cvta_to_global(srcPtrFn(lane)) % (BigPackSize + !BigPackSize);
    if (lane < nDsts) aligned &= 0 == cvta_to_global(dstPtrFn(lane)) % (BigPackSize + !BigPackSize);
    aligned = __all_sync(~0u, aligned);
    if (aligned) {
      reduceCopyPacks<RedFn, T, Unroll, BigPackSize,
        MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
        (nThreads, thread, redArg, preOpArgs, postOp,
         nSrcs, srcPtrFn, nDsts, dstPtrFn, nBytesBehind, nBytesAhead);
      if (nBytesAhead == 0) return;
      
      reduceCopyPacks<RedFn, T, 1, BigPackSize,
        MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
        (nThreads, thread, redArg, preOpArgs, postOp,
         nSrcs, srcPtrFn, nDsts, dstPtrFn, nBytesBehind, nBytesAhead);
      if (nBytesAhead == 0) return;
    }
  }
  
  // Fallback path
  reduceCopyPacks<RedFn, T, Unroll*(16/sizeof(T))/2, sizeof(T),
    MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
    (nThreads, thread, redArg, preOpArgs, postOp,
     nSrcs, srcPtrFn, nDsts, dstPtrFn, nBytesBehind, nBytesAhead);
     
  if (nBytesAhead == 0) return;

  reduceCopyPacks<RedFn, T, 1, sizeof(T),
    MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
    (nThreads, thread, redArg, preOpArgs, postOp,
     nSrcs, srcPtrFn, nDsts, dstPtrFn, nBytesBehind, nBytesAhead);
}


// ==========================================
// Part 4: Wrapper Kernel (入口函数)
// ==========================================

#define UNROLL_FACTOR 4 
// 假设2个源进行reduce，写入1个目的
#define MIN_SRCS 1 
#define MAX_SRCS 2
#define MIN_DSTS 1
#define MAX_DSTS 1

__global__ void benchmarkKernel(
    float* src0, float* src1, float* dst, 
    int64_t nElts) 
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int nThreads = blockDim.x * gridDim.x;

    // 构建指针获取 Functor
    void* srcs[2] = {src0, src1};
    void* dsts[1] = {dst};
    
    auto srcPtrFn = [=]__device__(int i) { return srcs[i]; };
    auto dstPtrFn = [=]__device__(int i) { return dsts[i]; };

    // 调用核心函数
    // 模板参数：Unroll=4, T=float, Multimem=0 (off), Srcs/Dsts 设定如上
    reduceCopy<UNROLL_FACTOR, FuncSum, float, 
               0, MIN_SRCS, MAX_SRCS, 
               0, MIN_DSTS, MAX_DSTS, 0, 
               int64_t>
               (tid, nThreads, 0, nullptr, false, 
                2, srcPtrFn, 1, dstPtrFn, nElts);
}

// ==========================================
// Part 5: Host 测试代码
// ==========================================

#include <iostream>

#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

int main() {
    // 1. 配置参数
    // 1G elements = 1024^3 * 4 bytes = 4GB per buffer
    // Total VRAM usage = 4GB * 3 = 12GB
    size_t nElts = 1UL << 30; 
    size_t bytes = nElts * sizeof(float);
    int nIters = 1; // 重复 100 次

    int deviceId = 0;
    CHECK_CUDA(cudaSetDevice(deviceId));
    
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, deviceId));
    printf("Device: %s\n", prop.name);
    printf("Element Count: %lu (%.2f GB per buffer)\n", nElts, bytes / 1e9);
    printf("Total VRAM required: %.2f GB\n", (bytes * 3) / 1e9);

    // 2. 显存分配
    float *d_src0, *d_src1, *d_dst;
    CHECK_CUDA(cudaMalloc(&d_src0, bytes));
    CHECK_CUDA(cudaMalloc(&d_src1, bytes));
    CHECK_CUDA(cudaMalloc(&d_dst, bytes));

    // (可选) 数据初始化，避免 NaN，虽然 benchmark 只看带宽不在乎结果
    CHECK_CUDA(cudaMemset(d_src0, 0, bytes));
    CHECK_CUDA(cudaMemset(d_src1, 0, bytes));

    // 3. Grid 配置
    int threadsPerBlock = 1024;
    int numBlocks = 1; 

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    printf("Starting Benchmark...\n");
    printf("Config: Unroll=%d, Vectorized=128bit, Iterations=%d\n", UNROLL_FACTOR, nIters);

    // 4. 性能测试循环
    // 确保之前的操作已完成
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < nIters; ++i) {
        benchmarkKernel<<<numBlocks, threadsPerBlock>>>(d_src0, d_src1, d_dst, nElts);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // 5. 结果计算
    float total_time_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&total_time_ms, start, stop));

    float avg_time_ms = total_time_ms / nIters;

    // Bandwidth: (Read A + Read B + Write C) = 3x bytes
    double totalBytesPerIter = 3.0 * bytes; 
    double gbps = (totalBytesPerIter / 1e9) / (avg_time_ms / 1000.0);

    printf("------------------------------------------------\n");
    printf("Total Time (100 iters): %.3f ms\n", total_time_ms);
    printf("Avg Time per Iter:      %.3f ms\n", avg_time_ms);
    printf("Effective Bandwidth:    %.2f GB/s\n", gbps);
    printf("------------------------------------------------\n");

    CHECK_CUDA(cudaFree(d_src0));
    CHECK_CUDA(cudaFree(d_src1));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}