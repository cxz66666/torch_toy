#include <iostream>
#include <cuda_runtime.h>

// ----------------------------------------------------------------
// 1. 模拟环境与辅助函数
// ----------------------------------------------------------------
#define CHECK(call) { const cudaError_t error = call; if (error != cudaSuccess) { printf("Error: %s:%d, %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); exit(1); } }

// 模拟 C++ atomicLoad(Relaxed) - 编译为普通 LDG，不带 Acquire，不清空 L1
__device__ __forceinline__ unsigned int atomicLoadRelaxed(unsigned int* addr) {
    unsigned int val;
    asm volatile("ld.global.b32 %0, [%1];" : "=r"(val) : "l"(addr) : "memory");
    return val;
}

// 模拟 C++ atomicStore(Relaxed) - 编译为普通 STG，不带 Release
__device__ __forceinline__ void atomicStoreRelaxed(unsigned int* addr, unsigned int val) {
    asm volatile("st.b32 [%0], %1;" :: "l"(addr), "r"(val) : "memory");
}

// 辅助：读取诱饵，强制触发 L1 填充
__device__ __forceinline__ int read_bait(int* ptr) {
    int val;
    // 使用 .ca (Cache All) 确保进 L1
    asm volatile("ld.global.ca.b32 %0, [%1];" : "=r"(val) : "l"(ptr) : "memory");
    return val;
}

// ----------------------------------------------------------------
// 2. 你的 SYNC 函数 (复刻版)
// ----------------------------------------------------------------
// 全局同步变量 (在 main 中分配)
struct SyncVars {
    unsigned int count; // count_
    unsigned int flag;  // flag_
};

// 你的原始 sync 逻辑适配
__device__ void user_sync(SyncVars* vars, unsigned int* preFlag_local, int blockNum) {
    unsigned int maxOldCnt = blockNum - 1;
    
    __syncthreads(); // 块内同步
    
    if (blockNum == 1) return;
    __threadfence();
    if (threadIdx.x == 0) {
        __threadfence(); 
        unsigned int tmp = (*preFlag_local) ^ 1;
        unsigned int oldVal = atomicAdd(&vars->count, 1);
        
        bool isLast = (oldVal == maxOldCnt); 

        if (isLast) {
            atomicStoreRelaxed(&vars->flag, tmp);
            
            // 重置 count 以便下次使用 (模拟 atomicInc 的 wrap 效果)
            atomicExch(&vars->count, 0); 
        } else {
            // [你的代码]: POLL_MAYBE_JAILBREAK...
            // Block 0 (Waiter) 轮询
            // 致命弱点：使用 Relaxed Load 轮询，成功后没有任何 Fence/Acquire
            while (atomicLoadRelaxed(&vars->flag) != tmp) {
                // spin
            }
        }
        *preFlag_local = tmp;
    }
    
    __syncthreads(); // 块内同步
}

__device__ void busy_wait(unsigned long long clocks) {
    unsigned long long start = clock64();
    while (clock64() - start < clocks);
}

// ----------------------------------------------------------------
// 3. 陷阱 Kernel
// ----------------------------------------------------------------
__global__ void trap_kernel(int* data_line, SyncVars* sync_vars, volatile int* setup_handshake) {
    // 线程局部状态
    unsigned int preFlag = 0; 
    
    // 我们只需要两个 Block，每个 Block 只需要一个线程参与逻辑
    int tid = threadIdx.x;
    if (tid > 0) return; // 简化：只看主线程

    // ==========================================
    // 场景布置
    // ==========================================
    if (blockIdx.x == 0) {
        // [Block 0 - 受害者]
        
        // 1. 吞下诱饵 (Bait)
        // 读取 data_line[1]。
        // 因为 data_line[0] 和 [1] 在同一个 128B Cache Line 上，
        // 这一步会把 data_line[0] (值为0) 顺便加载进 Block 0 的 L1 Cache。
        read_bait(&data_line[1]);

        // 2. 告诉 Block 1 我准备好了 (Cache 已经脏了)
        *setup_handshake = 1; 
        __threadfence_system(); // 确保 handshake 可见

        // 3. 调用你的 SYNC 函数
        // 等待 Block 1 修改数据

        user_sync(sync_vars, &preFlag, 2); 
        // 4. 读取数据 (这里会触发陷阱)
        // 此时，Block 1 已经把 data_line[0] 改成了 999。
        // 但是，user_sync 里的 atomicLoad 是 Relaxed 的，且没有后续 Acquire。
        // 所以 Block 0 的 L1 Cache 依然认为 data_line[0] 是 Valid 的（值为0）。
        int val = data_line[0]; // 普通读取，命中 L1

        // 打印结果
        if (val != 999) {
            printf("[Block 0] FAILED! read data[0] = %d (Expected 999). Stale L1 value!\n", val);
        } else {
            printf("[Block 0] SUCCESS! read data[0] = %d. (Cache trap didn't trigger?)\n", val);
        }

    } else if (blockIdx.x == 1) {
        // [Block 1 - 写入者]

        // 1. 等待 Block 0 吞下诱饵
        while (*setup_handshake != 1);

        // 延迟一下确保 Block 0 Cache 状态稳定
        busy_wait(1000000);

        // 2. 修改数据
        data_line[0] = 999;
        
        // 3. 强屏障，推送到 L2
        __threadfence(); 

        // 4. 调用你的 SYNC 函数 (通知 Block 0)
        user_sync(sync_vars, &preFlag, 2);
    }
}

int main() {
    int* d_data;
    SyncVars* d_sync_vars;
    int* d_handshake;

    // 分配
    cudaMalloc(&d_data, 128 * sizeof(int)); // 128 int = 512 bytes, 覆盖多个 cache lines
    cudaMalloc(&d_sync_vars, sizeof(SyncVars));
    cudaMalloc(&d_handshake, sizeof(int));

    // 初始化
    cudaMemset(d_data, 0, 128 * sizeof(int));
    cudaMemset(d_sync_vars, 0, sizeof(SyncVars));
    cudaMemset(d_handshake, 0, sizeof(int));

    printf("Running User-Sync Implicit Cache Trap...\n");
    printf("Setup: Block 0 caches '0', Block 1 writes '999'.\n");
    printf("Result depends on if User-Sync invalidates L1.\n");
    printf("------------------------------------------------\n");

    // 启动 2 个 Block，每个 32 线程 (warp)
    trap_kernel<<<2, 32>>>(d_data, d_sync_vars, d_handshake);
    CHECK(cudaDeviceSynchronize());
    
    printf("------------------------------------------------\n");
    printf("Experiment Finished.\n");

    cudaFree(d_data);
    cudaFree(d_sync_vars);
    cudaFree(d_handshake);
    return 0;
}