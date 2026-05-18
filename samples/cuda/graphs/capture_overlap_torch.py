import torch
from torch.utils.cpp_extension import load_inline

# CUDA_DEVICE_WAITS_ON_EXCEPTION=1 TRIGGER_ILLEGAL_ACCESS=1 python graph/capture_overlap_torch.py
src = {
    # =================================================================
    # 1. 纯 CUDA 代码：绝不包含 <torch/extension.h>
    #    只包含 <cuda_runtime.h>，NVCC 编译这个非常轻松
    # =================================================================
    "cuda": r"""
#include <cuda_runtime.h>

__global__ void computation_kernel(unsigned long long total_nanosec) {
    const unsigned long long max_sleep = 1'000'000;
    unsigned long long slept = 0;
    while (slept + max_sleep <= total_nanosec) {
        __nanosleep(max_sleep);
        slept += max_sleep;
    }
    if (slept < total_nanosec) {
        __nanosleep(total_nanosec - slept);
    }
}

__global__ void communication_kernel(unsigned long long total_nanosec) {
    const unsigned long long max_sleep = 1'000'000;
    unsigned long long slept = 0;
    while (slept + max_sleep <= total_nanosec) {
        __nanosleep(max_sleep);
        slept += max_sleep;
    }
    if (slept < total_nanosec) {
        __nanosleep(total_nanosec - slept);
    }
}

__global__ void illegal_access_kernel(float* data) {
    float *new_data = data;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Intentionally write to an invalid address.
        new_data[0] = 42.0f;
    }
}

// 定义启动函数 (Launcher)，接收原始的 cudaStream_t
// 注意：这里不要用 at::cuda::...，只用原生 CUDA 类型
void launch_computation_kernel(unsigned long long nanosec, cudaStream_t stream) {
    computation_kernel<<<1, 1, 0, stream>>>(nanosec);
}

void launch_communication_kernel(unsigned long long nanosec, cudaStream_t stream) {
    communication_kernel<<<1, 1, 0, stream>>>(nanosec);
}

void launch_illegal_access_kernel(cudaStream_t stream) {
    illegal_access_kernel<<<1, 1, 0, stream>>>(nullptr);
}
""",

    # =================================================================
    # 2. C++ 桥接代码：引入 Torch，负责处理 Python 交互和流获取
    #    这部分由 GCC 编译，它能看懂复杂的 C++ 标准库
    # =================================================================
    "cpp": r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

// 声明外部的 CUDA 启动函数 (在上面的 cuda 字符串中定义)
void launch_computation_kernel(unsigned long long nanosec, cudaStream_t stream);
void launch_communication_kernel(unsigned long long nanosec, cudaStream_t stream);
void launch_illegal_access_kernel(cudaStream_t stream);

// 暴露给 Python 的 wrapper
void computation(unsigned long long nanosec) {
    // 在这里 (GCC端) 安全地调用 PyTorch API 获取流
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    // 传给 CUDA 端
    launch_computation_kernel(nanosec, stream);
}

void communication(unsigned long long nanosec) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launch_communication_kernel(nanosec, stream);
}

void illegal_access() {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launch_illegal_access_kernel(stream);
}
"""
}
def build_extension():
    return load_inline(
        name="nanosleep_ext",
        cpp_sources=src["cpp"],
        cuda_sources=src["cuda"],
        functions=["computation", "communication", "illegal_access"],
        extra_cuda_cflags=[
            "-g",
            "-G",
        ],
        verbose=True,
        with_cuda=True,
    )


def one_giant_batch(nanosleep_module, nanosec: int):
    nanosleep_module.computation(nanosec)
    nanosleep_module.communication(nanosec)
    nanosleep_module.computation(nanosec)
    nanosleep_module.communication(nanosec)


def microbatch_overlapping(nanosleep_module, stream1, stream2, nanosec: int):
    # attach stream1 and steam2 to the current stream
    stream = torch.cuda.current_stream()
    event = torch.cuda.Event()
    event.record(stream)
    stream1.wait_event(event)
    stream2.wait_event(event)

    # stream 2 will wait for stream1
    stream1_event = None

    funcs = [
        nanosleep_module.computation,
        nanosleep_module.communication,
        nanosleep_module.computation,
        nanosleep_module.communication,
    ]

    for func in funcs:
        with torch.cuda.stream(stream1):
            func(nanosec)
            stream1_event = torch.cuda.Event()
            stream1_event.record(stream1)

        with torch.cuda.stream(stream2):
            stream2.wait_event(stream1_event)
            func(nanosec)

    # sync the streams back to the main stream
    e = torch.cuda.Event()
    e.record(stream1)
    stream.wait_event(e)

    e = torch.cuda.Event()
    e.record(stream2)
    stream.wait_event(e)

def main():
    import argparse
    import os
    import time

    parser = argparse.ArgumentParser(description="CUDA graph overlap sample with two streams.")
    parser.add_argument("--nanosec", type=int, default=200_000_000)
    parser.add_argument("--trigger-illegal-access", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This sample requires CUDA.")

    nanosleep_module = build_extension()
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    one_giant_batch(nanosleep_module, args.nanosec)
    torch.cuda.current_stream().synchronize()

    with torch.cuda.nvtx.range("one_giant_batch"):
        start = time.time()
        one_giant_batch(nanosleep_module, args.nanosec)
        torch.cuda.current_stream().synchronize()
        end = time.time()
        print(f"one_giant_batch takes: {end - start:.3f} sec")

    microbatch_overlapping(nanosleep_module, stream1, stream2, args.nanosec)
    torch.cuda.current_stream().synchronize()
    with torch.cuda.nvtx.range("microbatch_overlapping"):
        start = time.time()
        microbatch_overlapping(nanosleep_module, stream1, stream2, args.nanosec)
        torch.cuda.current_stream().synchronize()
        end = time.time()
        print(f"microbatch_overlapping takes: {end - start:.3f} sec")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        microbatch_overlapping(nanosleep_module, stream1, stream2, args.nanosec)

    with torch.cuda.nvtx.range("microbatch_overlapping (cudagraph mode)"):
        start = time.time()
        graph.replay()
        torch.cuda.current_stream().synchronize()
        end = time.time()
        print(f"microbatch_overlapping (cudagraph mode) takes: {end - start:.3f} sec")

    if args.trigger_illegal_access or os.getenv("TRIGGER_ILLEGAL_ACCESS") == "1":
        print("Triggering intentional illegal memory access...")
        nanosleep_module.illegal_access()
        torch.cuda.current_stream().synchronize()


if __name__ == "__main__":
    main()
