#!/usr/bin/env python3
import argparse
import time

import torch
from torch.utils.cpp_extension import load_inline

# TORCH_CUDA_ARCH_LIST=9.0 nsys profile -t cuda,cublas,nvtx  python bmm_with_sm_occupier.py      --occupy-blocks 1   --occupy-iters 100000   --batch 16 --m 4096 --n 4096 --k 4096 --dtype fp16 --repeats 1 --warmup 1

CUDA_SRC = r"""
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    TORCH_CHECK(err__ == cudaSuccess, #call " failed: ",                    \
                cudaGetErrorString(err__));                                  \
  } while (0)

__global__ __launch_bounds__(1024, 1)
void occupy_sm_kernel(
    int* stop,
    unsigned long long* sink,
    int sink_elems,
    int smem_bytes,
    unsigned long long max_iters) {
  extern __shared__ unsigned char smem[];
  int tid = threadIdx.x;

  unsigned long long r0 = 0x9e3779b97f4a7c15ULL ^ tid ^ (blockIdx.x << 16);
  unsigned long long r1 = r0 + 0xbf58476d1ce4e5b9ULL;
  unsigned long long r2 = r1 + 0x94d049bb133111ebULL;
  unsigned long long r3 = r2 + 0x2545f4914f6cdd1dULL;
  unsigned long long r4 = r3 + 0xda942042e4dd58b5ULL;
  unsigned long long r5 = r4 + 0xa24baed4963ee407ULL;
  unsigned long long r6 = r5 + 0x9fb21c651e98df25ULL;
  unsigned long long r7 = r6 + 0xc2b2ae3d27d4eb4fULL;
  unsigned long long r8 = r7 + 0x165667b19e3779f9ULL;
  unsigned long long r9 = r8 + 0xd6e8feb86659fd93ULL;
  unsigned long long r10 = r9 + 0xa5a3564e27f886afULL;
  unsigned long long r11 = r10 + 0x85ebca77c2b2ae63ULL;
  unsigned long long r12 = r11 + 0x27d4eb2f165667c5ULL;
  unsigned long long r13 = r12 + 0x3c79ac492ba7b653ULL;
  unsigned long long r14 = r13 + 0x1c69b3f74ac4ae35ULL;
  unsigned long long r15 = r14 + 0xdeadbeefcafebabeULL;

  if (smem_bytes > 0) {
    for (int i = tid; i < smem_bytes; i += blockDim.x) {
      smem[i] = static_cast<unsigned char>(i + blockIdx.x);
    }
  }
  __syncthreads();

  unsigned long long outer_iter = 0;
  while (__ldg(stop) == 0 && (max_iters == 0 || outer_iter < max_iters)) {
    #pragma unroll 64
    for (int i = 0; i < 64; ++i) {
      r0 = r0 * 2862933555777941757ULL + r8;
      r1 = r1 * 3202034522624059733ULL + r9;
      r2 = r2 * 3935559000370003845ULL + r10;
      r3 = r3 * 4768777513237032717ULL + r11;
      r4 = r4 * 2685821657736338717ULL + r12;
      r5 = r5 * 6364136223846793005ULL + r13;
      r6 = r6 * 1442695040888963407ULL + r14;
      r7 = r7 * 1181783497276652981ULL + r15;
      r8 ^= (r0 >> 17) + r1;
      r9 ^= (r1 >> 19) + r2;
      r10 ^= (r2 >> 23) + r3;
      r11 ^= (r3 >> 29) + r4;
      r12 ^= (r4 >> 31) + r5;
      r13 ^= (r5 >> 37) + r6;
      r14 ^= (r6 >> 41) + r7;
      r15 ^= (r7 >> 43) + r0;
    }

    if (smem_bytes > 0 && ((tid & 31) == 0)) {
      int idx = static_cast<int>((r0 + tid * 131 + blockIdx.x * 17) %
                                 static_cast<unsigned long long>(smem_bytes));
      smem[idx] ^= static_cast<unsigned char>(r15);
    }
    ++outer_iter;
  }

  if (tid == 0 && blockIdx.x < sink_elems) {
    sink[blockIdx.x] = r0 ^ r1 ^ r2 ^ r3 ^ r4 ^ r5 ^ r6 ^ r7 ^
                       r8 ^ r9 ^ r10 ^ r11 ^ r12 ^ r13 ^ r14 ^ r15;
  }
}

void launch_occupier(
    torch::Tensor stop,
    torch::Tensor sink,
    int blocks,
    int threads,
    int smem_bytes,
    unsigned long long max_iters) {
  TORCH_CHECK(stop.is_cuda(), "stop must be a CUDA tensor");
  TORCH_CHECK(sink.is_cuda(), "sink must be a CUDA tensor");
  TORCH_CHECK(stop.scalar_type() == torch::kInt32, "stop must be int32");
  TORCH_CHECK(sink.scalar_type() == torch::kInt64, "sink must be int64");
  TORCH_CHECK(stop.numel() >= 1, "stop must have at least one element");
  TORCH_CHECK(blocks > 0, "blocks must be positive");
  TORCH_CHECK(threads > 0 && threads <= 1024, "threads must be in [1, 1024]");
  TORCH_CHECK(smem_bytes >= 0, "smem_bytes must be non-negative");

  CHECK_CUDA(cudaFuncSetAttribute(
      occupy_sm_kernel,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared));

  if (smem_bytes > 48 * 1024) {
    CHECK_CUDA(cudaFuncSetAttribute(
        occupy_sm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes));
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  occupy_sm_kernel<<<blocks, threads, smem_bytes, stream>>>(
      stop.data_ptr<int>(),
      reinterpret_cast<unsigned long long*>(sink.data_ptr<int64_t>()),
      static_cast<int>(sink.numel()),
      smem_bytes,
      max_iters);
  CHECK_CUDA(cudaGetLastError());
}
"""


CPP_SRC = r"""
#include <torch/extension.h>

void launch_occupier(
    torch::Tensor stop,
    torch::Tensor sink,
    int blocks,
    int threads,
    int smem_bytes,
    unsigned long long max_iters);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch_occupier", &launch_occupier, "Launch persistent SM occupier");
}
"""


def build_extension():
    return load_inline(
        name="sm_occupier_ext",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        extra_cuda_cflags=["-O3"],
        extra_cflags=["-O3"],
        verbose=False,
    )


def make_inputs(args, device):
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "tf32": torch.float32,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    torch.backends.cuda.matmul.allow_tf32 = args.dtype == "tf32"
    a = torch.randn(args.batch, args.m, args.k, device=device, dtype=dtype)
    b = torch.randn(args.batch, args.k, args.n, device=device, dtype=dtype)
    return a, b


def time_bmm(a, b, repeats, warmup):
    print("begin time_bmm")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            torch.bmm(a, b)
        stream.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        for _ in range(repeats):
            c = torch.bmm(a, b)
        end.record(stream)
    end.synchronize()
    # Keep the final output live until timing has completed.
    _ = c
    print("end time_bmm")
    return start.elapsed_time(end) / repeats


def run_with_occupier(ext, a, b, args):
    stop = torch.zeros(1, device=a.device, dtype=torch.int32)
    sink = torch.empty(args.occupy_blocks, device=a.device, dtype=torch.int64)
    occ_stream = torch.cuda.Stream()

    with torch.cuda.stream(occ_stream):
        ext.launch_occupier(
            stop,
            sink,
            args.occupy_blocks,
            args.occupy_threads,
            args.occupy_shmem_kb * 1024,
            args.occupy_iters,
        )

    # Give the persistent CTAs a small head start without synchronizing on them.
    time.sleep(args.occupy_startup_sleep)
    ms = time_bmm(a, b, args.repeats, args.warmup)

    stop.fill_(1)
    occ_stream.synchronize()
    return ms


def tflops(batch, m, n, k, ms):
    flops = 2.0 * batch * m * n * k
    return flops / (ms * 1.0e-3) / 1.0e12


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "tf32", "fp32"], default="fp16")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--occupy-blocks", type=int, default=10)
    parser.add_argument("--occupy-threads", type=int, default=1024)
    parser.add_argument("--occupy-shmem-kb", type=int, default=200)
    parser.add_argument(
        "--occupy-iters",
        type=int,
        default=0,
        help="0 means run until stopped; positive values make the occupier finite.",
    )
    parser.add_argument("--occupy-startup-sleep", type=float, default=0.2)
    parser.add_argument("--skip-baseline", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(f"device={props.name}")
    print(f"sm_count={props.multi_processor_count}")
    print(f"shape: batch={args.batch}, m={args.m}, n={args.n}, k={args.k}, dtype={args.dtype}")
    print(
        "occupier: "
        f"blocks={args.occupy_blocks}, threads={args.occupy_threads}, "
        f"dynamic_smem={args.occupy_shmem_kb} KiB/block, "
        f"iters={'until-stop' if args.occupy_iters == 0 else args.occupy_iters}"
    )

    ext = build_extension()
    a, b = make_inputs(args, device)

    base_ms = None
    if not args.skip_baseline:
        torch.cuda.synchronize()
        base_ms = time_bmm(a, b, args.repeats, args.warmup)
        torch.cuda.synchronize()
    occ_ms = run_with_occupier(ext, a, b, args)
    torch.cuda.synchronize()

    occ_tflops = tflops(args.batch, args.m, args.n, args.k, occ_ms)

    expected = props.multi_processor_count / max(
        1, props.multi_processor_count - args.occupy_blocks
    )
    if base_ms is not None:
        base_tflops = tflops(args.batch, args.m, args.n, args.k, base_ms)
        print(f"baseline:      {base_ms:.3f} ms, {base_tflops:.2f} TFLOP/s")
    print(f"with occupier: {occ_ms:.3f} ms, {occ_tflops:.2f} TFLOP/s")
    if base_ms is not None:
        print(f"slowdown:      {occ_ms / base_ms:.3f}x")
    print(f"ideal sm-ratio slowdown for {args.occupy_blocks} occupied SMs: {expected:.3f}x")


if __name__ == "__main__":
    main()
