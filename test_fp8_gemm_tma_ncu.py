import torch
import triton
import triton.language as tl

from triton.tools.tensor_descriptor import TensorDescriptor



@triton.jit
def _compute_pid(tile_id, num_pid_in_matrix, num_pid_in_group, num_pid_m, GROUP_SIZE_M):
    pid_bs = tile_id // num_pid_in_matrix
    matrix_tild_id = tile_id % num_pid_in_matrix
    group_id = matrix_tild_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (matrix_tild_id % group_size_m)
    pid_n = (matrix_tild_id % num_pid_in_group) // group_size_m
    return pid_bs, pid_m, pid_n



def fp8_per_tensor_quant_bmm_tma_set_block_size_hook(nargs):
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", False)
    BLOCK_M = nargs["BLOCK_SIZE_M"]
    BLOCK_N = nargs["BLOCK_SIZE_N"]
    BLOCK_K = nargs["BLOCK_SIZE_K"]
    nargs["a_desc"].block_shape = [1, BLOCK_M, BLOCK_K]
    # nargs["b_desc"].block_shape = [1, BLOCK_K, BLOCK_N]
    nargs["b_desc"].block_shape = [1, BLOCK_N, BLOCK_K]
    if EPILOGUE_SUBTILE:
        nargs["c_desc"].block_shape = [1, BLOCK_M, BLOCK_N // 2]
    else:
        nargs["c_desc"].block_shape = [1, BLOCK_M, BLOCK_N]


def fp8_per_tensor_quant_bmm_tma_persistent_get_configs(pre_hook=None):
    return [
        triton.Config(
            {
                'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 2, "EPILOGUE_SUBTILE": False,
            }, 
            num_stages=3, num_warps=4, pre_hook=pre_hook),
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": gs, "EPILOGUE_SUBTILE":
        #         SUBTILE
        #     }, num_stages=s, num_warps=w, pre_hook=pre_hook)  #
        # for BM in [128, 256]
        # for BN in [128, 256]
        # for BK in [64, 128]
        # for gs in [1, 2, 8]
        # for s in ([2, 3, 4])
        # for w in [4, 8]
        # for SUBTILE in [True, False]
    ]

@triton.autotune(
    configs=fp8_per_tensor_quant_bmm_tma_persistent_get_configs(pre_hook=fp8_per_tensor_quant_bmm_tma_set_block_size_hook),
    key=["M", "N", "K", "WARP_SPECIALIZE"],
)
@triton.jit
def fp8_per_tensor_quant_bmm_persistent_tma_kernel(
    a_desc, b_desc, c_desc,
    scale_a_ptr, scale_b_ptr,
    BS, M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    num_tiles = BS * num_pid_m * num_pid_n
    num_pid_in_matrix = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    
    # NOTE: There is currently a bug in blackwell pipelining that means it can't handle a value being
    # used in both the prologue and epilogue, so we duplicate the counters as a work-around.
    tile_id_c = start_pid - NUM_SMS

    scale_a = tl.load(scale_a_ptr)
    scale_b = tl.load(scale_b_ptr)

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True, warp_specialize=WARP_SPECIALIZE):
        pid_bs, pid_m, pid_n = _compute_pid(tile_id, num_pid_in_matrix, num_pid_in_group, num_pid_m, GROUP_SIZE_M)

        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((1, BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([pid_bs, offs_am, offs_k])
            b = b_desc.load([pid_bs, offs_bn, offs_k])
            accumulator = tl.dot(a, tl.permute(b,(0, 2, 1)), accumulator)

        accumulator = accumulator * scale_a * scale_b
        
        tile_id_c += NUM_SMS
        pid_bs, pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_matrix, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_cm_c = pid_m * BLOCK_SIZE_M
        offs_cn_c = pid_n * BLOCK_SIZE_N

        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (1, BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 1, 3, 2))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(tl.bfloat16)
            c_desc.store([pid_bs, offs_cm_c, offs_cn_c], c0)
            c1 = acc1.to(tl.bfloat16)
            c_desc.store([pid_bs, offs_cm_c, offs_cn_c + BLOCK_SIZE_N // 2], c1)
        else:
            accumulator = accumulator.to(tl.bfloat16)
            c_desc.store([pid_bs, offs_cm_c, offs_cn_c], accumulator)
        


def fp8_per_tensor_quant_bmm_persistent_tma(
    fp8_a: torch.Tensor,
    fp8_b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    warp_specialize: bool = False,
):
    assert fp8_a.is_contiguous()
    assert fp8_b.is_contiguous()
    # assert fp8_a.shape[2] == fp8_b.shape[1]
    assert fp8_a.shape[2] == fp8_b.shape[2]

    BS, M, K = fp8_a.shape
    # N = fp8_b.shape[2]
    N = fp8_b.shape[1]
    c = torch.empty((BS, M, N), dtype=torch.bfloat16, device=fp8_a.device)

    dummy_block = [1, 1, 1]
    a_desc = TensorDescriptor(fp8_a, fp8_a.shape, fp8_a.stride(), dummy_block)
    b_desc = TensorDescriptor(fp8_b, fp8_b.shape, fp8_b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)
    
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (min(
            NUM_SMS,
            triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N) * BS,
        ), )
    
    fp8_per_tensor_quant_bmm_persistent_tma_kernel[grid](
        a_desc, b_desc, c_desc,
        scale_a, scale_b,
        BS, M, N, K,
        OUTPUT_DTYPE=c.dtype.__str__(),
        NUM_SMS=NUM_SMS,
        WARP_SPECIALIZE=warp_specialize,
    )
    return c


def bench(warmup_iter=10, benchmark_iter=30):
    BS, M, N, K = 14, 2048, 12288, 4096
    # 14	2048	12288	4096
    a = torch.randn((BS, M, K), dtype=torch.bfloat16, device="cuda")
    # b = torch.randn((BS, K, N), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((BS, N, K), dtype=torch.bfloat16, device="cuda")
    fp8_a = a.to(torch.float8_e4m3fn)
    fp8_b = b.to(torch.float8_e4m3fn)
    scale_a = torch.ones([], dtype=torch.float16, device="cuda")
    scale_b = torch.ones([], dtype=torch.float16, device="cuda")
    
    # warmup
    for _ in range(warmup_iter):
        c = fp8_per_tensor_quant_bmm_persistent_tma(fp8_a, fp8_b, scale_a, scale_b, warp_specialize=True)

    # benchmark
    # cuda event
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(benchmark_iter):
        c = fp8_per_tensor_quant_bmm_persistent_tma(fp8_a, fp8_b, scale_a, scale_b, warp_specialize=True)
    
    end_event.record()
    torch.cuda.synchronize()
    print(f"time: {start_event.elapsed_time(end_event)/benchmark_iter} ms")
    # TFLOPS:
    print(f"TFLOPS: {2 * BS * M * N * K / (start_event.elapsed_time(end_event)/benchmark_iter) / 1e9} TFLOPS")
    # print(c)


if __name__ == "__main__":
    bench()



"""
Persistent Matmul
=====================
This script demonstrates persistent kernel implementations of matrix multiplication using Triton.
Various matmul methods are included, such as naive, persistent, and TMA (Tensor Memory Accelerator) based approaches.
The kernels support both FP16 and FP8 data types but the FP8 implementation is only available on CUDA devices with compute capability >= 9.0.

Triton and cuBLAS implementations are benchmarked under different configurations and evaluated using the proton profiler.
Users can pass command-line arguments to specify matrix dimensions and iteration steps flexibly.

.. code-block:: bash

    # FP8
    python 09-persistent-matmul.py --prec fp8 --K_range 128 1024 --K_step 128

    # FP16
    python 09-persistent-matmul.py --prec fp16 --K_range 128 1024 --K_step 128

Note that currently this tutorial will fail on devices with a small shared memory size, such as RTX-4090.
"""

import argparse
import itertools

import torch
import triton
import triton.language as tl
import triton.profiler as proton
from triton.tools.tensor_descriptor import TensorDescriptor
from contextlib import contextmanager

from typing import Optional

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)
triton.set_allocator(alloc_fn)


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_hopper():
    return torch.cuda.get_device_capability()[0] == 9


def supports_ws():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    BSZ, M, N, K, WS = args["BSZ"], args["M"], args["N"], args["K"], args.get("WARP_SPECIALIZE", False)
    ws_str = "_ws" if WS else ""
    ret["name"] = f"{kernel.name}{ws_str} [BSZ={BSZ}, M={M}, N={N}, K={K}]"
    # if "c_ptr" in args:
    #     bytes_per_elem = args["c_ptr"].element_size()
    # else:
    bytes_per_elem = 2
    ret[f"flops{bytes_per_elem * 8}"] = 2. * BSZ * M * N * K
    ret["bytes"] = bytes_per_elem * BSZ * (M * K + N * K + M * N)
    return ret


HAS_TENSOR_DESC = supports_tma() and hasattr(tl, "make_tensor_descriptor")
HAS_HOST_TENSOR_DESC = supports_tma() and hasattr(triton.tools.tensor_descriptor, "TensorDescriptor")
HAS_WARP_SPECIALIZE = supports_ws() and HAS_TENSOR_DESC


def matmul_get_configs(pre_hook=None):
    return [
        triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": gs}, num_stages=s,
                      num_warps=w, pre_hook=pre_hook)
        for BM in [128, 256]
        for BN in [128, 256]
        for BK in [64, 128]
        for gs in [4, 8]
        for s in ([2, 3, 5])
        for w in [4, 8]
    ]


def matmul_tma_set_block_size_hook(nargs):
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", False)
    BLOCK_M = nargs["BLOCK_SIZE_M"]
    BLOCK_N = nargs["BLOCK_SIZE_N"]
    BLOCK_K = nargs["BLOCK_SIZE_K"]
    nargs["a_desc"].block_shape = [BLOCK_M, BLOCK_K]
    nargs["b_desc"].block_shape = [BLOCK_N, BLOCK_K]
    if EPILOGUE_SUBTILE:
        nargs["c_desc"].block_shape = [BLOCK_M, BLOCK_N // 2]
    else:
        nargs["c_desc"].block_shape = [BLOCK_M, BLOCK_N]


@triton.autotune(
    configs=matmul_get_configs(pre_hook=None),
    key=["M", "N", "K", "WARP_SPECIALIZE"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_tma(a_ptr, b_ptr, c_ptr,  #
                      scale_a_ptr, scale_b_ptr,
                      BSZ, M, N, K,  #
                      stride_ab,
                      stride_bb,
                      stride_cb,
                      BLOCK_SIZE_M: tl.constexpr,  #
                      BLOCK_SIZE_N: tl.constexpr,  #
                      BLOCK_SIZE_K: tl.constexpr,  #
                      GROUP_SIZE_M: tl.constexpr,  #
                      FP8_OUTPUT: tl.constexpr,  #
                      WARP_SPECIALIZE: tl.constexpr,  #
                      ):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)

    a_desc = tl.make_tensor_descriptor(
        a_ptr + bid * stride_ab, 
        shape=[M, K], 
        strides=[K, 1], 
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr + bid * stride_bb, 
        shape=[N, K], 
        strides=[K, 1], 
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr + bid * stride_cb, 
        shape=[M, N], 
        strides=[N, 1], 
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N]
    )

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in tl.range(k_tiles, warp_specialize=WARP_SPECIALIZE):
        offs_k = k * BLOCK_SIZE_K
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

    scale_a = tl.load(scale_a_ptr)
    scale_b = tl.load(scale_b_ptr)
    accumulator = accumulator * scale_a * scale_b
    c = accumulator.to(dtype)

    offs_cm = pid_m * BLOCK_SIZE_M
    offs_cn = pid_n * BLOCK_SIZE_N
    c_desc.store([offs_cm, offs_cn], c)


def matmul_tma(a, b, warp_specialize: bool):
    # Check constraints.
    assert a.shape[2] == b.shape[2], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    BSZ, M, K = a.shape
    BSZ, N, K = b.shape
    dtype = a.dtype

    c = torch.empty((BSZ, M, N), device=a.device, dtype=torch.float16)

    scale_a = torch.ones([], device=a.device, dtype=torch.float32)
    scale_b = torch.ones([], device=a.device, dtype=torch.float32)
    # A dummy block value that will be overwritten when we have the real block size
    # dummy_block = [1, 1]
    # a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    # b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    # c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    # def grid(META):
    #     BLOCK_M = META["BLOCK_SIZE_M"]
    #     BLOCK_N = META["BLOCK_SIZE_N"]
    #     return (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), BSZ,)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), BSZ, )

    matmul_kernel_tma[grid](
        a, b, c,  #
        scale_a, scale_b,
        BSZ, M, N, K,  #
        a.stride(0), b.stride(0), c.stride(0),
        FP8_OUTPUT=False,  #
        WARP_SPECIALIZE=warp_specialize,  #
    )
    return c


@contextmanager
def proton_context():
    proton.activate(0)
    try:
        yield
    finally:
        proton.deactivate(0)


def bench_fn(label, reps, warmup_reps, fn, *args):
    print(f"Benchmarking {label}: ...", end="")
    for _ in range(warmup_reps):
        fn(*args)
    with proton_context():
        for _ in range(reps):
            fn(*args)
    print(f"\rBenchmarking {label}: done")


def bench(K, dtype, reps=10000, warmup_reps=10000):
    BSZ = 14
    M = 2048
    N = 12288
    a = torch.randn((BSZ, M, K), device="cuda", dtype=torch.float16).to(dtype)
    b = torch.randn((BSZ, K, N), device="cuda", dtype=torch.float16).to(dtype)
    b = b.permute(0, 2, 1).contiguous()

    # M = 2048
    # N = 12288
    # a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype)
    # b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(dtype)
    # b = b.T.contiguous()

    warp_specialize = [False, True] if HAS_WARP_SPECIALIZE else [False]
    for ws in warp_specialize:
        ws_str = "_ws" if ws else ""
        bench_fn(f"tma{ws_str}", reps, warmup_reps, lambda a, b: matmul_tma(a, b, ws), a, b)


def show_profile(precision, profile_name):
    import triton.profiler.viewer as proton_viewer
    metric_names = ["time/ms"]
    if precision == 'fp8':
        metric_names = ["tflop16/s"] + metric_names
    elif precision == 'fp16':
        metric_names = ["tflop16/s"] + metric_names
    file_name = f"{profile_name}.hatchet"
    tree, metrics = proton_viewer.parse(metric_names, file_name)
    proton_viewer.print_tree(tree, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-K", type=int, required=False, default=512)
    parser.add_argument("--K_range", type=int, nargs=2)
    parser.add_argument("--K_step", type=int, default=512)
    parser.add_argument("--prec", type=str, choices=["fp8", "fp16"], default="fp16")
    args = parser.parse_args()

    if args.prec == 'fp8' and (not hasattr(torch, "float8_e4m3fn") or not is_cuda()):
        print("This example requires CUDA with fp8 support.")
    else:
        dtype = torch.float8_e4m3fn if args.prec == 'fp8' else torch.float16

        if args.K and args.K_range is None:
            args.K_range = [args.K, args.K]
            args.K_step = 1  # doesn't matter as long as it's not 0

        torch.manual_seed(0)


        proton.start("matmul", hook="triton")
        proton.deactivate()
        for K in range(args.K_range[0], args.K_range[1] + 1, args.K_step):
            bench(K, dtype)
        proton.finalize()
        show_profile(args.prec, "matmul")