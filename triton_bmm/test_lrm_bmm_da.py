import torch

import triton
import triton.language as tl

tensor_info_dict = {
    "input_shape": [
        # # actual
        (48, 512, 3072),
        (48, 512, 384),
        (48, 512, 6144),
        (48, 512, 768),
        (48, 512, 1024),
        (48, 512, 128),
        (48, 512, 1024),
        (48, 512, 2048),
        (48, 512, 256),
    ],
    "weight_shape": [
        # # actual
        (48, 384, 3072),
        (48, 3072, 384),
        (48, 3072, 6144),
        (48, 6144, 768),
        (48, 768, 6144),
        (48, 6144, 3072),
        (48, 1024, 1024),
        (48, 1024, 128),
        (48, 128, 1024),
        (48, 1024, 2048),
        (48, 2048, 256),
        (48, 256, 2048),
        (48, 2048, 1024),
    ],
}
@triton.autotune(
configs = [
    triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, 'BLOCK_SIZE_K': BK, 'GROUP_SIZE_M': GS}, num_stages=s, num_warps=w) \
    for BM in [32, 64, 128]\
    for BN in [32, 64, 128, 256]\
    for BK in [32, 64, 128, 256]\
    for GS in [1, 4, 8]\
    for s in ([1, 3, 4, 7])\
    for w in [4, 8]\
],
    key=['M', 'N', 'K'],
)
@triton.jit
def fp8_bmm_nt(
    A, B, C, out_scale,
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bn, stride_bk,
    stride_cb, stride_cm, stride_cn,
    OUT_DTYPE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)

    # zigzag mapping
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # setup ptrs
    A_ptr = tl.make_block_ptr(
        A + bid * stride_ab,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )
    B_ptr = tl.make_block_ptr(
        B + bid * stride_bb,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(0, 1),
    )
    
    # compute
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        A_block = tl.load(
            A_ptr,
            boundary_check=(1, 0),
            padding_option='',
        )

        B_block = tl.load(
            B_ptr,
            boundary_check=(0, 1),
            padding_option='',
        )
        accumulator = tl.dot(A_block, B_block, accumulator)

        A_ptr = tl.advance(A_ptr, (0, BLOCK_SIZE_K))
        B_ptr = tl.advance(B_ptr, (BLOCK_SIZE_K, 0))
   
    # dequantize
    SCALING_FACTOR = tl.load(out_scale)
    accumulator = accumulator * SCALING_FACTOR

    # convert to OUT_DTYPE
    if OUT_DTYPE == 'torch.bfloat16':
        accumulator = accumulator.to(tl.bfloat16)
    elif OUT_DTYPE == 'torch.float16':
        accumulator = accumulator.to(tl.float16)
    
    # setup ptrs
    C_ptr = tl.make_block_ptr(
        C + bid * stride_cb,
        shape=(M, N),
        strides=(stride_cm, stride_cn), 
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )

    # write back
    tl.store(
        C_ptr, 
        accumulator, 
        boundary_check=(0, 1),
    )

def _batch_dense( 
    A_mat, B_mat_column_major,
    A_mat_scale, B_mat_column_major_scale,
    output_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    BS, M, K = A_mat.shape
    N =  B_mat_column_major.shape[2]

    output = torch.empty(BS,M,N, device=A_mat.device, dtype=output_dtype)
    grid_grad_B = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), BS, )

    fp8_bmm_nt[grid_grad_B](
        A_mat, B_mat_column_major, output, A_mat_scale * B_mat_column_major_scale,
        M, N, K,
        A_mat.stride(0), A_mat.stride(1), A_mat.stride(2),
        B_mat_column_major.stride(0), B_mat_column_major.stride(2), B_mat_column_major.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        OUT_DTYPE=output_dtype.__str__(),
    )
    
    return output



def combine_matmul_shapes(data_dict):
    input_shapes = data_dict.get("input_shape", [])
    weight_shapes = data_dict.get("weight_shape", [])
    
    combined_shapes = set()

    for input_s in input_shapes:
        b_in, m, k_in = input_s
        
        for weight_s in weight_shapes:
            b_wt, k_wt, n = weight_s
            
            if b_in == b_wt and k_in == k_wt:
                combined_shapes.add((b_in, m, k_in, n))
    
    return list(combined_shapes)

configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["B", "M", "N", "K"],  # Argument names to use as an x-axis for the plot
        x_vals=combine_matmul_shapes(tensor_info_dict),  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot

        # line_vals=["always_nt", "original", "torch_fp16", "always_nt_pre_contiguous"],  # Label name for the lines
        # line_names=["always_nt", "original", "torch_fp16", "always_nt_pre_contiguous"],  # Line styles
        # styles=[("green", "-"), ("yellow", "-"), ("blue", "-"), ("red", "-")],

        line_vals=["always_nt"],  # Label name for the lines
        line_names=["always_nt"],  # Line styles
        styles=[("green", "-")],

        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance-fp8",  # Name for the plot, used also as a file name for saving the plot.
        args={}
    ))


DEVICE = "cuda"


@triton.testing.perf_report(configs)
def benchmark(B, M, N, K, provider):
    a = torch.ones((B, M, N), device=DEVICE, dtype=torch.float8_e5m2)
    b = torch.ones((B, K, N), device=DEVICE, dtype=torch.float8_e5m2)
    b = b.transpose(1, 2) # Ensure B is in column-major order
    scale_a = torch.tensor(1., device=DEVICE, dtype=torch.float32)
    scale_b = torch.tensor(1., device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "always_nt":        
        # triton.testing.do_bench(lambda: _batch_dense(a, b, scale_a, scale_b), quantiles=quantiles)
        _batch_dense(a, b, scale_a, scale_b)
        ms, min_ms, max_ms = 1, 1, 1
    perf = lambda ms: 2 * B * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True)
