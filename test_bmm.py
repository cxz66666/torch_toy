import torch

import triton
import triton.language as tl

tensor_info_dict = {
    "input_shape": [
        # # actual
        (32, 2048, 1120),
        (32, 2048, 140),
        (32, 2048, 1120),
        (32, 2048, 2240),
        (32, 2048, 280),
        (32, 2048, 2240),
        (35, 2048, 1024),
        (35, 2048, 128),
        (35, 2048, 1024),
        (35, 2048, 2048),
        (35, 2048, 256),
        (35, 2048, 2048),
        (32, 2048, 1120),
        (32, 2048, 140),
        (32, 2048, 1120),
        (32, 2048, 2240),
        (32, 2048, 280),
        (32, 2048, 2240),
        (35, 2048, 1024),
        (35, 2048, 128),
        (35, 2048, 1024),
        (35, 2048, 2048),
        (35, 2048, 256),
        (35, 2048, 2048)
    ],
    "weight_shape": [
        # # actual
        (32, 1120, 140),
        (32, 140, 1120),
        (32, 1120, 2240),
        (32, 2240, 280),
        (32, 280, 2240),
        (32, 2240, 1120),
        (35, 1024, 128),
        (35, 128, 1024),
        (35, 1024, 2048),
        (35, 2048, 256),
        (35, 256, 2048),
        (35, 2048, 1024),
        (32, 1120, 140),
        (32, 140, 1120),
        (32, 1120, 2240),
        (32, 2240, 280),
        (32, 280, 2240),
        (32, 2240, 1120),
        (35, 1024, 128),
        (35, 128, 1024),
        (35, 1024, 2048),
        (35, 2048, 256),
        (35, 256, 2048),
        (35, 2048, 1024)
    ],
}
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _batch_Dense_kernel(
    A, B, C, bias, out_scale,
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bn, stride_bk,
    stride_cb, stride_cm, stride_cn,
    stride_biasb, stride_biasm, stride_biasn,
    USE_BIAS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COLUMN_MAJOR: tl.constexpr = False,
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

    B_order: tl.constexpr = (0, 1) if COLUMN_MAJOR == True else (1, 0)
    B_ptr = tl.make_block_ptr(
        B + bid * stride_bb,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=B_order,
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

    # add bias
    if USE_BIAS == True:
        bias_ptr = tl.make_block_ptr(
            bias + bid * stride_biasb,
            shape=(1, N),
            strides=(stride_biasm, stride_biasn),
            offsets=(0, pid_n * BLOCK_SIZE_N),
            block_shape=(1, BLOCK_SIZE_N),
            order=(1, 0),
        )
        bias_block = tl.load(bias_ptr, boundary_check=(1, 0), padding_option='')
        accumulator = accumulator + bias_block

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

def _batch_matmul_fwd(
    A_mat: torch.Tensor, 
    B_mat: torch.Tensor, 
    bias: torch.Tensor = None,
    *, 
    use_bias: bool = False,
    out_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Args:
        A_mat (torch.Tensor): Input matrix A. Shape (B, M, K).
        B_mat (torch.Tensor): Input matrix B. Shape (B, K, N).
        out_scale (torch.Tensor): Scaling factor for the output.
        out_dtype (torch.dtype, optional): Output data type. Default is torch.float16.
        B_mat_order (bool, optional): Whether the order of B_mat is column-major. Default is True.
    Returns:
        torch.Tensor: The result tensor of the batch matrix multiplication. Shape (B, M, N).
    """
    BS, M, K = A_mat.shape
    N =  B_mat.shape[2]

    # determine major order of B mat
    B_mat_col_order = B_mat.stride(2) != 1
    C_mat = torch.empty((BS, M, N), device=A_mat.device, dtype=out_dtype)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), BS, )
    B_mat_stride = (
            B_mat.stride(0),
            B_mat.stride(2),
            B_mat.stride(1)
        )
    
    bias_strides = ((bias.stride(0), bias.stride(1), bias.stride(2)) if use_bias else (0, 0, 0))
    _batch_Dense_kernel[grid](
        A_mat, B_mat, C_mat, bias, out_scale,
        M, N, K,
        A_mat.stride(0), A_mat.stride(1), A_mat.stride(2),
        *B_mat_stride,
        C_mat.stride(0), C_mat.stride(1), C_mat.stride(2),
        *bias_strides,
        USE_BIAS=use_bias,
        OUT_DTYPE=out_dtype.__str__(),
        COLUMN_MAJOR=B_mat_col_order,
    )
    return C_mat

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

        # line_vals=["B_mat_row", "B_mat_row_to_col", "B_mat_col"],  # Label name for the lines
        # line_names=["B_mat_row", "B_mat_row_to_col", "B_mat_col"],  # Line styles
        # styles=[("green", "-"), ("blue", "-"), ("red", "-")],

        line_vals=["B_mat_row_to_col"],  # Label name for the lines
        line_names=["B_mat_row_to_col"],  # Line styles
        styles=[("red", "-")],

        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance-fp8",  # Name for the plot, used also as a file name for saving the plot.
        args={}
    ))


DEVICE = "cuda"


@triton.testing.perf_report(configs)
def benchmark(B, M, N, K, provider):
    a = torch.ones((B, M, K), device=DEVICE, dtype=torch.float8_e5m2)
    b = torch.ones((B, K, N), device=DEVICE, dtype=torch.float8_e5m2)
    scale_a = torch.tensor(1., device=DEVICE, dtype=torch.float32)
    scale_b = torch.tensor(1., device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "B_mat_col":
        b = b.transpose(1,2).contiguous().transpose(1,2)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: _batch_matmul_fwd(a, b, out_scale=scale_a), quantiles=quantiles)
    if provider == 'B_mat_row':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: _batch_matmul_fwd(a, b, out_scale=scale_a), quantiles=quantiles)
    if provider == "B_mat_row_to_col":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: _batch_matmul_fwd(a, b.transpose(1,2).contiguous().transpose(1,2), out_scale=scale_a), quantiles=quantiles)
    perf = lambda ms: 2 * B * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True)
