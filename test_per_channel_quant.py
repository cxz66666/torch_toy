import torch

import triton
import triton.language as tl

tensor_info_dict = {
    "input_shape": [
        # # actual
        (32, 1024, 288),
        (32, 1024, 2304),
        (32, 1024, 4608),
        (32, 1024, 576),
        (32, 1024, 4608),
        (32, 1024, 2304),
        (36, 1024, 2048),
        (36, 1024, 256),
        (36, 1024, 2048),
        (36, 1024, 4096),
        (36, 1024, 512),
        (36, 1024, 4096),
        (36, 1024, 2048),

    ],
    "weight_shape": [
        # # actual
        (288, 2304),
        (2304, 288),
        (4608, 2304),
        (576, 4608),
        (4608, 576),
        (2304, 4608),
        (2048, 2048),
        (256, 2048),
        (2048, 256),
        (4096, 2048),
        (512, 4096),
        (4096, 512),
        (2048, 4096),
    ],
}

def fp8_quant(
    hp_tensor: torch.Tensor,
    float8_dtype: torch.dtype,
    axiswise_dim: int,
):
    amax = torch.amax(torch.abs(hp_tensor), dim=axiswise_dim, keepdim=True)
    fp8_range = torch.finfo(float8_dtype).max
    scale = fp8_range / torch.clamp(amax.to(torch.float32), min=1e-12)
    fp8_tensor = (hp_tensor.to(torch.float32) * scale).clamp(min=-fp8_range, max=fp8_range).to(float8_dtype)
    return fp8_tensor, scale

compiled_fp8_quant = torch.compile(fp8_quant, backend="inductor")




@triton.autotune(
    configs=[
        # triton.Config({'BLOCK_N': 256}, num_stages=7, num_warps=4),
        # triton.Config({'BLOCK_N': 256}, num_stages=8, num_warps=4),
        # triton.Config({'BLOCK_N': 512}, num_stages=2, num_warps=4),
        # triton.Config({'BLOCK_N': 512}, num_stages=6, num_warps=4),
        # triton.Config({'BLOCK_N': 512}, num_stages=8, num_warps=4),
        # triton.Config({'BLOCK_N': 1024}, num_stages=1, num_warps=4),
        # triton.Config({'BLOCK_N': 1024}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_N': 1024}, num_stages=5, num_warps=4),
        # triton.Config({'BLOCK_N': 1024}, num_stages=6, num_warps=4),
        # triton.Config({'BLOCK_N': 1024}, num_stages=7, num_warps=4),
        # triton.Config({'BLOCK_N': 1024}, num_stages=8, num_warps=4),

        triton.Config({'BLOCK_N': BN}, num_stages=s, num_warps=w) \
        for BN in [128, 256, 512, 1024]\
        for s in ([1, 2, 3, 4, 5, 6, 7, 8])\
        for w in [4, 8]
    
    ],
    key=['M', 'N'],
)
@triton.jit
def fp8_per_channel_quant_kernel(
    x_ptr, y_ptr, scale_ptr,
    B, M, N,
    stride_x_b, stride_x_m, stride_x_n,
    stride_y_b, stride_y_m, stride_y_n,
    stride_scale_b, stride_scale_m, stride_scale_n,   # scale keepdim shape (B,M,1)
    fp8_max: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OUT_IS_E4M3: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    base_in  = pid * stride_x_b + bid * stride_x_m
    base_out = pid * stride_y_b + bid * stride_y_m
    # 1) reduce over K
    amax = EPS
    for k0 in range(0, N, BLOCK_N):
        offs_k = k0 + tl.arange(0, BLOCK_N)
        mask_k = offs_k < N
        ptrs = x_ptr + (base_in + offs_k * stride_x_n)
        x = tl.load(ptrs, mask=mask_k, other=0.0)
        x = tl.abs(x)
        amax = tl.maximum(amax, tl.max(x, axis=0))

    # 2) compute scale and store keepdim at [pid,bid,0]
    scale = fp8_max / tl.cast(amax, tl.float32)
    tl.store(scale_ptr + (pid * stride_scale_b + bid * stride_scale_m), scale)
    # 3) second pass: quantize and store
    for k0 in range(0, N, BLOCK_N):
        offs_k = k0 + tl.arange(0, BLOCK_N)
        mask_k = offs_k < N
        in_ptrs  = x_ptr  + (base_in  + offs_k * stride_x_n)
        out_ptrs = y_ptr + (base_out + offs_k * stride_y_n)
        x = tl.load(in_ptrs, mask=mask_k, other=0.0)
        y = tl.cast(x, tl.float32) * scale

        y = tl.clamp(y, min=-fp8_max, max=fp8_max)
        if OUT_IS_E4M3:
            y_fp8 = tl.cast(y, tl.float8e4nv)
        else:
            y_fp8 = tl.cast(y, tl.float8e5)
        tl.store(out_ptrs, y_fp8, mask=mask_k)

def fp8_quant_triton_simple(
    hp_tensor: torch.Tensor,
    float8_dtype: torch.dtype,
    axiswise_dim: int,
):
    if axiswise_dim not in (-1, -2):
        raise ValueError("axiswise_dim must be -1 or -2")
    if axiswise_dim == -2:
        x = hp_tensor.transpose(-1,-2)
    else:
        x = hp_tensor

    stride_x_b, stride_x_m, stride_x_n = x.stride()
    B, M, N = x.shape
    y = torch.empty((B, M, N), device=x.device, dtype=float8_dtype)
    stride_y_b, stride_y_m, stride_y_n = y.stride()
    scale = torch.empty((B, M, 1), dtype=torch.float32, device=x.device)
    stride_scale_b, stride_scale_m, stride_scale_n = scale.stride()
    
    fp8_max = float(torch.finfo(float8_dtype).max)
    out_is_e4m3 = float8_dtype == torch.float8_e4m3fn
    fp8_per_channel_quant_kernel[(B, M)](
        x, y, scale,
        B, M, N,
        stride_x_b, stride_x_m, stride_x_n,
        stride_y_b, stride_y_m, stride_y_n,
        stride_scale_b, stride_scale_m, stride_scale_n,
        fp8_max=fp8_max,
        OUT_IS_E4M3=out_is_e4m3,
        EPS=1e-12,
    )

    if axiswise_dim == -2:
        y = y.transpose(-1, -2)
        scale = scale.transpose(-1, -2)

    return y, scale


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=8, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_stages=8, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_stages=7, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_stages=8, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_stages=8, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=7, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=8, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=7, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=7, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=8, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=7, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=8, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=8),
    
        # triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
        # for BM in [32, 64, 128]\
        # for BN in [32, 64, 128, ]\
        # for s in ([1, 2, 3, 4, 7, 8])\
        # for w in [4,8]
    ],
    key=['M', 'N'],
)
@triton.jit
def fp8_per_channel_quant_kernel_blockptr(
    x_ptr, y_ptr, scale_ptr,
    B, M, N,
    stride_x_b, stride_x_m, stride_x_n,
    stride_y_b, stride_y_m, stride_y_n,
    stride_s_b, stride_s_m, stride_s_n,
    fp8_max: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    REDUCE_ALONG_N: tl.constexpr,    # True: 规约 N（per-N）；False: 规约 M（per-M）
    X_ROW_MAJOR: tl.constexpr,  # True: x沿着N维度连续；False: x沿着M维度连续
    Y_ROW_MAJOR: tl.constexpr,  # True: y沿着N维度连续；False: y沿着M维度连续
    OUT_IS_E4M3: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    x_base = x_ptr + pid * stride_x_b
    y_base = y_ptr + pid * stride_y_b

    if REDUCE_ALONG_N:
        m0 = bid * BLOCK_M
        amax_m = tl.zeros((BLOCK_M,), dtype=tl.float32)
        x_bp = tl.make_block_ptr(
            base=x_base,
            shape=(M, N),
            strides=(stride_x_m, stride_x_n),
            offsets=(m0, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0) if X_ROW_MAJOR else (0, 1)
        )
        x_bp1 = x_bp
        for _ in range(0, tl.cdiv(N, BLOCK_N)):
            x_tile = tl.load(x_bp1, boundary_check=(0, 1), padding_option='zero')
            x_abs = tl.abs(x_tile)
            amax_m = tl.maximum(amax_m, tl.max(x_abs, axis=1))
            x_bp1 = tl.advance(x_bp1, (0, BLOCK_N))
        safe_amax = tl.maximum(amax_m, EPS)
        scale_m = fp8_max / tl.cast(safe_amax, tl.float32)
        scale_bp = tl.make_block_ptr(
            base=scale_ptr + pid * stride_s_b,
            shape=(M, 1),
            strides=(stride_s_m, stride_s_n),
            offsets=(m0, 0),
            block_shape=(BLOCK_M, 1),
            order=(1, 0)
        )
        tl.store(scale_bp, scale_m[:, None], boundary_check=(0,))
        x_bp2 = x_bp
        y_bp2 = tl.make_block_ptr(
            base=y_base,
            shape=(M, N),
            strides=(stride_y_m, stride_y_n),
            offsets=(m0, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0) if Y_ROW_MAJOR else (0, 1)
        )
        for _ in range(0, tl.cdiv(N, BLOCK_N)):
            x_tile = tl.load(x_bp2, boundary_check=(0, 1), padding_option='zero')
            y_tile = tl.cast(x_tile, tl.float32) * scale_m[:, None]  # 广播到 (BM, BN)
            y_tile = tl.clamp(y_tile, min=-fp8_max, max=fp8_max)
            y_fp8 = tl.cast(y_tile, tl.float8e4nv) if OUT_IS_E4M3 else tl.cast(y_tile, tl.float8e5)
            tl.store(y_bp2, y_fp8, boundary_check=(0, 1))
            x_bp2 = tl.advance(x_bp2, (0, BLOCK_N))
            y_bp2 = tl.advance(y_bp2, (0, BLOCK_N))

    else:
        n0 = bid * BLOCK_N
        amax_n = tl.zeros((BLOCK_N,), dtype=tl.float32)
        x_bp = tl.make_block_ptr(
            base=x_base,                
            shape=(M, N),
            strides=(stride_x_m, stride_x_n),
            offsets=(0, n0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0) if X_ROW_MAJOR else (0, 1)
        )
        x_bp1 = x_bp
        for _ in range(0, tl.cdiv(M, BLOCK_M)):
            x_tile = tl.load(x_bp1, boundary_check=(0, 1), padding_option='zero')  # 检查 M,N 维度的边界
            x_abs = tl.abs(x_tile)
            amax_n = tl.maximum(amax_n, tl.max(x_abs, axis=0))
            x_bp1 = tl.advance(x_bp1, (BLOCK_M, 0))
        safe_amax = tl.maximum(amax_n, EPS)
        scale_n = fp8_max / tl.cast(safe_amax, tl.float32)
        # 把 scale 写到 keepdim (B, 1, N)：只在 m=0 行写入
        scale_bp = tl.make_block_ptr(
            base=scale_ptr + pid * stride_s_b,
            shape=(1, N),
            strides=(stride_s_m, stride_s_n),  # 注意stride_s_m 应对应 keepdim 的 m 维（这里为 1）
            offsets=(0, n0),
            block_shape=(1, BLOCK_N),
            order=(1, 0)
        )
        tl.store(scale_bp, scale_n[None, :], boundary_check=(1,))
        x_bp2 = x_bp
        y_bp2 = tl.make_block_ptr(
            base=y_base,
            shape=(M, N),
            strides=(stride_y_m, stride_y_n),
            offsets=(0, n0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0) if Y_ROW_MAJOR else (0, 1)
        )
        for _ in range(0, tl.cdiv(M, BLOCK_M)):
            x_tile = tl.load(x_bp2, boundary_check=(0, 1), padding_option='zero')
            y_tile = tl.cast(x_tile, tl.float32) * scale_n[None, :]  # 广播到 (BM, BN)
            y_tile = tl.clamp(y_tile, min=-fp8_max, max=fp8_max)
            y_fp8 = tl.cast(y_tile, tl.float8e4nv) if OUT_IS_E4M3 else tl.cast(y_tile, tl.float8e5)
            tl.store(y_bp2, y_fp8, boundary_check=(0, 1))
            x_bp2 = tl.advance(x_bp2, (BLOCK_M, 0))
            y_bp2 = tl.advance(y_bp2, (BLOCK_M, 0))


def fp8_quant_triton_block(
    hp_tensor: torch.Tensor,
    float8_dtype: torch.dtype,
    axiswise_dim: int,
    output_row_major: bool = True,
):
    if axiswise_dim not in (-1, -2):
        raise ValueError("axiswise_dim must be -1 or -2")
    reduce_along_n = True if axiswise_dim == -1 else False

    x = hp_tensor

    stride_x_b, stride_x_m, stride_x_n = x.stride()
    assert stride_x_m == 1 or stride_x_n == 1
    x_row_major = True if stride_x_n == 1 else False

    B, M, N = x.shape
    if output_row_major:
        y = torch.empty((B, M, N), device=x.device, dtype=float8_dtype)
    else:
        y = torch.empty((B, N, M), device=x.device, dtype=float8_dtype)
        y = y.transpose(-1, -2)
    stride_y_b, stride_y_m, stride_y_n = y.stride()
    scale = torch.empty((B, M, 1) if reduce_along_n else (B, 1, N), dtype=torch.float32, device=x.device)
    stride_scale_b, stride_scale_m, stride_scale_n = scale.stride()
    
    fp8_max = float(torch.finfo(float8_dtype).max)
    out_is_e4m3 = float8_dtype == torch.float8_e4m3fn
    grid = lambda META: (B, triton.cdiv(M, META["BLOCK_M"]), ) if reduce_along_n else (B, triton.cdiv(N, META["BLOCK_N"]),)
    fp8_per_channel_quant_kernel_blockptr[grid](
        x, y, scale,
        B, M, N,
        stride_x_b, stride_x_m, stride_x_n,
        stride_y_b, stride_y_m, stride_y_n,
        stride_scale_b, stride_scale_m, stride_scale_n,
        fp8_max=fp8_max,
        OUT_IS_E4M3=out_is_e4m3,
        REDUCE_ALONG_N=reduce_along_n,
        X_ROW_MAJOR=x_row_major,
        Y_ROW_MAJOR=output_row_major,
        EPS=1e-12,
    )
    
    return y, scale



def combine_matmul_shapes(data_dict):
    input_shapes = data_dict.get("input_shape", [])
    weight_shapes = data_dict.get("weight_shape", [])
    
    combined_shapes = set()

    for input_s in input_shapes:
        b_in, m, k_in = input_s
        
        for weight_s in weight_shapes:
            k_wt, n = weight_s
            
            if k_in == k_wt:
                combined_shapes.add((b_in, m, k_in))
                combined_shapes.add((b_in, k_in, n))
    
    return list(combined_shapes)

configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["B", "M", "N"],  # Argument names to use as an x-axis for the plot
        x_vals=combine_matmul_shapes(tensor_info_dict),  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot

        line_vals=["native", "compiled", "triton", "native_col_major", "compiled_col_major", "triton_col_major"],  # Label name for the lines
        line_names=["native", "compiled", "triton", "native_col_major", "compiled_col_major", "triton_col_major"],  # Line styles

        # line_vals=["triton_block"],  # Label name for the lines
        # line_names=["triton_block"],  # Line styles

        styles=[("green", "-"), ("yellow", "-"), ("black", "-"), ("blue", "-"), ("red", "-"), ("pink", "-")],
        # styles=[("green", "-")],

        ylabel="ms",  # Label name for the y-axis
        plot_name="quant-fp8",  # Name for the plot, used also as a file name for saving the plot.
        args={}
    ))


DEVICE = "cuda"


@triton.testing.perf_report(configs)
def benchmark(B, M, N, provider):
    a = torch.rand((B, M, N), device=DEVICE, dtype=torch.bfloat16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'native':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fp8_quant(a, torch.float8_e4m3fn, -1), quantiles=quantiles)
    if provider == "native_col_major":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fp8_quant(a, torch.float8_e4m3fn, -2), quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fp8_quant_triton_block(a, torch.float8_e4m3fn, -1), quantiles=quantiles)
    if provider == "triton_col_major":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fp8_quant_triton_block(a, torch.float8_e4m3fn, -2), quantiles=quantiles)    
    if provider == "compiled":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: compiled_fp8_quant(a, torch.float8_e4m3fn, -1), quantiles=quantiles)
    if provider == "compiled_col_major":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: compiled_fp8_quant(a, torch.float8_e4m3fn, -2), quantiles=quantiles)
    if provider == "triton_block":
        for trans in [False, True]:
            for channel in [-1, -2]:
                if trans:
                    a_input = a.transpose(-1, -2)
                else:
                    a_input = a
                tmp1,tmp2 = fp8_quant_triton_block(a_input, torch.float8_e4m3fn, channel)
                tmp3, tmp4 = fp8_quant(a_input, torch.float8_e4m3fn, channel)
                print(f"{trans=}, {channel=}, {torch.allclose(tmp2, tmp4), torch.allclose(tmp1.to(torch.bfloat16), tmp3.to(torch.bfloat16))}")
                if not torch.allclose(tmp2, tmp4) or not torch.allclose(tmp1.to(torch.bfloat16), tmp3.to(torch.bfloat16)):
                    tmp1_bf16 = tmp1.to(torch.bfloat16)
                    tmp3_bf16 = tmp3.to(torch.bfloat16)
                    diff_mask = tmp1_bf16 != tmp3_bf16
                    coords = torch.nonzero(diff_mask, as_tuple=False)
                    values = (tmp1_bf16 - tmp3_bf16)[diff_mask]
                    for (i, j, k), v in zip(coords.tolist(), values.tolist()):
                        print(f"位置 ({i}, {j}, {k}) 差值 = {v}")
        
        ms , min_ms, max_ms = (1,1,1)
        # ms, min_ms, max_ms = triton.testing.do_bench(lambda: fp8_quant_triton_block(a, torch.float8_e4m3fn, -1), quantiles=quantiles)
    perf = lambda ms: B * M * N * 1e-9 * (2*a.dtype.itemsize + 1 ) / (ms * 1e-3)
    return perf(ms), perf(min_ms), perf(max_ms)

benchmark.run(show_plots=True, print_data=True)
