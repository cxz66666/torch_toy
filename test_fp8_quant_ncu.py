import torch

import triton
import triton.language as tl

# /usr/local/cuda/bin/ncu --print-source cuda,ptx --import-source 1 --page source -o ncu_report%i --set full python3 ./test_fp8_quant_ncu.py

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def fp8_per_channel_quant_kernel_blockptr(
    x_ptr,
    y_ptr,
    scale_ptr,
    B,
    M,
    N,
    stride_x_b,
    stride_x_m,
    stride_x_n,
    stride_y_b,
    stride_y_m,
    stride_y_n,
    stride_s_b,
    stride_s_m,
    stride_s_n,
    fp8_max: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    REDUCE_ALONG_N: tl.constexpr,  # True: 规约 N（per-N）；False: 规约 M（per-M）
    X_ROW_MAJOR: tl.constexpr,  # True: x沿着N维度连续；False: x沿着M维度连续
    Y_ROW_MAJOR: tl.constexpr,  # True: y沿着N维度连续；False: y沿着M维度连续
    EPS: tl.constexpr,
    dep_token,  # useless
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
            order=(1, 0) if X_ROW_MAJOR else (0, 1),
        )
        x_bp1 = x_bp
        for _ in range(0, tl.cdiv(N, BLOCK_N)):
            x_tile = tl.load(x_bp1, boundary_check=(0, 1), padding_option="zero")
            x_abs = tl.abs(x_tile)
            amax_m = tl.maximum(amax_m, tl.max(x_abs, axis=1))
            x_bp1 = tl.advance(x_bp1, (0, BLOCK_N))
        safe_amax = tl.maximum(amax_m, EPS)
        scale_m = tl.div_rn(fp8_max, tl.cast(safe_amax, tl.float32))
        reciprocal_scale_m = tl.div_rn(tl.cast(safe_amax, tl.float32), fp8_max)
        scale_bp = tl.make_block_ptr(
            base=scale_ptr + pid * stride_s_b,
            shape=(M, 1),
            strides=(stride_s_m, stride_s_n),
            offsets=(m0, 0),
            block_shape=(BLOCK_M, 1),
            order=(1, 0),
        )
        tl.store(scale_bp, reciprocal_scale_m[:, None], boundary_check=(0,))
        x_bp2 = x_bp
        y_bp2 = tl.make_block_ptr(
            base=y_base,
            shape=(M, N),
            strides=(stride_y_m, stride_y_n),
            offsets=(m0, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0) if Y_ROW_MAJOR else (0, 1),
        )
        for _ in range(0, tl.cdiv(N, BLOCK_N)):
            x_tile = tl.load(x_bp2, boundary_check=(0, 1), padding_option="zero")
            y_tile = tl.cast(x_tile, tl.float32) * scale_m[:, None]  # 广播到 (BM, BN)
            y_tile = tl.clamp(y_tile, min=-fp8_max, max=fp8_max)
            y_fp8 = y_tile.to(y_ptr.dtype.element_ty)
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
            order=(1, 0) if X_ROW_MAJOR else (0, 1),
        )
        x_bp1 = x_bp
        for _ in range(0, tl.cdiv(M, BLOCK_M)):
            x_tile = tl.load(
                x_bp1, boundary_check=(0, 1), padding_option="zero"
            )  # 检查 M,N 维度的边界
            x_abs = tl.abs(x_tile)
            amax_n = tl.maximum(amax_n, tl.max(x_abs, axis=0))
            x_bp1 = tl.advance(x_bp1, (BLOCK_M, 0))
        safe_amax = tl.maximum(amax_n, EPS)
        scale_n = tl.div_rn(fp8_max, tl.cast(safe_amax, tl.float32))
        reciprocal_scale_n = tl.div_rn(tl.cast(safe_amax, tl.float32), fp8_max)
        # 把 scale 写到 keepdim (B, 1, N)：只在 m=0 行写入
        scale_bp = tl.make_block_ptr(
            base=scale_ptr + pid * stride_s_b,
            shape=(1, N),
            strides=(stride_s_m, stride_s_n),  # 注意stride_s_m 应对应 keepdim 的 m 维（这里为 1）
            offsets=(0, n0),
            block_shape=(1, BLOCK_N),
            order=(1, 0),
        )
        tl.store(scale_bp, reciprocal_scale_n[None, :], boundary_check=(1,))
        x_bp2 = x_bp
        y_bp2 = tl.make_block_ptr(
            base=y_base,
            shape=(M, N),
            strides=(stride_y_m, stride_y_n),
            offsets=(0, n0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0) if Y_ROW_MAJOR else (0, 1),
        )
        for _ in range(0, tl.cdiv(M, BLOCK_M)):
            x_tile = tl.load(x_bp2, boundary_check=(0, 1), padding_option="zero")
            y_tile = tl.cast(x_tile, tl.float32) * scale_n[None, :]  # 广播到 (BM, BN)
            y_tile = tl.clamp(y_tile, min=-fp8_max, max=fp8_max)
            y_fp8 = y_tile.to(y_ptr.dtype.element_ty)
            tl.store(y_bp2, y_fp8, boundary_check=(0, 1))
            x_bp2 = tl.advance(x_bp2, (BLOCK_M, 0))
            y_bp2 = tl.advance(y_bp2, (BLOCK_M, 0))


def fp8_quant_triton_block(
    x: torch.Tensor,
    float8_dtype: torch.dtype,
    axiswise_dim: int,
    output_row_major: bool = True,
):
    if axiswise_dim not in (-1, -2):
        raise ValueError("axiswise_dim must be -1 or -2")
    reduce_along_n = True if axiswise_dim == -1 else False
    
    x_row_major = x.is_contiguous()

    B, M, N = x.shape
    if output_row_major:
        y = torch.empty((B, M, N), device=x.device, dtype=float8_dtype)
    else:
        y = torch.empty((B, N, M), device=x.device, dtype=float8_dtype)
        y = y.transpose(-1, -2)
    scale = torch.empty((B, M, 1) if reduce_along_n else (B, 1, N), dtype=torch.float32, device=x.device)
    stride_scale_b, stride_scale_m, stride_scale_n = scale.stride()
    
    fp8_max = float(torch.finfo(float8_dtype).max)

    grid = lambda META: (B, triton.cdiv(M, META["BLOCK_M"]), ) if reduce_along_n else (B, triton.cdiv(N, META["BLOCK_N"]),)
    fp8_per_channel_quant_kernel_blockptr[grid](
        x, y, scale,
        B, M, N,
        x.stride(0), x.stride(1), x.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        stride_scale_b, stride_scale_m, stride_scale_n,
        fp8_max=fp8_max,
        REDUCE_ALONG_N=reduce_along_n,
        X_ROW_MAJOR=x_row_major,
        Y_ROW_MAJOR=output_row_major,
        EPS=1e-12,
        dep_token=None
    )
    
    return y, scale


@triton.autotune(
    configs=[
        # triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=8),
        # triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=8),
        # triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=8),
        # triton.Config({'BLOCK_SIZE': 2048}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=4, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def _block_reduce_minmax_kernel(x_ptr, n_elements,
                                out_abs_max_ptr,
                                BLOCK_SIZE: tl.constexpr):
    """
    第一阶段：并行块归约。每个 program 处理多个 stride 的 BLOCK_SIZE 切片，
    最终将一个局部 min/max 写到 out_min/out_max 的对应位置。
    """
    pid = tl.program_id(0)
    stride = tl.num_programs(0) * BLOCK_SIZE
    start = pid * BLOCK_SIZE
    # 标量累积器（float32）
    local_min = float('inf')
    local_max = -float('inf')
    offs = start + tl.arange(0, BLOCK_SIZE)
    # grid-stride loop
    while tl.max(offs, axis=0) < n_elements:
        mask = offs < n_elements
        # 以 float32 归约（输入可为 fp16/bf16/fp32）
        x = tl.load(x_ptr + offs, mask=mask, other=0.)
        # 局部块内归约 -> 标量
        blk_min = tl.min(x, axis=0)
        blk_max = tl.max(x, axis=0)
        local_min = tl.minimum(local_min, blk_min)
        local_max = tl.maximum(local_max, blk_max)
        offs += stride
    
    out_abs_max = tl.maximum(tl.abs(local_min), tl.abs(local_max))
    tl.store(out_abs_max_ptr + pid, out_abs_max)

@triton.autotune(
    configs=[
        # triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=8),
        # triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=8),
        # triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=8),
        # triton.Config({'BLOCK_SIZE': 2048}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=4, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def _block_reduce_minmax_kernel_tma(x_ptr, n_elements,
                                    out_abs_max_ptr,
                                    BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    # --- 1. 设置 Grid-Stride Loop 的初始参数 ---
    num_programs = tl.num_programs(0)
    stride = num_programs * BLOCK_SIZE
    start_offset = pid * BLOCK_SIZE
    # --- 2. 创建一个 Tensor Pointer ---
    # 这个指针描述了整个输入张量 x 的元数据
    # shape 和 strides 描述的是全局张量，而不是我们要加载的块
    
    # 标量累积器
    local_min = float('inf')
    local_max = -float('inf')
    # --- 3. 修改 Grid-Stride Loop ---
    # 循环条件更简洁，我们只需要检查当前处理的起始偏移量是否越界
    while start_offset < n_elements:
        # --- 4. 使用 TMA 加载数据 ---
        # tl.load 现在接收一个 tensor pointer
        # boundary_check=(0,) 告诉 Triton 在第 0 维上进行边界检查
        # 越界的部分会自动用 0.0 填充，替代了手动 mask 和 other=0.
        x = tl._experimental_descriptor_load(desc_pointer=x_ptr,
                                        offsets=[start_offset], 
                                      shape=[BLOCK_SIZE],
                                      dtype=x_ptr.dtype.element_ty
                                      )
        # 局部块内归约 -> 标量 (这部分逻辑不变)
        blk_min = tl.min(x, axis=0)
        blk_max = tl.max(x, axis=0)
        local_min = tl.minimum(local_min, blk_min)
        local_max = tl.maximum(local_max, blk_max)
        # --- 5. 移动指针到下一个数据块 ---
        # 使用 tl.advance 将指针向前移动 `stride` 个元素
        start_offset += stride
    # 归约和存储逻辑不变
    out_abs_max = tl.maximum(tl.abs(local_min), tl.abs(local_max))
    tl.store(out_abs_max_ptr + pid, out_abs_max)

@torch.compile
def fp8_quant(x):
    amin, amax = x.aminmax()
    amin, amax = amin.to(torch.float32), amax.to(torch.float32)
    max_abs = torch.maximum(amin.abs(), amax.abs()).clamp(min=1e-12)
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    scale = fp8_max / max_abs
    reciprocal_scale = max_abs / fp8_max
    return scale, reciprocal_scale

def triton_fp8_quant(x):
    x_flat = x.contiguous().view(-1)
    n = x_flat.numel()
    partial_max = torch.empty(32*1024*4, device=x.device, dtype=x.dtype)
    _block_reduce_minmax_kernel_tma[(lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),))](
        x_flat, n,
        partial_max,
    )

a = torch.randn((32, 1024, 2048), device="cuda", dtype=torch.bfloat16)

# fp8_quant_triton_block(a, torch.float8_e4m3fn, -1)
triton_fp8_quant(a)