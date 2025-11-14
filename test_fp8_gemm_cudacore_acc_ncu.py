import torch

import triton
import triton.language as tl

def create_fp8_bmm_per_channel_nt_kernel(autotune_configs, autotune_key=['M', 'N', 'K']):
    @triton.autotune(
        configs=autotune_configs, 
        key=autotune_key,
    )
    @triton.jit
    def _fp8_bmm_per_channel_nt(
        A, B, C, bias, A_scale, B_scale,
        M, N, K,
        stride_ab, stride_am, stride_ak,
        stride_bb, stride_bn, stride_bk,
        stride_cb, stride_cm, stride_cn,
        stride_biasb, stride_biasm, stride_biasn,
        stride_a_scaleb, stride_a_scalem, stride_a_scalek,
        stride_b_scaleb, stride_b_scalek, stride_b_scalen,
        USE_BIAS: tl.constexpr,
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

        A_scale_ptr = tl.make_block_ptr(
            A_scale + bid * stride_a_scaleb,
            shape = (M, 1),
            strides=(stride_a_scalem, stride_a_scalek),
            offsets=(pid_m * BLOCK_SIZE_M, 0),
            block_shape=(BLOCK_SIZE_M, 1),
            order=(1, 0),
        )

        B_scale_ptr = tl.make_block_ptr(
            B_scale + bid * stride_b_scaleb,
            shape = (1, N),
            strides=(stride_b_scalek, stride_b_scalen),
            offsets=(0, pid_n * BLOCK_SIZE_N),
            block_shape=(1, BLOCK_SIZE_N),
            order=(1, 0),
        )

        # compute
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        partial_accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        A_SCALING_FACTOR = tl.load(A_scale_ptr, boundary_check=(0, 1), padding_option='zero')
        B_SCALING_FACTOR = tl.load(B_scale_ptr, boundary_check=(0, 1), padding_option='zero')            
        SCALING_FACTOR = A_SCALING_FACTOR * B_SCALING_FACTOR

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
            partial_accumulator = tl.dot(A_block, B_block, partial_accumulator)

            if (k + 1) % 4 == 0:
                accumulator += partial_accumulator
                partial_accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            A_ptr = tl.advance(A_ptr, (0, BLOCK_SIZE_K))
            B_ptr = tl.advance(B_ptr, (BLOCK_SIZE_K, 0))

        accumulator += partial_accumulator
    
        # dequantize
        # WARNING: THIS IS DON"T BUG
        accumulator = accumulator / SCALING_FACTOR

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
    return _fp8_bmm_per_channel_nt

forward_kernel_configs = [
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}, num_stages=4, num_warps=4),
]

forward_kernel = create_fp8_bmm_per_channel_nt_kernel(forward_kernel_configs)

B, M, K, N = 32, 1024, 1024, 1024

A_mat_fp8 = torch.ones(B, M, K, dtype=torch.float8_e5m2, device="cuda")
B_mat_fp8 = torch.ones(B, K, N, dtype=torch.float8_e4m3fn, device="cuda")
A_mat_scale = torch.randn(B, M, 1, device="cuda")
B_mat_scale = torch.randn(B, 1, N, device="cuda")
C_mat = torch.empty((B, M, N), device=A_mat_fp8.device, dtype=torch.bfloat16)

grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), B, )

forward_kernel[grid](
    A_mat_fp8, B_mat_fp8, C_mat, None, A_mat_scale, B_mat_scale,
    M, N, K,
    A_mat_fp8.stride(0), A_mat_fp8.stride(1), A_mat_fp8.stride(2),
    B_mat_fp8.stride(0), B_mat_fp8.stride(2), B_mat_fp8.stride(1),
    C_mat.stride(0), C_mat.stride(1), C_mat.stride(2),
    0,0,0,
    A_mat_scale.stride(0), A_mat_scale.stride(1), A_mat_scale.stride(2),
    B_mat_scale.stride(0), B_mat_scale.stride(1), B_mat_scale.stride(2),
    USE_BIAS=False,
    OUT_DTYPE="torch.bfloat16",
)
