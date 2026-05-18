import argparse

import torch

import triton
import triton.language as tl


def create_fp8_bmm_per_channel_nt_kernel(autotune_configs, autotune_key=("M", "N", "K")):
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
        ACCUMULATE_EVERY: tl.constexpr,
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

            if ACCUMULATE_EVERY > 0 and (k + 1) % ACCUMULATE_EVERY == 0:
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


FORWARD_KERNEL_CONFIGS = [
    triton.Config(
        {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1},
        num_stages=4,
        num_warps=4,
    ),
]


forward_kernel = create_fp8_bmm_per_channel_nt_kernel(FORWARD_KERNEL_CONFIGS)


def run_smoke(batch: int, m: int, k: int, n: int, accumulate_every: int) -> torch.Tensor:
    a_mat_fp8 = torch.ones(batch, m, k, dtype=torch.float8_e5m2, device="cuda")
    b_mat_fp8 = torch.ones(batch, k, n, dtype=torch.float8_e4m3fn, device="cuda")
    a_mat_scale = torch.randn(batch, m, 1, device="cuda")
    b_mat_scale = torch.randn(batch, 1, n, device="cuda")
    c_mat = torch.empty((batch, m, n), device=a_mat_fp8.device, dtype=torch.bfloat16)

    grid = lambda META: (
        triton.cdiv(m, META["BLOCK_SIZE_M"]) * triton.cdiv(n, META["BLOCK_SIZE_N"]),
        batch,
    )

    forward_kernel[grid](
        a_mat_fp8, b_mat_fp8, c_mat, None, a_mat_scale, b_mat_scale,
        m, n, k,
        a_mat_fp8.stride(0), a_mat_fp8.stride(1), a_mat_fp8.stride(2),
        b_mat_fp8.stride(0), b_mat_fp8.stride(2), b_mat_fp8.stride(1),
        c_mat.stride(0), c_mat.stride(1), c_mat.stride(2),
        0, 0, 0,
        a_mat_scale.stride(0), a_mat_scale.stride(1), a_mat_scale.stride(2),
        b_mat_scale.stride(0), b_mat_scale.stride(1), b_mat_scale.stride(2),
        USE_BIAS=False,
        OUT_DTYPE="torch.bfloat16",
        ACCUMULATE_EVERY=accumulate_every,
    )
    return c_mat


def main():
    parser = argparse.ArgumentParser(description="FP8 per-channel BMM NCU smoke")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument(
        "--accumulate-every",
        type=int,
        default=0,
        help="Flush partial accumulator every N K-tiles; 0 keeps the default accumulation path.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This sample requires CUDA.")

    out = run_smoke(args.batch, args.m, args.k, args.n, args.accumulate_every)
    torch.cuda.synchronize()
    print(f"finished FP8 BMM smoke, output_shape={tuple(out.shape)}")


if __name__ == "__main__":
    main()
