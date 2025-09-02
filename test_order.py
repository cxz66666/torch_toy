import torch
import triton
import triton.language as tl

@triton.autotune(
configs = [
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64,  'GROUP_SIZE_M': 1}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64,  'GROUP_SIZE_M': 4}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64,  'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
],
    key=['M', 'N', 'K'],
)
@triton.jit
def load_block_kernel(
    A, B,
    M, N,
    stride_ab, stride_am, stride_an,
    stride_bb, stride_bm, stride_bn,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
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
        shape=(M, N),
        strides=(stride_am, stride_an),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    B_ptr = tl.make_block_ptr(
        B + bid * stride_bb,
        shape=(M, N),
        strides=(stride_bm, stride_bn),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    
    a = tl.load(A_ptr)

    # write back
    tl.store(
        B_ptr, 
        a, 
        boundary_check=(0, 1),
    )



def test_load(load_block_order:bool):
    bs, m, n = 4, 4, 4
    a = torch.randn((bs, m, n), dtype=torch.float16, device='cuda')
    b = torch.zeros((bs, m, n), dtype=a.dtype, device="cuda")

    grid = lambda META: (triton.cdiv(m, META["BLOCK_SIZE_M"]) * triton.cdiv(n, META["BLOCK_SIZE_N"]), bs, )
    load_block_kernel[grid](a, b, m, n, a.stride(0), a.stride(1), a.stride(2), b.stride(0), b.stride(1), b.stride(2))

    print(a)
    print(b)
    assert torch.allclose(a, b) is True

test_load(False)
test_load(True)
