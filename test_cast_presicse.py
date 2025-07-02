import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}, num_stages=7, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_stages=7, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_stages=7, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256}, num_stages=7, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def _to_fp8_quant_kernel(
    x_ptr, 
    y_ptr,
    M, N,
    stride_xb, stride_xm, stride_xn,
    scale,
    fp8_range: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    bid = tl.program_id(1)
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    x_ptr = x_ptr + bid * stride_xb
    off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    x_ptr = x_ptr + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn
    x = tl.load(x_ptr, mask=(off_m[:, None] < M) & (off_n[None, :] < N), other=0.0)
    scale_factor = tl.load(scale)
    value = tl.clamp(x.to(tl.float32) * scale_factor, min=-fp8_range, max=fp8_range)
    value = value.to(y_ptr.dtype.element_ty)
    y_ptr = y_ptr + bid * M * N
    y_ptr = y_ptr + off_m[:, None] * N + off_n[None, :]
    tl.store(y_ptr, value, mask=(off_m[:, None] < M) & (off_n[None, :] < N))

def _batch_granul_to_fp8_quant(x, dtype=torch.float8_e5m2, scale_tol=1e-12):
    amin, amax = x.aminmax()
    max_abs = torch.maximum(amin.abs(), amax.abs()).clamp(min=scale_tol).to(torch.float32)
    fp8_range = torch.finfo(dtype).max
    scale = fp8_range / max_abs
    BS, M, N = x.shape
    quant_x = torch.empty_like(x, device=x.device, dtype=dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), BS,)
    _to_fp8_quant_kernel[grid](
        x, quant_x, M, N,
        x.stride(0), x.stride(1), x.stride(2),
        scale,
        fp8_range=fp8_range,
    )
    return quant_x, scale


def snr(y_true, y_pred):
    """计算信噪比"""
    y_true = y_true.to(torch.float32)
    y_pred = y_pred.to(torch.float32)
    noise = y_true - y_pred
    signal_power = torch.mean(y_true**2)
    noise_power = torch.mean(noise**2)
    snr_val = 10 * torch.log10(signal_power / noise_power)
    return snr_val.item()


if __name__ == "__main__":
    # --- 测试设置 ---
    B, M, N = 32, 2240, 1120
    DTYPE = torch.float8_e4m3fn # 使用e4m3, 和之前的例子保持一致
    WARMUP, REPS = 100, 10

    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        exit()

    device = torch.device("cuda:0")
    # 创建一个动态范围在batch维度上变化的输入数据，以更好地体现Per-Batch量化的优势

    print(f"测试环境: PyTorch {torch.__version__}, Triton {triton.__version__}")
    print(f"测试设备: {torch.cuda.get_device_name(0)}")
    print(f"测试张量形状: B={B}, M={M}, N={N}")
    print("-" * 60)

    import time
    time.sleep(1)
    # --- 速度测试 ---
    print("--- 速度测试 (Speed Test) ---")
    print(f"(预热 {WARMUP} 次, 重复 {REPS} 次)")

    # 1. 测试原始函数
    for _ in range(WARMUP):
        base_data = torch.randn(B, M, N, device=device)
        quant_data, scale = _batch_granul_to_fp8_quant(base_data, dtype=DTYPE)
        requant_data = (quant_data.to(torch.float32)) / scale
        print(torch.nn.functional.mse_loss(requant_data, base_data))
