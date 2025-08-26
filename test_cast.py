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
    is_transpose: tl.constexpr,
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

    if is_transpose:
        y_ptr = y_ptr + off_n[None, :] * M + off_m[:, None]
    else:
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
        is_transpose=False,
    )
    return quant_x, scale


@torch.compile(fullgraph=True)
def _batch_granul_to_fp8_quant_v2(x, dtype=torch.float8_e5m2, scale_tol=1e-12):
    BS, M, N = x.shape
    amin, amax = x.aminmax()
    max_abs = torch.maximum(amin.abs(), amax.abs()).clamp(min=scale_tol).to(torch.float32)
    fp8_range = torch.finfo(dtype).max

    scale = fp8_range / max_abs

    quant_x =  (x.to(torch.float32)*scale).clamp(min=-fp8_range, max=fp8_range).to(dtype)

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
    DTYPE = torch.float8_e5m2 # 使用e4m3, 和之前的例子保持一致
    WARMUP, REPS = 100, 10

    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        exit()

    device = torch.device("cuda:0")
    # 创建一个动态范围在batch维度上变化的输入数据，以更好地体现Per-Batch量化的优势
    base_data = torch.randn(B, M, N, device=device)

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
        _ = _batch_granul_to_fp8_quant(base_data, dtype=DTYPE)

    for _ in range(WARMUP):
        _ = _batch_granul_to_fp8_quant_v2(base_data, dtype=DTYPE)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(WARMUP):
        _ = _batch_granul_to_fp8_quant(base_data, dtype=DTYPE)
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(REPS):
        _ = _batch_granul_to_fp8_quant(base_data, dtype=DTYPE)
    end_event.record()
    torch.cuda.synchronize()
    time_original = start_event.elapsed_time(end_event) / REPS
    print(f"原始函数 : {time_original:.4f} ms")

    for _ in range(WARMUP):
        _ = _batch_granul_to_fp8_quant_v2(base_data, dtype=DTYPE)
    torch.cuda.synchronize()

    # 2. 测试优化函数
    start_event.record()
    for _ in range(REPS):
        _ = _batch_granul_to_fp8_quant_v2(base_data, dtype=DTYPE)
    end_event.record()
    torch.cuda.synchronize()
    time_optimized = start_event.elapsed_time(end_event) / REPS
    print(f"优化函数 :  {time_optimized:.4f} ms")
    print("-" * 60)

    time.sleep(2)

    # --- 精度与功能对比 ---
    print("--- 精度与功能对比 (Accuracy and Functional Comparison) ---")
    
    quant_x_orig, scale_orig = _batch_granul_to_fp8_quant(base_data.clone(), dtype=DTYPE)
    quant_x_opt, scale_opt = _batch_granul_to_fp8_quant_v2(base_data.clone(), dtype=DTYPE)

    # 1. 对比 Scale
    print(f"原始函数的Scale (标量): {scale_orig.item():.4f}")
    print(f"优化函数的Scale (标量): {scale_opt.item():.4f}")

    # 2. 对比精度
    # 反量化回 FP16
    dequant_x_orig = (quant_x_orig.to(torch.float16) / scale_orig).clamp(-65504, 65504)
    # 需要 unsqueeze 来进行正确的广播
    dequant_x_opt = (quant_x_opt.to(torch.float16) / scale_opt).clamp(-65504, 65504)

    # 1. 验证 Scale 是否一致
    # scale_orig 是 CPU float, scale_opt 是 GPU 0-dim tensor
    scale_close = math.isclose(scale_orig, scale_opt.item(), rel_tol=1e-6)
    print(f"Scale值是否一致: {scale_close}")
    print(f"  - 原始函数 scale: {scale_orig:.6f}")
    print(f"  - 优化函数 scale: {scale_opt.item():.6f}")

    # 2. 验证量化结果是否完全相同
    outputs_equal = torch.allclose(dequant_x_orig, dequant_x_opt)
    print(f"量化后的张量是否完全相同: {outputs_equal}")

    # 3. 计算信噪比 (两者结果应几乎一致)
    dequant_x = (quant_x_opt.to(torch.float16) / scale_opt).clamp(-65504, 65504)
    snr_val = snr(base_data, dequant_x)
    print(f"\n两种方法得到的信噪比 (SNR) 均为: {snr_val:.2f} dB")
    print("-" * 70)

    # --- 总结 ---
    print("--- 总结 (Conclusion) ---")
    if outputs_equal and scale_close:
        print("✅ 功能验证通过: 两种方法计算出的 scale 和最终量化张量几乎完全相同。")
        speed_gain = (time_original / time_optimized)
        print(f"⚡️ 性能对比: 优化后的函数速度是原始函数的 {speed_gain:.2f} 倍。")
        print("   这证明了消除 GPU->CPU->GPU 的同步点可以带来巨大的性能提升，")
        print("   即使最终的计算结果是完全一样的。")
    else:
        print("❌ 功能验证失败: 两种方法的输出不一致，请检查代码。")
