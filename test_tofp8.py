import torch
import torch.nn.functional as F
import time
def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    # Calculate the scale as dtype max divided by absmax
    scale = finfo.max / x.abs().max().clamp(min=1e-12)
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    return x_scl_sat.to(dtype), scale.float().reciprocal()

def to_float8_per_row(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    # Calculate the scale as dtype max divided by absmax
    scale = finfo.max / x.abs().max(dim=1, keepdim=True).values.clamp(min=1e-12)
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    return x_scl_sat.to(dtype), scale.float().reciprocal()

def compare_f8_mm(size=(64000, 64000), dtype=torch.float8_e4m3fn) -> None:
    # create test inputs
    # Note: cuBLASLt float8 matmul requires column major
    #        for the second argument
    x = torch.randn(size, dtype=torch.bfloat16, device='cuda')
    w = torch.randn(size, dtype=torch.bfloat16, device='cuda').t()
    # do a scaled cast to float8 on the inputs
    x_f8, x_inv_s = to_float8_per_row(x, dtype=dtype)
    w_f8, w_inv_s = to_float8_per_row(w, dtype=dtype)
    w_inv_s = w_inv_s.t()
    print(f'x_f8: {x_f8.shape}, w_f8: {w_f8.shape}')
    print(f'x_inv_s: {x_inv_s.shape}, w_inv_s: {w_inv_s.shape}')
    # perform the float8 matmul
    print(x_f8.device)
    time.sleep(2)
    y,_= torch._scaled_mm(x_f8, w_f8, out_dtype=torch.bfloat16,
                             scale_a=x_inv_s , scale_b=w_inv_s, use_fast_accum=True)
    time.sleep(2)

    print(y.device)
    print(y.shape)
    # compare output of float8 matmul to the fp16 baseline
    # cos_sim = F.cosine_similarity(torch.mm(x, w).reshape(-1),
    #                               y.reshape(-1), dim=0)
    # Cosine similarity between scaled mm and reference
    # should be close to 1.0
    # print(f'cos_sim {cos_sim.item():.4f}')

if __name__ == "__main__":
    compare_f8_mm()