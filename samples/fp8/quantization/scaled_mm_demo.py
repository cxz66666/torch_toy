import argparse

import torch
import torch.nn.functional as F


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

def compare_f8_mm(m: int, n: int, k: int, dtype=torch.float8_e4m3fn) -> None:
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    w_source = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")

    x_f8, x_inv_s = to_float8_per_row(x, dtype=dtype)
    w_f8, w_inv_s = to_float8_per_row(w_source, dtype=dtype)
    w = w_source.t()
    w_f8 = w_f8.t()
    w_inv_s = w_inv_s.t()

    bias = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    y = torch._scaled_mm(
        x_f8,
        w_f8,
        out_dtype=torch.bfloat16,
        bias=bias,
        scale_a=x_inv_s,
        scale_b=w_inv_s,
        use_fast_accum=True,
    )

    reference = torch.mm(x, w) + bias
    cos_sim = F.cosine_similarity(reference.reshape(-1), y.reshape(-1), dim=0)
    print(f"x_f8={tuple(x_f8.shape)}, w_f8={tuple(w_f8.shape)}, y={tuple(y.shape)}")
    print(f"x_inv_s={tuple(x_inv_s.shape)}, w_inv_s={tuple(w_inv_s.shape)}")
    print(f"cos_sim={cos_sim.item():.4f}")


def main():
    parser = argparse.ArgumentParser(description="torch._scaled_mm FP8 smoke")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This sample requires CUDA.")

    compare_f8_mm(args.m, args.n, args.k)


if __name__ == "__main__":
    main()
