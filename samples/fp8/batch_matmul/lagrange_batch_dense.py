import argparse
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _aligned_dim(dim: int, alignment: int = 16) -> int:
    if alignment <= 0:
        return dim
    return (dim + alignment - 1) // alignment * alignment


def _to_float8_per_row(
    tensor: torch.Tensor,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    finfo = torch.finfo(dtype)
    scale = finfo.max / tensor.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scaled = (tensor * scale).clamp(min=finfo.min, max=finfo.max)
    return scaled.to(dtype), scale.float().reciprocal()


@dataclass
class BatchDenseResult:
    output: torch.Tensor
    reference: torch.Tensor
    cosine_similarity: float


class FP8BatchMatMulDense(nn.Module):
    """Batch-specific dense layer implemented with torch._scaled_mm FP8 inputs."""

    def __init__(
        self,
        batch: int,
        in_features: int,
        units: int,
        activation: Optional[nn.Module] = None,
        use_bias: bool = True,
        fp8_all_gather: bool = False,
        alignment: int = 16,
    ):
        super().__init__()
        self.batch = batch
        self.in_features = in_features
        self.units = units
        self.activation = activation
        self.fp8_all_gather = fp8_all_gather

        self.aligned_in_features = _aligned_dim(in_features, 16)
        self.aligned_units = _aligned_dim(units, 16)
        self.sequence_alignment = alignment

        weight_shape = (batch, self.aligned_in_features, self.aligned_units)
        self.weight = nn.Parameter(torch.empty(weight_shape, dtype=torch.bfloat16))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(batch, 1, self.aligned_units, dtype=torch.bfloat16))
        else:
            self.register_parameter("bias", None)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._run_fp8(inputs).output

    def _run_fp8(self, inputs: torch.Tensor) -> BatchDenseResult:
        if inputs.ndim != 3:
            raise ValueError(f"Expected [batch, seq_len, in_features], got {tuple(inputs.shape)}")
        if inputs.shape[0] != self.batch:
            raise ValueError(f"Batch mismatch: input has {inputs.shape[0]}, module expects {self.batch}")
        if inputs.shape[2] != self.in_features:
            raise ValueError(f"Feature mismatch: input has {inputs.shape[2]}, module expects {self.in_features}")

        seq_len = inputs.shape[1]
        aligned_seq_len = _aligned_dim(seq_len, self.sequence_alignment)
        padded_inputs = F.pad(
            inputs.to(torch.bfloat16),
            (0, self.aligned_in_features - self.in_features, 0, aligned_seq_len - seq_len),
        )

        outputs = []
        references = []
        for batch_idx in range(self.batch):
            x = padded_inputs[batch_idx]
            weight = self.weight[batch_idx]

            x_f8, x_inv_scale = _to_float8_per_row(x)
            weight_f8_t, weight_inv_scale_t = _to_float8_per_row(weight.t().contiguous())

            out = torch._scaled_mm(
                x_f8,
                weight_f8_t.t(),
                out_dtype=torch.bfloat16,
                scale_a=x_inv_scale,
                scale_b=weight_inv_scale_t.t(),
                use_fast_accum=True,
            )
            ref = torch.mm(x, weight)
            outputs.append(out)
            references.append(ref)

        output = torch.stack(outputs, dim=0)
        reference = torch.stack(references, dim=0)

        if self.bias is not None:
            output = output + self.bias
            reference = reference + self.bias

        output = output[:, :seq_len, : self.units]
        reference = reference[:, :seq_len, : self.units]

        if self.activation is not None:
            output = self.activation(output)
            reference = self.activation(reference)

        cosine = F.cosine_similarity(output.float().reshape(-1), reference.float().reshape(-1), dim=0)
        return BatchDenseResult(output=output, reference=reference, cosine_similarity=float(cosine.item()))

    def extra_repr(self) -> str:
        return (
            f"batch={self.batch}, in_features={self.in_features}, "
            f"aligned_in_features={self.aligned_in_features}, units={self.units}, "
            f"aligned_units={self.aligned_units}, fp8_all_gather={self.fp8_all_gather}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="FP8 batch-specific dense layer smoke.")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--in-features", type=int, default=64)
    parser.add_argument("--out-features", type=int, default=64)
    parser.add_argument("--min-cosine", type=float, default=0.98)
    parser.add_argument("--no-fp8-all-gather", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This sample requires CUDA.")
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("This sample requires PyTorch float8 support.")

    torch.manual_seed(0)
    module = FP8BatchMatMulDense(
        batch=args.batch,
        in_features=args.in_features,
        units=args.out_features,
        fp8_all_gather=not args.no_fp8_all_gather,
    ).cuda()
    x = torch.randn(
        args.batch,
        args.seq_len,
        args.in_features,
        device="cuda",
        dtype=torch.bfloat16,
    )

    result = module._run_fp8(x)
    torch.cuda.synchronize()

    print(f"FP8 batch dense smoke passed, output_shape={tuple(result.output.shape)}")
    print(f"cosine_similarity={result.cosine_similarity:.4f}")
    if result.cosine_similarity < args.min_cosine:
        raise AssertionError(
            f"FP8 output is too far from BF16 reference: {result.cosine_similarity:.4f} < {args.min_cosine:.4f}"
        )


if __name__ == "__main__":
    main()
