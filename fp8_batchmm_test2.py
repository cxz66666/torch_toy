import torch
import torch.nn as nn
from typing import Optional
import torch.utils.checkpoint as checkpoint

from bytedance.lagrange_torch.ops.float8.float8_cast_tensor import WeightWithDynamicFloat8CastTensor
from bytedance.lagrange_torch.ops.float8.float8_tensor import (
    GemmInputRole,
    LinearMMConfig,
    ScaledMMConfig,
    hp_tensor_and_scale_to_float8,
)

from bytedance.lagrange_torch.ops.float8.float8_scaling_utils import (
    get_maybe_axiswise_dim,
    hp_tensor_to_float8_dynamic,
)

from bytedance.lagrange_torch.ops.float8.config import Float8LinearConfig, ScalingType
from bytedance.lagrange_torch.ops.float8.distributed_utils import tensor_already_casted_to_fp8
from bytedance.lagrange_torch.ops.float8.float8_utils import tensor_to_scale
from bytedance.lagrange_torch.module.float8.float8_linear import matmul_with_hp_or_float8_args

def _get_weight_scale(
    weight: torch.Tensor,
    scaling_type_weight: ScalingType,
    config: Float8LinearConfig,
) -> Optional[torch.Tensor]:
    if tensor_already_casted_to_fp8(weight):
        return None
    assert scaling_type_weight is ScalingType.DYNAMIC
    return tensor_to_scale(weight, config.cast_config_weight.target_dtype)


def _cast_weight_to_float8_t(
    weight: torch.Tensor,
    config: Float8LinearConfig,
    linear_mm_config: LinearMMConfig,
    weight_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if tensor_already_casted_to_fp8(weight):
        return weight
    weight_fp8 = hp_tensor_and_scale_to_float8(
        weight,
        weight_scale,
        config.cast_config_weight.target_dtype,
        linear_mm_config,
        gemm_input_role=GemmInputRole.WEIGHT,
    )
    return weight_fp8

def _aligned_dim(dim: int, alignment: int = 0) -> int:
    """Aligns the dimension to the nearest multiple of the alignment."""
    if alignment <= 0:
        return dim
    return (dim + alignment - 1) // alignment * alignment

class FP8BatchMatMulDense(nn.Module):
    def __init__(
        self,
        batch: int,        
        in_features: int,
        units: int,
        activation=None,        
        use_bias=True,
        kernel_initializer=torch.nn.init.normal_, 
        bias_initializer=torch.nn.init.zeros_,
        fp8_all_gather=True,
        fp8_force_recompute=True,
        alignment = 8,
        **kwargs
    ):
        super(FP8BatchMatMulDense, self).__init__()

        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        self.no_clip = kwargs.get("no_clip", False)

        self.token_num = batch
        if alignment > 0:
            self.aligned_token_num = _aligned_dim(batch, alignment)
        else:
            self.aligned_token_num = batch
        self.token_dim = in_features
        self.aligend_token_dim = _aligned_dim(in_features, 16) 
        self.units = units
        self.aligned_units = _aligned_dim(units, 16)

        self.fp8_all_gather = fp8_all_gather
        self.force_recompute = fp8_force_recompute
        config = Float8LinearConfig(
                enable_fsdp_float8_all_gather=self.fp8_all_gather,
                force_recompute_fp8_weight_in_bwd=self.force_recompute,
        )

        self.config = config

        self.linear_mm_config = LinearMMConfig(
            # output
            ScaledMMConfig(
                config.emulate,
                self.config.gemm_config_output.use_fast_accum,
                False,
                self.config.pad_inner_dim,
            ),
            # grad_input
            ScaledMMConfig(
                config.emulate,
                self.config.gemm_config_grad_input.use_fast_accum,
                False,
                self.config.pad_inner_dim,
            ),
            # grad_weight
            ScaledMMConfig(
                config.emulate,
                self.config.gemm_config_grad_weight.use_fast_accum,
                False,
                self.config.pad_inner_dim,
            ),
        )

        kernel_shape = (self.aligned_token_num, self.aligend_token_dim, self.aligned_units)
        self.weight = torch.empty(kernel_shape)
        self.kernel_initializer(self.weight)

        if self.fp8_all_gather:
            self.weight = torch.nn.Parameter(
                    WeightWithDynamicFloat8CastTensor(
                        self.weight,
                        self.linear_mm_config,
                        self.config.cast_config_weight.target_dtype,
                    ),
                )
        else:
            self.weight = torch.nn.Parameter(self.weight)

        if self.use_bias:
            bias_shape = (self.aligned_token_num, 1, self.units)
            self.bias = nn.Parameter(torch.empty(bias_shape))
            self.bias_initializer(self.bias)
        else:
            self.bias = None 

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # inputs 形状: (B_pad, S, D_in)，其中 B 是 self.batch
        # self.weight 形状: (B_pad, D_in_pad, D_out)，其中 D_out 是 self.out_features
 
        B1, M1, K1 = input.shape
        assert B1 == self.token_num, f"Batch size mismatch: {B1} != {self.token_num}"
        assert _aligned_dim(K1, 16)  == self.aligend_token_dim , f"Input size mismatch: {K1} != {K2}"

        if torch.is_autocast_enabled():
            # For now, hardcode to GPU's autocast dtype
            # if we need CPU support in the future, we can add it
            autocast_dtype = torch.get_autocast_gpu_dtype()
            input = input.to(autocast_dtype)
        
        if K1 % 16 != 0 or M1 % 16 != 0:
            input = torch.nn.functional.pad(input, (0, _aligned_dim(K1, 16) - K1, 0, _aligned_dim(M1, 16) - M1))
        # print("input shape:", input.shape)
        # print("weight shape:", self.weight.shape)

        weight_scale = _get_weight_scale(
            self.weight, self.config.cast_config_weight.scaling_type, self.config
        )

        if self.config.force_recompute_fp8_weight_in_bwd:
            weight_fp8_t = checkpoint.checkpoint(
                _cast_weight_to_float8_t,
                self.weight,
                self.config,
                self.linear_mm_config,
                weight_scale,
            )
        else:
            weight_fp8_t = _cast_weight_to_float8_t(
                self.weight,
                self.config,
                self.linear_mm_config,
                weight_scale,
            )

        weight_maybe_fp8_t = weight_fp8_t

        input_maybe_fp8 = hp_tensor_to_float8_dynamic(
                input,
                self.config.cast_config_input.target_dtype,
                self.linear_mm_config,
                gemm_input_role=GemmInputRole.INPUT,
                scaling_granularity=self.config.cast_config_input.scaling_granularity,
                axiswise_dim=get_maybe_axiswise_dim(
                    -1, self.config.cast_config_input.scaling_granularity
                ),
                round_scales_to_power_of_2=self.config.round_scales_to_power_of_2,
            )

        output = torch.empty((self.token_num, _aligned_dim(M1, 16), self.aligned_units), dtype=weight_maybe_fp8_t._orig_dtype, device=weight_maybe_fp8_t.device)
        for i in range(self.token_num):
            # Get the i-th slice for input and weight
            current_input_slice = input_maybe_fp8[i]
            current_weight_slice = weight_maybe_fp8_t[i]
            # Perform the 2D matrix multiplication
            output[i] = matmul_with_hp_or_float8_args.apply(
                current_input_slice,
                current_weight_slice,
                self.linear_mm_config, 
                self.config,          
            )

        output = output[..., :M1, :self.units]
        if self.bias is not None: # 检查 self.bias 是否存在
            # self.bias 形状: (B, 1, D_out)
            output = output + self.bias[:self.token_num].to(output.dtype)

        if self.activation is not None:
            output = self.activation(output)
            
        return output

    def extra_repr(self) -> str:
        # 自定义模块的字符串表示，以反映其真实参数和批处理维度
        return (f'batch={self.token_num}, aligned_batch={self.aligned_token_num}, in_features={self.token_dim}, in_aligned_features={self.aligend_token_dim}, '
                f'out_features={self.units}, out_aligned_features={self.aligned_units}, bias={self.bias is not None}')
