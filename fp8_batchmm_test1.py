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


class Float8BMM(nn.Module): # 继承自 nn.Linear
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
        **kwargs
    ):
        super(Float8BMM, self).__init__()

        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        self.no_clip = kwargs.get("no_clip", False)
        self.token_num = batch
        self.token_dim = in_features
        
        self.fp8_all_gather = fp8_all_gather
        self.force_recompute = fp8_force_recompute
        config = Float8LinearConfig(
                enable_fsdp_float8_all_gather=self.fp8_all_gather,
                force_recompute_fp8_weight_in_bwd=self.force_recompute,
        )

        self.scaling_type_input = config.cast_config_input.scaling_type
        self.scaling_type_weight = config.cast_config_weight.scaling_type
        self.scaling_type_grad_output = config.cast_config_grad_output.scaling_type
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

        kernel_shape = (batch, self.token_dim, self.units)

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


        if self.use_bias:
            bias_shape = (batch, 1, self.units)
            self.bias = nn.Parameter(torch.empty(bias_shape))
            self.bias_initializer(self.bias)
        else:
            self.bias = None 

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # inputs 形状: (B, S, D_in)，其中 B 是 self.batch
        # self.weight 形状: (B, D_in, D_out)，其中 D_out 是 self.out_features
 
        B1, M, K1 = input.shape
        B2, K2, N = self.weight.shape
        assert B1 == B2, f"Batch size mismatch: {B1} != {B2}"
        assert K1 == K2, f"Input size mismatch: {K1} != {K2}"

        if torch.is_autocast_enabled():
            # For now, hardcode to GPU's autocast dtype
            # if we need CPU support in the future, we can add it
            autocast_dtype = torch.get_autocast_gpu_dtype()
            input = input.to(autocast_dtype)
        
        # print("input shape:", input.shape)
        # print("weight shape:", self.weight.shape)


        weight_scale = _get_weight_scale(
            self.weight, self.scaling_type_weight, self.config
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


        # print("weight_maybe_fp8_t shape:", weight_maybe_fp8_t.shape)

        output = torch.empty((B1, M, N), dtype=weight_maybe_fp8_t._orig_dtype, device=weight_maybe_fp8_t.device)
        for i in range(input_maybe_fp8.shape[0]):
            # Get the i-th slice for input and weight
            current_input_slice = input_maybe_fp8[i]
            current_weight_slice = weight_maybe_fp8_t[i]

            # print("input type and weight type:", type(current_input_slice), type(current_weight_slice))
            # Perform the 2D matrix multiplication
            output[i] = matmul_with_hp_or_float8_args.apply(
                current_input_slice,
                current_weight_slice,
                self.linear_mm_config, 
                self.config,          
            )

        if self.bias is not None: # 检查 self.bias 是否存在
            # self.bias 形状: (B, 1, D_out)
            output = output + self.bias.to(output.dtype)

        if self.activation is not None:
            output = self.activation(output)
            
        return output

    def extra_repr(self) -> str:
        # 自定义模块的字符串表示，以反映其真实参数和批处理维度
        return (f'batch={self.batch}, in_features={self.in_features}, '
                f'out_features={self.out_features}, bias={self.bias is not None}')
