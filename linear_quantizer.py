from torch import Tensor, dtype, finfo, iinfo, zeros
from torch import round as t_round
from torch import float32, int32
from torch import int as t_int

from typing import Callable, List, Tuple, Union
from enum import Enum


class Mode(Enum):
    SymmetricMode = True
    AsymmetricMode = False


class Granularity(Enum):
    PerTensor = 0
    PerChannel = 1
    PerGroup = 2


"""
    r_max = s(q_max - z)
    r_min = s(q_min - z)
    - 
    r_max - r_min = s(q_max - q_min)    
    s = (r_max - r_min) / (q_max - q_min)
    z = r/s - q
"""

FloatOrTensor = Union[float, Tensor]
IntOrTensor = Union[int, Tensor]


class QuantizationParameters:

    def __init__(
        self,
        scale: FloatOrTensor,
        zero_point: IntOrTensor,
        mode: Mode,
        granularity: Granularity,
        group_size: Union[int, None] = None,
        dim: Union[int, None] = None,  # dim : [0, len(tensor.shape)]
    ):
        self.scale = scale
        self.zero_point = zero_point
        self.mode = mode
        self.granularity = granularity
        self.group_size = group_size
        self.dim = dim


# dim -> per rows : 0, per columns : 1
def per_channel_scale_and_zero_for(
    tensor: Tensor, data_type: dtype, mode: Mode, dim: Union[int, None] = None
) -> Tuple[Tensor, Tensor]:
    shape = tensor.shape
    if None == dim or dim >= len(shape) or dim < 0:
        raise Exception(
            f"dimension for channel is out of bounds, not in 0 < {dim} <= {len(shape)}"
        )
    zeroes = zeros(shape[dim]).to(int32)
    scales = zeros(shape[dim]).to(float32)
    for i in range(shape[dim]):
        sz: Tuple[float, int] = get_scale_and_zero_for(
            tensor.select(dim, i), data_type, mode
        )
        scales[i] = sz[0]
        zeroes[i] = sz[1]

    return scales, zeroes


# returns the grouped tensor and the ungroup function
def group(tensor: Tensor, by: int) -> Tuple[Tensor, Callable]:
    shape = tensor.shape
    if len(shape) != 2:
        raise Exception(
            f"tensors with more than 2 dimensions are not supported {shape}"
        )

    if (shape[0] * shape[1]) % by != 0:
        raise Exception(f"tensor {shape} is not groupabled by {by}")
    ungroup = lambda t: t.view(shape)

    return tensor.view(-1, by), ungroup


# finds and returns s and z for each group in the tensor. The compression rate depends on the group size and the mode
# in symmetric mode, for float8 compression with float16 scale tensors and group size 32,
# the compression rate has additional memory load of 16/32 = 0.5 coming from scales for each group.
def get_info(data_type):
    return finfo(data_type) if data_type.is_floating_point else iinfo(data_type)


# r = s(q-z), finds and returns s and z
def get_scale_and_zero_for(
    tensor: Tensor, data_type: dtype, mode: Mode
) -> Tuple[float, int]:
    dtype_info = get_info(data_type)
    q_max = dtype_info.max
    r_max = (
        tensor.abs().max().item()
        if Mode.SymmetricMode == mode.value
        else tensor.max().item()
    )
    zero = 0
    if Mode.AsymmetricMode == mode.value:
        r_max -= tensor.min().item()
        q_max -= dtype_info.min

    scale = r_max / q_max

    if Mode.AsymmetricMode == mode.value:
        zero = int(round(r_max / scale - q_max))
        zero = clamp(zero, dtype_info.min, dtype_info.max)

    return scale, zero


def clamp(val, _min, _max):
    return min(max(val, _min), _max)


quantization_strategies: List[Callable] = [
    get_scale_and_zero_for,
    per_channel_scale_and_zero_for,
    per_channel_scale_and_zero_for,
]


def quantize_linear(
    tensor: Tensor,
    data_type: dtype,
    granularity: Granularity,
    mode: Mode,
    **additional_parameters,  # dim : [0, len(tensor.shape)]
) -> Tuple[Tensor, QuantizationParameters]:
    get_scale_and_zero: Callable[..., Tuple[FloatOrTensor, IntOrTensor]] = (
        quantization_strategies[granularity.value]
    )

    ungroup = None
    dim, group_size = None, None

    if Granularity.PerTensor == granularity:
        (s, z) = get_scale_and_zero(tensor, data_type, mode)
    else:
        if not "dim" in additional_parameters:
            raise Exception(
                "dimension is required for per channel or per group quantizations"
            )

        if (
            Granularity.PerGroup == granularity
            and "group_size" in additional_parameters
        ):
            group_size = additional_parameters["group_size"]
            (tensor, ungroup) = group(tensor, group_size)

        dim = additional_parameters["dim"]
        (s, z) = get_scale_and_zero(tensor, data_type, mode, dim)

        scale_shape = [1] * tensor.dim()
        scale_shape[dim] = -1
        s = s.view(scale_shape)

        z = z.view(scale_shape)

    q_params = QuantizationParameters(s, z, mode, granularity, group_size, dim)

    # r/s - z = q
    rounded_tensor = t_round(tensor / s - z)

    d_info = get_info(data_type)
    d_min, d_max = d_info.min, d_info.max

    rounded_tensor = rounded_tensor.clamp(d_min, d_max).to(data_type)
    if ungroup:
        rounded_tensor = ungroup(rounded_tensor)

    return (rounded_tensor, q_params)


def dequantize_linear(tensor: Tensor, parameters: QuantizationParameters) -> Tensor:
    if Granularity.PerGroup == parameters.granularity:
        (grouped, ungroup) = group(tensor.float(), parameters.group_size)
        return ungroup(parameters.scale * (grouped - parameters.zero_point))
    return parameters.scale * (tensor.float() - parameters.zero_point)
