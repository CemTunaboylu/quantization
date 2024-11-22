from dataclasses import dataclass
from functools import partial

from torch import Tensor, dtype, finfo, iinfo, zeros
from torch import round as t_round
from torch import float32, int32
from torch import int as t_int

from typing import Callable, List, Tuple, Union

from rust_enum import enum, Case
from enum import Enum


class Mode(Enum):
    Symmetric = False
    Asymmetric = True


@enum
class Granularity:
    PerTensor = Case()
    PerDimension = Case(dim=int)
    PerGroup = Case(by=int, dim=int)


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


@dataclass
class QuantizationParameters:
    scale: FloatOrTensor
    zero_point: IntOrTensor
    mode: Mode
    granularity: Granularity
    group_by: Union[int, None] = None
    dim: Union[int, None] = None


PER_ROWS = 0
PER_COLUMNS = 1


def per_dim_scale_and_zero_for(
    tensor: Tensor, data_type: dtype, mode: Mode, dim: int = PER_ROWS
) -> Tuple[Tensor, Tensor]:
    shape = tensor.shape
    if dim >= len(shape) or dim < 0:
        raise Exception(f"dimension is out of bounds, not in 0 < {dim} < {len(shape)}")
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
        raise Exception(f"tensor {shape} is not groupable by {by}")
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
    r_max, q_max = None, dtype_info.max
    match mode:
        case Mode.Symmetric:
            r_max = tensor.abs().max().item()
        case Mode.Asymmetric:
            r_max = tensor.max().item() - tensor.min().item()
            q_max -= dtype_info.min

    scale = r_max / q_max

    zero = 0
    if Mode.Asymmetric == mode.value:
        zero = int(round(r_max / scale - q_max))
        zero = clamp(zero, dtype_info.min, dtype_info.max)

    return scale, zero


def clamp(val, _min, _max):
    return min(max(val, _min), _max)


def quantize_linear(
    tensor: Tensor,
    data_type: dtype,
    granularity: Granularity,
    mode: Mode,
) -> Tuple[Tensor, QuantizationParameters]:
    ungroup, dim, multi_dim = None, None, True
    q_params = partial(
        QuantizationParameters,
        mode=mode,
        granularity=granularity,
    )
    scale_and_zero_for: Callable
    match granularity:
        case Granularity.PerTensor:
            scale_and_zero_for = partial(get_scale_and_zero_for)
            multi_dim = False
        case Granularity.PerDimension(dim):
            scale_and_zero_for = partial(per_dim_scale_and_zero_for, dim=dim)
            q_params = partial(q_params, dim=dim)
        case Granularity.PerGroup(by, dim):
            scale_and_zero_for = partial(per_dim_scale_and_zero_for, dim=dim)
            tensor, ungroup = group(tensor, by)
            q_params = partial(q_params, dim=dim, group_by=by)
        case _:
            print(f"g: {granularity}")

    (s, z) = scale_and_zero_for(tensor, data_type, mode)
    if multi_dim:
        scale_shape = [1] * tensor.dim()
        scale_shape[dim] = -1
        s = s.view(scale_shape)
        z = z.view(scale_shape)

    # r/s - z = q
    rounded_tensor = t_round(tensor / s - z)

    d_info = get_info(data_type)
    d_min, d_max = d_info.min, d_info.max

    rounded_tensor = rounded_tensor.clamp(d_min, d_max).to(data_type)
    if ungroup:
        rounded_tensor = ungroup(rounded_tensor)

    return (rounded_tensor, q_params(s, z))


def dequantize_linear(tensor: Tensor, parameters: QuantizationParameters) -> Tensor:
    if Granularity.PerGroup == parameters.granularity:
        (grouped, ungroup) = group(tensor.float(), parameters.group_by)
        return ungroup(parameters.scale * (grouped - parameters.zero_point))
    return parameters.scale * (tensor.float() - parameters.zero_point)
