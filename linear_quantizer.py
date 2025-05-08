from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Callable, List, Tuple, Union

from torch import Tensor, dtype, finfo, iinfo, zeros
from torch import round as t_round
from torch import clamp as t_clamp
from torch import float32, int32

from rust_enum import enum, Case

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


PER_COLUMNS = 0
PER_ROWS = 1


def per_dim_scale_and_zero_for(
    tensor: Tensor, data_type: dtype, mode: Mode, reduce_dim: int = PER_ROWS
) -> Tuple[Tensor, Tensor]:
    shape = tensor.shape
    if reduce_dim >= len(shape) or reduce_dim < 0:
        raise Exception(
            f"dimension is out of bounds, not in 0 < {reduce_dim} < {len(shape)}"
        )
    sz: Tuple[Tensor, Tensor] = get_scale_and_zero_for(
        tensor, data_type, mode, reduce_dim=reduce_dim
    )
    return sz


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


def get_info(data_type):
    return finfo(data_type) if data_type.is_floating_point else iinfo(data_type)


# r = s(q-z), finds and returns s and z
def get_scale_and_zero_for(
    tensor: Tensor,
    data_type: dtype,
    mode: Mode,
    reduce_dim: None | int = None,
) -> Tuple[Tensor, Tensor]:
    dtype_info = get_info(data_type)
    r_max, q_max = None, dtype_info.max

    match mode:
        case Mode.Symmetric:
            if not reduce_dim:
                r_max = tensor.abs().max()
            else:
                r_max = tensor.abs().amax(dim=reduce_dim, keepdim=True)
        case Mode.Asymmetric:
            if not reduce_dim:
                r_max = tensor.max() - tensor.min()
            else:
                r_max = tensor.amax(dim=reduce_dim) - tensor.amin(dim=reduce_dim)
            q_max -= dtype_info.min

    # scale, zero = (r_max / q_max).to(float32), zeros([1] if not axis else r_max.shape)
    scale, zero = (r_max / q_max).to(float32), zeros(r_max.shape)

    if Mode.Asymmetric == mode.value:
        zero = t_round(r_max / scale - q_max).to(int32)
        zero = t_clamp(zero, dtype_info.min, dtype_info.max)

    return scale, zero


# The compression rate depends on the group size and the mode in symmetric mode,
# for float8 compression with float16 scale tensors and group size 32,
# the compression rate has additional memory load of 16/32 = 0.5 coming from scales for each group.
def quantize_linear(
    tensor: Tensor,
    data_type: dtype,
    granularity: Granularity,
    mode: Mode = Mode.Symmetric,
) -> Tuple[Tensor, QuantizationParameters]:
    q_params = partial(
        QuantizationParameters,
        mode=mode,
        granularity=granularity,
    )
    func_scale_and_zero_for: Callable
    ungroup, dim, group_by = None, None, None
    match granularity:
        case Granularity.PerTensor:
            func_scale_and_zero_for = get_scale_and_zero_for
        case Granularity.PerDimension(dim):
            func_scale_and_zero_for = per_dim_scale_and_zero_for
        case Granularity.PerGroup(group_by, dim):
            func_scale_and_zero_for = per_dim_scale_and_zero_for
            tensor, ungroup = group(tensor, group_by)
        case _:
            raise Exception(f"Granularity {granularity} is not implemented")

    q_params = partial(q_params, dim=dim, group_by=group_by)
    (s, z) = func_scale_and_zero_for(tensor, data_type, mode, reduce_dim=dim)

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
