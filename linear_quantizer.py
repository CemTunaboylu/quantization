from torch import FloatTensor, IntTensor, Tensor, dtype, finfo, ones, zeros, empty
from torch import round as t_round
from torch import float32, int32

from typing import Tuple, Union
from enum import Enum


class Mode(Enum):
    SymmetricMode = True
    AsymmetricMode = False


class Granularity(Enum):
    PerTensor = 0
    PerChannel = 1
    # PerGroup = 2


"""
    r_max = s(q_max - z)
    r_min = s(q_min - z)
    - 
    r_max - r_min = s(q_max - q_min)    
    s = (r_max - r_min) / (q_max - q_min)
    z = r/s - q
"""


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


# r = s(q-z), finds and returns s and z
def get_scale_and_zero_for(
    tensor: Tensor, data_type: dtype, mode: Mode
) -> Tuple[float, int]:
    dtype_info = finfo(data_type)
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


def quantize_linear(
    tensor: Tensor,
    data_type: dtype,
    mode: Mode,
) -> Tuple[Tensor, float, int]:
    (s, z) = get_scale_and_zero_for(tensor, data_type, mode)
    # r/s - z = q
    rounded_tensor = t_round(tensor / s - z)

    d_info = finfo(data_type)
    d_min, d_max = d_info.min, d_info.max

    return (rounded_tensor.clamp(d_min, d_max).to(data_type), s, z)


def dequantize_linear(
    tensor: Tensor,
    with_scale_and_zero: Tuple[float, int],
) -> Tensor:
    (s, z) = with_scale_and_zero
    return s * (tensor.float() - z)
