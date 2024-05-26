from torch import Tensor, dtype, finfo
from torch import round as t_round

from typing import Tuple, Union
from enum import Enum


class Mode(Enum):
    SymmetricMode = True
    AsymmetricMode = False


# r = s(q-z), finds and returns s and z
"""
    r_max = s(q_max - z)
    r_min = s(q_min - z)
    - 
    r_max - r_min = s(q_max - q_min)    
    s = (r_max - r_min) / (q_max - q_min)
    z = r/s - q
"""
# TODO
# - finfo + iinfo
# - typehint float or tensor : FloatOrTensor = Union[float, Tensor]


# r = s(q-z), finds and returns s and z
def get_scale_and_zero_for(
    tensor: Tensor, data_type: dtype, mode: Mode, dim: Union[int, None] = None
) -> Tuple[float, int]:
    dtype_info = finfo(data_type)
    q_max = dtype_info.max
    r_max = (
        tensor.abs().max().item()
        if Mode.SymmetricMode == mode.value
        else tensor.max().item()
    )
    zero: int = 0
    if Mode.AsymmetricMode == mode.value:
        r_max -= tensor.min().item()
        q_max -= dtype_info.min

    scale: float = r_max / q_max
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
