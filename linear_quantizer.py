from torch import Tensor, dtype, iinfo
from torch import round as t_round

from typing import Tuple, Union


# r = s(q-z), finds and returns s and z
def get_scale_and_zero_point_for_dtype(
    tensor: Tensor, data_type: dtype
) -> Tuple[float, int]:
    _dtype_info = iinfo(data_type)
    q_min, q_max = _dtype_info.min, _dtype_info.max
    r_min, r_max = tensor.min().item(), tensor.max().item()
    """
    r_max = s(q_max - z)
    r_min = s(q_min - z)
    - 
    r_max - r_min = s(q_max - q_min)    
    s = (r_max - r_min) / (q_max - q_min)
    """
    scale: float = (r_max - r_min) / (q_max - q_min)

    """
    z = r/s - q
    """
    zero_point: int = int(round(r_max / scale - q_max))
    zero_point = clamp(zero_point, q_min, q_max)

    return scale, zero_point


def clamp(val, _min, _max):
    return min(max(val, _min), _max)


def quantize_linear(
    tensor: Tensor,
    data_type: dtype,
    with_scale_and_zero: Union[Tuple[float, int], None],
) -> Tuple[Tensor, float, int]:
    if None == with_scale_and_zero:
        with_scale_and_zero = get_scale_and_zero_point_for_dtype(tensor, data_type)

    # r/s - z = q
    (s, z) = with_scale_and_zero

    rounded_tensor = t_round(tensor / s - z)

    _d_iinfo = iinfo(data_type)
    d_min, d_max = _d_iinfo.min, _d_iinfo.max

    return (rounded_tensor.clamp(d_min, d_max).to(data_type), s, z)


def dequantize_linear(
    tensor: Tensor,
    data_type: dtype,
    with_scale_and_zero: Tuple[float, int],
) -> Tensor:
    (s, z) = with_scale_and_zero
    return s * (tensor.float() - z)
