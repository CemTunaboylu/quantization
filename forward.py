from torch.functional import F
from torch import Tensor


def quantized_forward(weights, input: Tensor, scales, bias=None):
    casted_weights = weights.to(dtype=input.dtype)
    output = F.linear(input, casted_weights) * scales
    if bias is not None:
        output += bias
    return output
