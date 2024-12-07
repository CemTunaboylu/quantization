from functools import partial
from torch import bfloat16, nn, dtype
from torch import int8, float32, Tensor, iinfo
from torch import randn, randint
from torch import round as t_round

from torch.functional import F

from typing import Type, Protocol

from rust_enum import enum, Case


def quantized_forward(weights, input: Tensor, scales, bias=None):
    casted_weights = weights.to(dtype=input.dtype)
    output = F.linear(input, casted_weights) * scales
    if bias is not None:
        output += bias
    return output


class LinearLayerKeywordedConstructor(Protocol):
    def __init__(
        self, *, in_features: int, out_features: int, bias: bool, dtype
    ) -> None:
        pass


@enum
class Target:
    # could have just used Case(nn.Linear) but this is more fun (is more dynamic)
    Linear = Case(kwarged=LinearLayerKeywordedConstructor)


def _linear(child: nn.Linear, target_class: Type):
    return target_class(
        in_features=child.in_features,
        out_features=child.out_features,
        bias=child.bias is not None,
        dtype=child.weight.dtype,
    )


def target_class_from(child: nn.Module, target_class: Target) -> nn.Module:
    target_class_constructor: LinearLayerKeywordedConstructor
    new_module: nn.Module
    match target_class:
        case Target.Linear(target_class_constructor):
            new_module = _linear(child, target_class_constructor)
        case _:
            raise NotImplemented(f"{target_class}")

    return new_module


def name_for_quantized_weight_buffer(dtype: dtype) -> str:
    r = dtype.__repr__()
    i = r.rfind(".")
    return f"{r[i + 1 :]}_weights"


class QWQALinearLayer(nn.Module, LinearLayerKeywordedConstructor):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype=float32,
    ):
        super().__init__()
        """
            note: storing weights as a parameter will fail since pytorch will expect it to be able to
            be calculate gradients on it, but currently pytorch cannot calculate gradients on int8 tensors.

            self.int8_weights = nn.Parameter(Tensor([0, 1]).to(dtype=int8))

            Thus, just trying to create an this linear layer with a dtype of int8 will not work.
                w = W8A16LinearLayer(1,1)
            
              RuntimeError: Only Tensors of floating point and complex dtype can require gradients
        """

        self.data_type = int8
        _info = iinfo(self.data_type)
        self.min, self.max = _info.min, _info.max
        self.q_w_name = name_for_quantized_weight_buffer(self.data_type)

        self.register_buffer(
            self.q_w_name,
            _r := randint(
                low=self.min,
                high=self.max,
                size=(out_features, in_features),
                dtype=self.data_type,
            ),
        )

        self.s = "scales"
        self.register_buffer(self.s, randn(out_features, dtype=dtype))

        if bias:
            self.b = "bias"
            self.register_buffer(self.b, randn((1, out_features), dtype=dtype))
        else:
            self.bias = None

    def forward(self, input: Tensor):
        return quantized_forward(
            self._buffers[self.q_w_name],
            input,
            self._buffers[self.s],
            self.bias,
        )

    def quantize(self, weights: Tensor):
        # for stability cast the weights into f32 first
        weights_f32 = weights.clone().to(float32)

        # get abs max of the 32 elements in each 6 row, put them into int8 range i.e. symmetric quantization
        scales = weights_f32.abs().max(dim=-1).values / self.max
        self._buffers[self.s] = scales.to(weights.dtype)

        self._buffers[self.q_w_name] = t_round(weights_f32 / scales.unsqueeze(1)).to(
            self.data_type
        )


def replace_linear_layers_with_w8a16(
    module: nn.Module, target_class: Target, module_names_to_exclude, quantize=True
):
    _target_class_from = partial(target_class_from, target_class=target_class)
    for name, child in module.named_children():
        if not isinstance(child, nn.Linear) or name in module_names_to_exclude:
            replace_linear_layers_with_w8a16(
                child, target_class, module_names_to_exclude, quantize
            )
            continue

        old_bias = child.bias
        if quantize:
            old_weights = child.weight

        new_module = _target_class_from(child)
        setattr(module, name, new_module)

        if quantize:
            new_module.quantize(old_weights)

        if old_bias is not None:
            getattr(module, name).bias = old_bias


if __name__ == "__main__":
    dt = int8
    assert "int8" == name_for_quantized_weight_buffer(dt)
