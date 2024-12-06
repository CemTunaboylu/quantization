from linear_quantizer import *
from linear_layer import *

import torch
import sys

from functools import partial
from itertools import product

DEBUG = False


def debugger(*args, name: str = ""):
    if not DEBUG:
        return
    print(f"    [{name}]---", end="")
    for a in args:
        print(f"{str(a)}", end="")
    print()


def per_channel_row_and_column(test_tensor) -> bool:

    quantized_col_and_row = []
    exp_errors = [2.5091912746429443, 1.8084441423416138]

    frame = sys._getframe()
    name = frame.f_code.co_name
    _debugger = partial(debugger, name=name)

    dims = [0, 1]
    results = [False] * len(dims)
    for dim in dims:
        (q, params) = quantize_linear(
            test_tensor,
            torch.int8,
            Granularity.PerDimension(dim),
            Mode.Symmetric,
        )
        quantized_col_and_row.append(q)
        chan = ["row", "col"][dim]

        _debugger(f"quantized for channel {chan}:{'\n'}", q)

        deq = dequantize_linear(quantized_col_and_row[dim], params)
        _debugger(f"dequantized for channel {chan} :{'\n'}", deq)

        err = quantization_error(test_tensor, deq)

        _debugger(f"quantization error for channel {chan}:{'\n'}", err)

        results[dim] = err <= exp_errors[dim]
    return all(results)


def per_tensor(test_tensor) -> bool:
    (q, params) = quantize_linear(
        test_tensor, torch.int8, Granularity.PerTensor, Mode.Symmetric
    )
    frame = sys._getframe()
    name = frame.f_code.co_name
    _debugger = partial(debugger, name=name)

    _debugger("quantized tensor:\n", q)

    deq = dequantize_linear(q, params)
    _debugger("dequantized tensor:\n", deq)

    err_per_tensor = quantization_error(test_tensor, deq)
    _debugger("quantization error for per tensor:\n", err_per_tensor)

    return err_per_tensor <= 2.5092


def per_group():
    test_tensor = torch.tensor(
        [
            [0.4963, 0.7682, 0.0885, 0.1320, 0.3074, 0.6341],
            [0.4901, 0.8964, 0.4556, 0.6323, 0.3489, 0.4017],
            [0.0223, 0.1689, 0.2939, 0.5185, 0.6977, 0.8000],
            [0.1610, 0.2823, 0.6816, 0.9152, 0.3971, 0.8742],
            [0.4194, 0.5529, 0.9527, 0.0362, 0.1852, 0.3734],
            [0.3051, 0.9320, 0.1759, 0.2698, 0.1507, 0.0317],
        ]
    )
    dim, group_size = 0, 3
    (q, params) = quantize_linear(
        test_tensor,
        torch.int8,
        Granularity.PerGroup(group_size, dim),
        Mode.Symmetric,
    )

    frame = sys._getframe()
    name = frame.f_code.co_name
    _debugger = partial(debugger, name=name)

    _debugger("[%s]quantized tensor:\n" % name, q)

    deq = dequantize_linear(q, params)
    _debugger("[%s]dequantized tensor:\n" % name, deq)

    exp_err = 1.9472
    err_per_group = quantization_error(test_tensor, deq)
    _debugger(
        f"quantization error for per group:\n : {err_per_group}",
        f"exp_err: {exp_err}, err_per_group: {err_per_group}",
        f"quantization error for per group:\n {err_per_group}",
        f"exp_err >= err_per_group: {exp_err >= err_per_group}",
    )

    return exp_err >= err_per_group


def group_ungroup(test_tensor) -> bool:
    shape = test_tensor.shape
    by = shape[0] * shape[1]

    grouped_tensor, ungroup_f = group(test_tensor, by)

    frame = sys._getframe()
    name = frame.f_code.co_name
    _debugger = partial(debugger, name=name)

    ungrouped = ungroup_f(grouped_tensor)
    _debugger(
        f"grouped_tensor:{grouped_tensor}, with shape: {grouped_tensor.shape}",
        f"unggrouped: {ungrouped}",
    )

    return ungrouped.shape == shape


def quantization_error(tensor, dequantized_tensor):
    return (dequantized_tensor - tensor).abs().square().mean()


def linear_forward() -> bool:
    # i_shape, o_shape = 16, 32
    batch, i_shape, o_shape = 2, 4, 6
    linear_layer = QWQALinearLayer(i_shape, o_shape)
    hidden_state = randn(batch, i_shape, i_shape)
    output = linear_layer(hidden_state)
    return output.dtype == float32 and output.shape[-1] == o_shape


def linear_quantize() -> bool:
    i_shape, o_shape = 4, 8
    linear_layer = QWQALinearLayer(i_shape, o_shape)

    original_weights = randn((i_shape, o_shape), dtype=bfloat16)

    linear_layer.quantize(original_weights)

    frame = sys._getframe()
    name = frame.f_code.co_name
    _debugger = partial(debugger, name=name)

    _debugger(
        f"random weights: {linear_layer.int8_weights.shape} \n{linear_layer.int8_weights}",
        f"original weights: {original_weights.shape} \n{original_weights}",
        f"weights after quantization: {linear_layer.int8_weights.shape} \n{linear_layer.int8_weights}",
        f"scales after quantization: {linear_layer.scales}",
    )

    is_scale_shape_ok = linear_layer.scales.shape[-1] == i_shape
    r = linear_layer.int8_weights * linear_layer.scales.unsqueeze(1)
    quantization_error = (original_weights - r).abs().mean()
    is_err_ok = quantization_error < 0.01
    _debugger(
        f"multiplied with squeezed scale : {r}, quantization error: {quantization_error}"
    )

    return is_scale_shape_ok and is_err_ok


def replace_linear_layers() -> bool:
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(1, 1)
            self.linear_1 = nn.Linear(1, 1)
            self.linear_2 = nn.Linear(1, 1, bias=False)
            self.lm_head = nn.Linear(1, 1, bias=False)

    ignore_cases = [[], ["lm_head"]]
    quantize = [False, True]

    frame = sys._getframe()
    name = frame.f_code.co_name
    _debugger = partial(debugger, name=name)

    expected_class_names_match = [False] * (len(ignore_cases) * len(quantize))

    for i, (case, q) in enumerate(product(ignore_cases, quantize)):
        model = DummyModel()
        replace_linear_layers_with_w8a16(model, Target.Linear(QWQALinearLayer), case, q)
        _debugger(model)
        children = model.named_children()

        expected_class_names_match[i] = all(
            [
                ch.__class__.__name__ != "W8A16LinearLayer"
                for ch in children
                if ch[0] in case
            ]
        )
        class_name = "Linear" if q else "W8A16LinearLayer"
        expected_class_names_match[i] = expected_class_names_match[i] and all(
            [
                ch.__class__.__name__ == class_name
                for ch in children
                if ch[0] not in case
            ]
        )
        _debugger(f"case:{case}, q:{q}", expected_class_names_match[i])

    return all(expected_class_names_match)


if __name__ == "__main__":
    from sys import argv

    DEBUG = "debug" in argv

    test_tensor = torch.tensor(
        [[191.6, -13.5, 728.6], [92.14, 295.5, -184], [0, 684.6, 245.5]]
    )

    _debugger = partial(debugger, name="main")

    _debugger("test tensor:", test_tensor)
    icons = ["❌", "✅"]

    for f in [per_tensor, per_channel_row_and_column, group_ungroup]:
        result = f(test_tensor)
        print(
            f"- {f.__name__}  {icons[int(result)]}",
        )

    for f in [per_group, linear_forward, linear_quantize, replace_linear_layers]:
        result = f()
        print(
            f"- {f.__name__}  {icons[int(result)]}",
        )

    # TODO: use other models as well, and move these into test_ file.
    # from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    # model = "Salesforce/codegen-350M-mono"
