from unittest import main, TestCase

from linear_quantizer import *
from linear_layer import *

import torch
import sys

from functools import partial
from itertools import product

DEBUG = False


def quantization_error(tensor, dequantized_tensor):
    return (dequantized_tensor - tensor).abs().square().mean()


def debugger(*args, name: str = ""):
    if not DEBUG:
        return
    print(f"    [{name}]---", end="")
    for a in args:
        print(f"{str(a)}", end="")
    print()


def get_test_tensor() -> torch.Tensor:
    return torch.tensor(
        [[191.6, -13.5, 728.6], [92.14, 295.5, -184], [0, 684.6, 245.5]]
    )


class TestLinearLayer(TestCase):
    def test_per_channel_row_and_column(self):
        test_tensor = get_test_tensor()
        quantized_col_and_row = []
        exp_errors = [2.5091912746429443, 1.8084441423416138]
        frame = sys._getframe()
        name = frame.f_code.co_name
        _debugger = partial(debugger, name=name)

        dims = [0, 1]
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

            self.assertTrue(
                err <= exp_errors[dim], f"got:{err}, exp: {exp_errors[dim]}"
            )

    def test_per_tensor(self):
        test_tensor = get_test_tensor()
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

        exp = 2.5092
        self.assertTrue(err_per_tensor <= exp, f"got:{err_per_tensor}, exp: {exp}")

    def test_per_group(self):
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

        _debugger(f"quantized tensor:\n {q}")

        deq = dequantize_linear(q, params)
        _debugger(f"dequantized tensor:\n {deq}")

        exp_err = 1.9472
        err_per_group = quantization_error(test_tensor, deq)
        _debugger(
            f"quantization error for per group:\n : {err_per_group}",
            f"exp_err: {exp_err}, err_per_group: {err_per_group}",
            f"quantization error for per group:\n {err_per_group}",
            f"exp_err >= err_per_group: {exp_err >= err_per_group}",
        )

        self.assertTrue(
            exp_err >= err_per_group, f"got:{err_per_group}, exp: {exp_err}"
        )

    def test_group_ungroup(self):
        test_tensor = get_test_tensor()
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

        self.assertTrue(
            ungrouped.shape == shape, f"got:{ungrouped.shape}, exp: {shape}"
        )

    def test_linear_forward(self):
        batch = 2
        for i_shape, o_shape in [(4, 6), (16, 32)]:
            linear_layer = QWQALinearLayer(i_shape, o_shape)
            hidden_state = randn(batch, i_shape, i_shape)
            output = linear_layer(hidden_state)
            self.assertTrue(
                output.dtype == float32 and output.shape[-1] == o_shape,
                f"got:{output.shape}, exp: {o_shape}",
            )

    def test_linear_quantize(self):
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
        self.assertTrue(
            is_scale_shape_ok, f"got: {linear_layer.scales.shape[-1]}, exp: {i_shape}"
        )
        r = linear_layer.int8_weights * linear_layer.scales.unsqueeze(1)
        quantization_error = (original_weights - r).abs().mean()
        exp = 0.01
        is_err_ok = quantization_error < exp
        self.assertTrue(is_err_ok, f"got: {quantization_error}, exp: {exp}")
        _debugger(
            f"multiplied with squeezed scale : {r}, quantization error: {quantization_error}"
        )

    def test_replace_linear_layers(self):
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
        )
        from accelerate import init_empty_weights

        model = "Salesforce/codegen-350M-mono"
        config = AutoConfig.from_pretrained(model)
        causal_lm = None
        with init_empty_weights():
            causal_lm = AutoModelForCausalLM.from_config(config)

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

        test_models = [causal_lm, DummyModel()]
        for model, case, q in product(test_models, ignore_cases, quantize):
            replace_linear_layers_with_w8a16(
                model, Target.Linear(QWQALinearLayer), case, q
            )
            _debugger(model)
            children = model.named_children()

            self.assertTrue(
                all(
                    [
                        ch.__class__.__name__ != "W8A16LinearLayer"
                        for ch in children
                        if ch[0] in case
                    ]
                ),
            )
            class_name = "Linear" if q else "W8A16LinearLayer"
            self.assertTrue(
                all(
                    [
                        ch.__class__.__name__ == class_name
                        for ch in children
                        if ch[0] not in case
                    ]
                )
            )


if __name__ == "__main__":
    main()
