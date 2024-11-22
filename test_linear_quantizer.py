from unittest import TestCase, main
import torch

from linear_quantizer import (
    PER_COLUMNS,
    PER_ROWS,
    Granularity,
    QuantizationParameters,
    get_scale_and_zero_for,
    per_dim_scale_and_zero_for,
    Mode,
    group,
    quantize_linear,
)

from itertools import product


def is_close(a, b, rel_tol=1e-04, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def are_of_types(to_check, types):
    return any([isinstance(to_check, t) for t in types])


def are_tensors_close(t1, t2):
    if are_of_types(t1, [float, int]) and are_of_types(t2, [float, int]):
        return is_close(t1, t2)
    if isinstance(t1, torch.Tensor):
        t1 = t1.view(-1, 1)
    if isinstance(t2, torch.Tensor):
        t2 = t2.view(-1, 1)
    return all([is_close(e1, e2) for e1, e2 in zip(t1, t2)])


def is_tensor_with_same_element(t1, val):
    if isinstance(t1, torch.Tensor):
        t1 = t1.view(-1, 1)
        return all([is_close(e1, val) for e1 in t1])
    return is_close(t1, val)


def is_zero_tensor(t1):
    return is_tensor_with_same_element(t1, 0.0)


class TestScale(TestCase):
    def test_per_dim_scale_and_zero_for(self):
        types = [torch.float16, torch.float32]
        max_vals = [torch.finfo(t).max for t in types]

        ratios = {
            (0, torch.float16): [1.0, 5.1948e33],
            (0, torch.float32): [1.9250e-34, 1.0],
            (1, torch.float16): [5.1948e33] * 3,
            (1, torch.float32): [1.0] * 3,
        }

        elms = [[elm] * 3 for elm in max_vals]
        t = torch.tensor(elms, dtype=torch.float32)
        for ty, dim in product(types, [PER_ROWS, PER_COLUMNS]):
            s, z = per_dim_scale_and_zero_for(t, ty, Mode.Symmetric, dim=dim)
            exp = torch.tensor(ratios[(dim, ty)], dtype=torch.float32)
            self.assertTrue(
                all([is_close(ss, ee) for ss, ee in zip(s, exp)]),
                f"exp: {exp}, got: {s}",
            )
            self.assertTrue(
                torch.equal(
                    z,
                    exp := torch.zeros_like(z),
                ),
                f"exp: {exp}, got: {z}",
            )

    def test_get_scale_for(self):
        types = [torch.float16, torch.float32]
        max_vals = [torch.finfo(t).max for t in types]
        tensors = [torch.tensor([1, e]) for e in max_vals]

        types.append(types[0])
        tensors.append(torch.tensor([torch.finfo(types[-1]).max / 2]))

        exp_s = [1.0, 1.0, 0.5]

        for ix, (ty, tn) in enumerate(zip(types, tensors)):
            # symmetric mode test
            s = get_scale_and_zero_for(tn, ty, Mode.Symmetric)
            self.assertEqual(s[0], exp_s[ix])
            self.assertEqual(s[1], 0)

    def test_group(self):
        test_cases = [
            (24, (6, 4), 2),
            (24, (3, 8), 3),
            (24, (2, 12), 4),
            (24, (1, 24), 6),
        ]
        for r, (h, w), by in test_cases:
            tensor = torch.arange(r).reshape(h, w)
            grouped_tensor, ungroup = group(tensor, by=by)
            h_exp = r // by
            assert grouped_tensor.shape == (
                h_exp,
                by,
            ), f"Expected shape ({h_exp}, {by}), got {grouped_tensor.shape}"

            ungrouped_tensor = ungroup(grouped_tensor)
            assert torch.equal(
                ungrouped_tensor, tensor
            ), "Ungrouped tensor does not match the original"

        r, (h, w), by = (24, (6, 4), 5)
        tensor = torch.arange(r).reshape(h, w)
        self.assertRaises(Exception, group, tensor=tensor, by=by)
        tensor_3d = torch.arange(r).reshape(h, w, 1)
        self.assertRaises(Exception, group, tensor=tensor_3d, by=by)

    def test_quantize_linear(self):
        dtypes = [torch.float16, torch.float32]
        max_vals = [torch.finfo(t).max for t in dtypes]
        # modes = [Mode.Symmetric, Mode.Asymmetric]
        modes = [Mode.Symmetric]

        group_by, rows = 4, 8
        elms = [[elm] * rows for elm in max_vals]
        t = torch.tensor(elms, dtype=torch.float32)
        test_cases = [
            Granularity.PerTensor,
            Granularity.PerDimension(PER_ROWS),
            Granularity.PerDimension(PER_COLUMNS),
            Granularity.PerGroup(group_by, PER_ROWS),
            Granularity.PerGroup(group_by, PER_COLUMNS),
        ]
        rowed = lambda elms: [[elm] * rows for elm in elms]
        exps = [  # quantized_tensor, scale
            (rowed([0.0, 65504.0]), 5.194833088643882e33),
            (rowed([6.5504e04, 3.4028e38]), 1.0),
            (rowed([65504.0] * 2), [[1.0000e00], [5.1948e33]]),
            (rowed([3.4028e38] * 2), [[1.9250e-34], [1.0000e00]]),
            (rowed([0.0, 65504.0]), [5.194833088643882e33] * rows),
            (rowed([6.5504e04, 3.4028e38]), [1.0] * rows),
            (
                rowed([6.5504e04] * 2),
                [[1.0000e00], [1.0000e00], [5.1948e33], [5.1948e33]],
            ),
            (
                rowed([3.4028e38] * 2),
                [[1.9250e-34], [1.9250e-34], [1.0000e00], [1.0000e00]],
            ),
            (rowed([0.0, 65504.0]), [5.1948e33] * group_by),
            (
                rowed([65504.0, 3.4028e38]),
                [1.0] * group_by,
            ),
        ]
        for ix, (g, dt, m) in enumerate(product(test_cases, dtypes, modes)):
            qt, qp = quantize_linear(tensor=t, data_type=dt, granularity=g, mode=m)
            exp_t, s = exps[ix]
            group_by, dim = None, None
            match g:
                case Granularity.PerDimension(dim):
                    dim = dim
                case Granularity.PerGroup(by, dim):
                    group_by = by
                    dim = dim
            self.assertTrue(
                are_tensors_close(
                    qt,
                    exp := torch.tensor(exp_t),
                ),
                f"exp: {exp}, got: {qt}",
            )
            f = None
            if isinstance(qp.scale, torch.Tensor):
                f = are_tensors_close
                s = torch.tensor(s)
            else:
                f = is_tensor_with_same_element
            self.assertTrue(f(qp.scale, s), f"exp: {s}, got: {qp}")
            self.assertTrue(
                is_zero_tensor(qp.zero_point),
                f"expected zero tensor, got: {qp.zero_point}",
            )
            exp_q = QuantizationParameters(
                None,
                None,
                mode=m,
                granularity=g,
                group_by=group_by,
                dim=dim,
            )
            qp.scale = None
            qp.zero_point = None
            self.assertEqual(
                qp,
                exp_q,
                f"exp: {exp_q}, got: {qp}",
            )


if __name__ == "__main__":
    main()
