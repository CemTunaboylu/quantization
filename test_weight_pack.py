from unittest import TestCase, main
from functools import partial
from itertools import product

from weight_pack import (
    bit_shifts,
    pack_weights,
    unpack_weights,
    get_dtype_size_in_bits,
)

from torch import (
    rand,
    seed,
    uint8,
    uint16,
    uint32,
    uint64,
    int8,
    int16,
    int32,
    int64,
    iinfo,
)
from torch import tensor


class TestWeightPack(TestCase):
    def test_2d_pack_unpack_2_bits(self):
        ts = [[1, 0, 3, 2, 2, 3, 0, 1], [0, 0, 0, 0], [1, 1, 1, 1], [3, 3, 3, 3]]
        num_rows = 4
        exps = [
            [
                "01" * 4,
                "0" * 8,
                "11" * 4,
                "10" * 4,
                "10" * 4,
                "11" * 4,
                "0" * 8,
                "01" * 4,
            ],
            ["0" * 4] * 4,
            ["01" * num_rows] * 4,
            ["11" * num_rows] * 4,
        ]
        pack_bits = 2

        for exp, t in zip(exps, ts):
            data = [t] * num_rows
            test_tensor = tensor(data, dtype=uint8)
            packed = pack_weights(test_tensor, pack_bits=pack_bits, dtype=uint8)
            exp_ints = [[int(e, 2) for e in exp]]
            self.assertEqual(
                exp_ints, packed.tolist(), f"[{t}] exp:{exp_ints}, got:{packed}"
            )
            unpacked = unpack_weights(packed, pack_bits=pack_bits)
            self.assertEqual(
                data, unpacked.tolist(), f"[{t}] exp:{data}, got:{unpacked}"
            )

    def test_2d_pack_unpack_4_8_bits(self):
        pack_bits = [4, 8]
        dtypes = [uint8, uint16, uint32, uint64, int8, int16, int32, int64]
        num_rows = 4
        row = [1, 0, 3, 2]

        from typing import List

        for p_bit, dtype in product(pack_bits, dtypes):
            num_rows_test = num_rows * p_bit
            t_id = (p_bit, dtype)
            data = [row[:]] * num_rows_test
            test_tensor = tensor(data, dtype=dtype)
            packed = pack_weights(test_tensor, pack_bits=p_bit)
            unpacked = unpack_weights(packed, pack_bits=p_bit)
            self.assertEqual(
                data, unpacked.tolist(), f"[{t_id}] exp:{data}, got:{unpacked}"
            )

    def test_bench_packing_unpacking(self):
        self.skipTest("Benchmarking is relatively slow")
        import time
        from weight_pack import pack_loop, unpack_loop
        from rust_enum import Case, enum

        @enum
        class WeightApply:
            Pack = Case(current=int, into=int)
            Unpack = Case(current=int, into=int)

            def get_num_rows(self, num_rows: int) -> int:
                match self:
                    case self.Pack(num_bits, bits_to_pack_to):
                        num_rows = num_rows * bits_to_pack_to // num_bits
                    case self.Unpack(num_bits, bits_to_unpack_to):
                        num_rows = num_rows * num_bits // bits_to_unpack_to
                    case _:
                        raise ValueError(f"Unexpected value for Weight: {self}")
                return num_rows

        def get_tensor_for_loop(
            self, num_cols: int, num_rows: int, into_dtype
        ) -> Tensor:
            tensor_f: Callable
            match self:
                case self.Pack(_, _):
                    size = [num_rows, num_cols]
                    tensor_f = partial(zeros, size)

                case self.Unpack(_, bits_to_unpack_to):
                    tensor_f = partial(
                        tensor,
                        data=[
                            [(1 << bits_to_unpack_to) - 1] * num_cols
                            for _ in range(num_rows)
                        ],
                    )
                case _:
                    raise ValueError(f"Unexpected value for Weight: {self}")

            return tensor_f(dtype=into_dtype)

        # old implementation with loops, it is vectorized now as above which is orders of magnitude faster
        def pack_loop_(
            to_pack_2d: Tensor, pack_bits: int, num_bits: int, data_type=uint8
        ):
            to_pack_num_rows, to_pack_num_cols = to_pack_2d.shape
            wa: WeightApply = WeightApply.Pack(num_bits, pack_bits)
            num_rows = wa.get_num_rows(to_pack_num_rows)
            new_tensor = get_tensor_for_loop(wa, to_pack_num_cols, num_rows, data_type)
            num_weights_to_pack = num_bits // pack_bits
            to_pack_2d = to_pack_2d.to(data_type)

            for c in range(to_pack_num_cols):
                col = to_pack_2d[:, c]
                for r in range(num_rows):
                    for i in range(num_weights_to_pack):
                        rr = r * num_weights_to_pack
                        e = col[rr + i]
                        new_tensor[r, c] |= e << (pack_bits * i)

            return new_tensor

        def unpack_loop_(
            to_unpack_2d: Tensor,
            pack_bits: int,
            num_bits: int,
            data_type=uint8,
        ):
            wa: WeightApply = WeightApply.Unpack(num_bits, pack_bits)
            row = get_tensor_for_loop(wa, 1, 1, data_type)

            num_rows, num_cols = to_unpack_2d.shape
            to_num_rows = wa.get_num_rows(num_rows)

            num_weights_to_unpack = num_bits // pack_bits
            size = [to_num_rows, to_unpack_2d.shape[-1]]
            new_tensor = full(size, row.min().item())

            for c in range(num_cols):
                col = to_unpack_2d[:, c]
                for r in range(to_num_rows):
                    rr = r // num_weights_to_unpack
                    for i in range(num_weights_to_unpack):
                        e = col[rr]
                        new_tensor[r, c] &= e >> (pack_bits * i)
            return new_tensor

        seed(9)
        pack_test_tensors = [
            (rand(sz) * 10).to(dt)
            for sz, dt in product([(128, 256), (256, 256), (512, 720)], [int64, uint64])
        ]
        unpack_test_tensors = []

        print()
        pack_functions = [pack_loop, pack_loop_]
        unpack_functions = [unpack_loop, unpack_loop_]
        pack_times = [[] for _ in pack_functions]
        unpack_times = [[] for _ in unpack_functions]

        def timed(f):
            def _wrapper(*args):
                start = time.time()
                r = f(*args)
                tm = time.time() - start
                return r, tm

            return _wrapper

        for test_tensor in pack_test_tensors:
            for i, (fp, fu) in enumerate(zip(pack_functions, unpack_functions)):
                fp = timed(fp)
                p, tm = fp(test_tensor, 2, 4)
                pack_times[i].append(tm)

                unpack_test_tensors.append(p)
                fu = timed(fu)
                u, tm = fu(p, 2, 4)
                unpack_times[i].append(tm)

        for i, (pt, ut) in enumerate(zip(pack_times, unpack_times)):
            sh_p = pack_test_tensors[i].shape
            sh_u = unpack_test_tensors[i].shape
            for p, u in zip(pt[:-1], ut[:-1]):
                print(
                    f"    {pack_functions[0].__name__} is  {pt[-1]/p:.1f}x faster for {sh_p}"
                )
                print(
                    f"    {unpack_functions[0].__name__} is  {ut[-1]/u:.1f}x faster for {sh_u}"
                )


if __name__ == "__main__":
    main()
