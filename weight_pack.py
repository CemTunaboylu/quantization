from torch import (
    empty,
    Tensor,
    stack,
    tensor,
    bitwise_or,
)
from torch import uint8, uint16, uint32, uint64

from typing import Callable, Generator, List
from functools import reduce

MIN_TO_PACK = 4


def check_smaller_than_min_to_pack(num_bits: int):
    if num_bits < MIN_TO_PACK:
        raise Exception(f"nothing to pack with tensor of {num_bits}.")


def check_tensor_shape_is_multiple_of_pack_bits(shape, pack_bits: int, num_bits: int):
    if shape[0] * pack_bits % num_bits != 0:
        raise Exception(
            f"tensor shape should be multiple of { num_bits // pack_bits }, got: {shape}."
        )


def get_dtype_size_in_bits(dtype) -> int:
    return dtype.itemsize * 8


def get_max_bits_needed(t: Tensor) -> int:
    if t.is_floating_point() or t.is_complex():
        return get_dtype_size_in_bits(t.dtype)
    else:
        mx: int = t.max().item()
        if not t.is_signed():
            return mx.bit_length()
        if abs(mn := t.min().item()) > mx:
            return mn.bit_length()
        return mx.bit_length()


# TODO: should pack/unpack allow partials/leftovers which are filled with auto values
# e.g. last column has less elements in the packed integer
def pack_weights(
    to_pack_2d: Tensor,
    pack_bits: int,
    dtype=uint8,
    ensure_methods: List[Callable] = [],
    compress: bool = False,
) -> Tensor:
    d = len(to_pack_2d.shape)
    assert (
        2 == d
    ), f"weight packing is not implemented in {d}-D, consider slicing it into 2d."

    [e(to_pack_2d, pack_bits) for e in ensure_methods]

    num_bits: int = (
        get_dtype_size_in_bits(to_pack_2d.dtype)
        if not compress
        else get_max_bits_needed(to_pack_2d)
    )

    check_smaller_than_min_to_pack(num_bits)
    check_tensor_shape_is_multiple_of_pack_bits(to_pack_2d.shape, pack_bits, num_bits)

    return pack_loop(to_pack_2d, pack_bits, num_bits, dtype)


def unpack_weights(
    packed: Tensor,
    pack_bits: int,
    ensure_methods: List[Callable] = [],
    dtype=None,
) -> Tensor:
    [e(packed, pack_bits) for e in ensure_methods]

    return unpack_loop(packed, pack_bits, dtype)


def get_num_rows_for_packing(
    current_num_rows: int, target_dtype, bits_to_pack: int
) -> int:
    num_target_bits = get_dtype_size_in_bits(target_dtype)
    available_spots = num_target_bits // bits_to_pack
    if 0 == available_spots:
        msg = (
            f"Cannot pack tensor with {current_num_rows} rows "
            + f"into {num_target_bits} with {bits_to_pack} bits each"
        )
        raise Exception(msg)
    if 0 != (lo := current_num_rows % available_spots):
        msg = (
            f"There will be leftovers if tensor with {current_num_rows} rows are packed "
            + f"into {num_target_bits} with {bits_to_pack} bits each : left overs: {lo}"
        )
        raise Exception(msg)
    return current_num_rows // available_spots


def bit_shifts(
    num_shifts: int,
    shift_amt: int,
    dtype,
    *after_transforms: Callable,
) -> Tensor:
    t = tensor([shift_amt * i for i in range(num_shifts)], dtype=dtype).unsqueeze_(1)
    for ft in after_transforms:
        t = ft(t)
    return t


def create_ranges(until_incl: int, step: int, start: int = 0) -> Generator:
    p = start
    for n in range(start + 1, until_incl + 1):
        yield (p, m := n * step)
        p = m


def pack_loop(to_pack_2d: Tensor, pack_bits: int, num_bits: int, dtype):
    to_pack_num_rows, to_pack_num_cols = to_pack_2d.shape
    to_num_rows = get_num_rows_for_packing(to_pack_num_rows, dtype, pack_bits)
    num_weights_to_pack = to_pack_num_rows // to_num_rows

    bits_to_move = bit_shifts(
        num_weights_to_pack,
        pack_bits,
        dtype,
        lambda t: t.expand(-1, to_pack_num_cols),
    )

    ranges = create_ranges(to_num_rows, num_weights_to_pack)

    packed: List[Tensor] = [empty([])] * to_num_rows

    for n_r in range(to_num_rows):
        (start, finish) = next(ranges)
        weight_rows_batch = to_pack_2d[start:finish, :].to(dtype)
        shifted_weights = weight_rows_batch.bitwise_left_shift_(bits_to_move)
        squished_on_or = reduce(bitwise_or, shifted_weights)
        packed[n_r] = squished_on_or

    return stack(packed)


def unpack_loop(
    to_unpack_2d: Tensor,
    pack_bits: int,
    dtype=uint8,
):
    num_bits: int = get_dtype_size_in_bits(to_unpack_2d.dtype)
    to_unpack_num_rows, to_unpack_num_cols = to_unpack_2d.shape

    if 0 == to_unpack_num_rows:
        return empty([])

    if None == dtype:
        pass

    num_weights_to_unpack = num_bits // pack_bits

    bits_to_move = bit_shifts(num_weights_to_unpack, pack_bits, dtype)

    unpacked: Tensor = empty(
        [to_unpack_num_rows * num_weights_to_unpack, to_unpack_num_cols], dtype=dtype
    )
    and_with = tensor(data=[(1 << pack_bits) - 1], dtype=dtype).expand(
        num_weights_to_unpack,
        to_unpack_num_cols,
    )

    ranges = create_ranges(to_unpack_num_rows, num_weights_to_unpack)
    for r in range(to_unpack_num_rows):
        row = to_unpack_2d[r].repeat(num_weights_to_unpack, 1)
        shifted_weights = row.bitwise_right_shift_(bits_to_move)
        removed_interference_with_and = shifted_weights.bitwise_and_(and_with)
        (start, finish) = next(ranges)
        unpacked[start:finish] = removed_interference_with_and

    return unpacked
