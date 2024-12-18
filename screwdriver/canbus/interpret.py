import pandas as pd
import numpy as np


def _uint_helper(row, start, end, endianness):
    return sum(
        [
            row[f"Bit{i}"] * 2 ** (ii)
            for ii, i in (
                enumerate(range(end, start - 1, -1))
                if endianness == "big"
                else enumerate(range(start, end + 1, +1))
            )
        ]
    )


def unsigned_integer(df, start, end, bit_endianness="big"):
    return df.apply(
        _uint_helper, axis=1, start=start, end=end, endianness=bit_endianness
    )


def _int_helper(row, start, end, endianness):
    quant = sum(
        [
            row[f"Bit{i}"] * 2 ** (ii)
            for ii, i in (
                enumerate(range(end, start, -1))
                if endianness == "big"
                else enumerate(range(start, end, 1))
            )
        ]
    )
    return quant - 2 ** (end - start) if quant >= 2 ** (end - start - 1) else quant


def signed_integer(df, start, end, bit_endianness="big"):
    return df.apply(
        _int_helper, axis=1, start=start, end=end, endianness=bit_endianness
    )
