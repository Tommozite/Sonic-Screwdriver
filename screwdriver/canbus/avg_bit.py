import numpy as np
import pandas as pd

from .capture import Capture


def bit_avg_diff(capture, mask):
    bit_avg_on = capture.calc_bit_avg(mask)
    bit_avg_off = capture.calc_bit_avg(~mask)
    diff = [
        [id_, np.abs(np.array(x) - np.array(y))]
        for (id_, *x), (_, *y) in zip(bit_avg_on, bit_avg_off)
    ]
    result = []
    for id_, x in diff:
        result.append([id_, *x])
    result = pd.DataFrame(result, columns=["ID", *[f"Bit {i}" for i in range(64)]])
    return result


def bit_power_avg_diff(capture, power=1, mask=None):
    bit_avg_on = capture.calc_bit_power_avg(power, mask)
    bit_avg_off = capture.calc_bit_power_avg(power, ~mask)
    diff = [[id_, np.abs(x - y)] for (id_, x), (_, y) in zip(bit_avg_on, bit_avg_off)]
    result = []
    for id_, x in diff:
        result.append([id_, *x])
    result = pd.DataFrame(result, columns=["ID", *[f"Bit {i}" for i in range(64)]])
    return result


def bit_avg_intervals(capture, intervals):
    result = []
    for i, int in enumerate(intervals):
        mask = np.array([False] * len(capture.data))
        mask[
            np.logical_and(
                (capture.data["Timestamp"] > int[0]).to_numpy(),
                (capture.data["Timestamp"] > int[1]).to_numpy(),
            )
        ] = True
        bit_avg = capture.calc_bit_avg(mask)
        tmp = []
        for id_, x in bit_avg:
            tmp.append([id_, *x])
        tmp = pd.DataFrame(tmp, columns=["ID", *[f"Bit {i}" for i in range(64)]])
        result.append(tmp)

    return result
