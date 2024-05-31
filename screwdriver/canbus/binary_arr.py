import numpy as np
import pandas as pd


def _binary_arr_helper(data):
    bin_arr = np.unpackbits(np.frombuffer(bytearray.fromhex(data[2:]), "uint8")).astype(
        int
    )
    return np.concatenate((bin_arr, np.array([np.nan] * (64 - len(bin_arr)))))


def binary_arr(data):
    hex = data["Hex"]
    binary = pd.DataFrame(
        np.array(list(map(_binary_arr_helper, hex.to_numpy()))),
        columns=[f"Bit {i}" for i in range(64)],
        dtype=object,
    )
    # print(data["ID"])
    binary["ID"] = data["ID"].values
    # self.binary["Time Seconds"] = self.data["Time Seconds"]
    # binary["Time Epoch"] = data["Time Epoch"]
    binary["Timestamp"] = data["Timestamp"].values

    return binary
