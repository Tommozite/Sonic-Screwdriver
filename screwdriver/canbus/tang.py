import numpy as np
import pandas as pd
import datetime


def rolling_tang_binned(capture, bin_interval=100):
    if capture.binary is None:
        capture.binary_arr()
    if type(bin_interval) == int:
        bins = [range(0, len(capture.binary), bin_interval)]
    else:
        try:
            delta = pd.Timedelta(bin_interval)
        except:
            raise ValueError("bin_interval could not be interpreted as timedelta")
        timerange = (
            capture.binary["Timestamp"].max() - capture.binary["Timestamp"].min()
        )
        num_bins = timerange // delta
        bins = [
            capture.binary["Timestamp"].min() + i * delta for i in range(num_bins + 1)
        ]
    result = {}
    for int_min, int_max in zip(bins, bins[1:]):
        result[int_min] = _rolling_tang_binned_helper(capture, int_min, int_max)
    return result


def _rolling_tang_binned_helper(capture, int_min, int_max):
    result = []
    if type(int_min) == pd.Timestamp:
        temp = capture.binary[
            (capture.binary["Timestamp"] > int_min)
            & (capture.binary["Timestamp"] < int_max)
        ]
    elif type(int_min) == int:
        temp = capture.binary.iloc[int_min : int_max + 1]
    tang = temp.groupby("ID").apply(_tang_helper)

    for id in capture.ids:
        try:
            result.append(tang.loc[id].to_list())
        except KeyError:
            result.append([0] * 64)
    result = pd.DataFrame(result, columns=[f"Bit {i}" for i in range(64)])
    result["ID"] = capture.ids
    return result


def rolling_tang(capture, interval=1):
    result = []
    if capture.binary is None:
        capture.binary_arr()
    temp = capture.binary
    temp = temp.sort_values("Timestamp")
    result = []
    for i, window in enumerate(temp.rolling(interval, on="Timestamp")):
        result.append(_rolling_tang_helper(window))
    result = pd.DataFrame(
        result,
        columns=[
            "ID",
            *[f"Bit {i}" for i in range(64)],
            "Time Seconds",
            "Time Epoch",
            "Timestamp",
        ],
    )
    return result


def _rolling_tang_helper(capture_window):
    temp_id = capture_window["ID"].iloc[0]

    tang = _tang_helper(temp_id, capture_window)
    result = [
        temp_id,
        *tang,
        capture_window["Time Seconds"].iloc[0],
        capture_window["Time Epoch"].iloc[0],
        capture_window["Timestamp"].iloc[0],
    ]
    return result


def _tang_helper(data):
    temp = data[[f"Bit {i}" for i in range(64)]]
    temp = temp.fillna(0)
    result = temp.diff().fillna(0).abs().sum()
    return result


def calc_tang(capture, mask=None):
    result = []
    if capture.binary is None:
        capture.binary_arr()
    if mask is None:
        data = capture.binary[[f"Bit {i}" for i in range(64)]]
        ids = capture.binary["ID"]
    else:
        data = capture.binary[[f"Bit {i}" for i in range(64)]][mask]
        ids = capture.binary["ID"][mask]
    for id in capture.ids:
        tmp = data[(ids == id).to_numpy()]
        tang = np.abs(np.roll(tmp, 1, axis=0) - tmp).mean(axis=0)
        result.append([id, *tang])
    result = pd.DataFrame(result, columns=["ID", *[f"Bit {i}" for i in range(64)]])
    if mask is None:
        capture.tang = result
    return result


def tang_pearson_coeff(df_dict, event_mask):
    event_mask = event_mask.astype(int)
    temp = [df for df in df_dict.values()]
    ids = temp[0]["ID"]

    for df, time in zip(temp, df_dict.keys()):
        df["Timestamp"] = time
    temp = pd.concat(temp, ignore_index=True)
    result = []
    for id in ids:
        data = temp[temp["ID"] == id][[f"Bit {i}" for i in range(64)]]
        cov = np.dot(event_mask - event_mask.mean(), data - data.mean(axis=0)) / (
            event_mask.shape[0] - 1
        )
        corr = cov / np.sqrt(np.var(event_mask, ddof=1) * np.var(data, axis=0, ddof=1))
        corr[np.isnan(corr)] = 0
        result.append([id, *corr])
    result = pd.DataFrame(result, columns=["ID", *[f"Bit {i}" for i in range(64)]])
    return result
