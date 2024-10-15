import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t, poisson, norm


def interrupt_finder(data, n_largest=500, alpha=1e-6):
    times = data["TimeEpoch"]
    times_diff = times.diff()
    mean, std = times_diff.mean(), times_diff.std()
    z_scores = ((times_diff - mean).abs() / (std)).nlargest(n_largest)
    result = []
    outlier = True
    while outlier:
        outlier_mask = z_scores.index.isin(result)
        tmp = z_scores[~outlier_mask]
        grubbs_mean, grubbs_std = tmp.mean(), tmp.std()
        grubbs = ((tmp - grubbs_mean) / grubbs_std).max()
        N = len(tmp)
        val = t.ppf(alpha / 2, df=N - 2)
        crit = (N - 1) / np.sqrt(N) * np.sqrt((val**2) / (N - 2 + val**2))
        outlier = grubbs > crit
        if outlier:
            result.append(tmp.idxmax())

    return result


def avg_rate_rem_interrupts(data):
    arb_ids = data["ID"].unique()
    interrupts = sorted(interrupt_finder(data))
    total_time = data["TimeEpoch"].max() - data["TimeEpoch"].min()
    for i in interrupts:
        total_time -= data["TimeEpoch"].iloc[i + 1] - data["TimeEpoch"].iloc[i - 1]
    rates = {"total": [len(data), total_time]}
    for arb_id in arb_ids:
        tmp = data[data["ID"] == arb_id]
        rates[arb_id] = [len(tmp), total_time]
    return rates


def avg_rate_slices(data):
    arb_ids = data["ID"].unique()
    interrupts = [0] + sorted(interrupt_finder(data)) + [len(data)]
    rates = {arb_id: [0, 0] for arb_id in arb_ids}
    rates["total"] = [0, 0]
    for i0, i1 in zip(interrupts[:-1], interrupts[1:]):
        if i1 - i0 < 100:
            continue
        tmp = data.iloc[i0 + 2 : i1 - 2]
        rates["total"][0] += len(tmp)
        rates["total"][1] += tmp["TimeEpoch"].max() - tmp["TimeEpoch"].min()
        for arb_id in arb_ids:
            tmp2 = tmp[tmp["ID"] == arb_id]
            if len(tmp2) > 1:
                rates[arb_id][0] += len(tmp2)
                rates[arb_id][1] += tmp2["TimeEpoch"].max() - tmp2["TimeEpoch"].min()
    return rates
