import numpy as np
import dill as pickle
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from .chi2 import χ2_weighted
from functools import partial


def save_fit(
    path,
    x,
    data,
    xfiltered,
    xrej,
    xfit,
    datafit,
    fitfunc,
    paramsfit,
    χ2ν,
):
    out_dict = {
        "data": {
            "x": x,
            "values": data,
            "avg": np.mean(data, axis=0),
            "std": np.std(data, axis=0),
            "xfiltered": xfiltered,
            "xrej": xrej,
        },
        "fit": {
            "x": xfit,
            "values": datafit,
            "avg": datafit.mean(axis=0),
            # "avg": fitfunc(xfit, *paramsfit.mean(axis=0)),
            "std": np.std(datafit, axis=0),
            "fit_func": fitfunc,
            "params": paramsfit,
            "χ2ν": χ2ν,
        },
    }
    with open(path, "wb") as file_out:
        pickle.dump(out_dict, file_out)

    return out_dict


def ensemble_curve_fit(
    fit_func, xdata, ydata, guess, bounds=None, data_axis=0, **kwargs
):
    Ndata = ydata.shape[data_axis]
    fit_params = np.array(
        [
            curve_fit(
                fit_func,
                xdata,
                ydata[data],
                sigma=ydata.std(axis=data_axis),
            )[0]
            for data in range(Ndata)
        ]
    )

    cov_matrix = np.cov(ydata.T)
    cov_matrix_inv = np.linalg.pinv(cov_matrix)
    fit_params_mean = fit_params.mean(axis=data_axis)
    χ2 = χ2_weighted(
        fit_params_mean,
        fit_func,
        xdata,
        ydata.take(indices=0, axis=data_axis),
        cov_matrix_inv,
    )
    dof = len(xdata) - len(guess)

    return fit_params, χ2 / dof, dof


def ensemble_fit(
    fit_func,
    xdata,
    ydata,
    guess,
    fixed_params=None,
    bounds=None,
    data_axis=0,
    **kwargs,
):
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    Ndata = ydata.shape[data_axis]

    if fixed_params != None:
        fixed_params_names, fixed_params_values = fixed_params

    weight_matrix = np.diag(np.var(ydata, axis=data_axis) ** (-1))

    cov_matrix = np.cov(ydata.T)
    try:
        cov_matrix_inv = np.linalg.pinv(cov_matrix)
    except:
        cov_matrix_inv = weight_matrix

    if fixed_params != None:
        fixed_params_mean = {
            fixed_params_names[iarg]: (fixed_params_values[:, iarg]).mean(axis=0)
            for iarg in range(len(fixed_params_names))
        }
        fit_func_mean = partial(fit_func, **fixed_params_mean)
    else:
        fit_func_mean = fit_func

    proper_guess = minimize(
        χ2_weighted,
        guess,
        args=(fit_func_mean, xdata, ydata.mean(axis=data_axis), weight_matrix),
        method="Nelder-Mead",
        bounds=bounds,
    ).x
    fit_data = []
    for n in range(Ndata):
        if fixed_params != None:
            fixed_params_dict = {
                fixed_params_names[i]: fixed_params_values[n, i]
                for i in range(len(fixed_params_names))
            }
            fit_func_n = partial(fit_func, **fixed_params_dict)
        else:
            fit_func_n = fit_func
        fit_data.append(
            minimize(
                χ2_weighted,
                proper_guess,
                args=(fit_func_n, xdata, ydata[n], weight_matrix),
                method="Nelder-Mead",
                bounds=bounds,
            )
        )

    fit_params = np.array([fit.x for fit in fit_data], dtype=fit_data[0].x.dtype)
    χ2 = χ2_weighted(
        fit_params.mean(axis=data_axis),
        fit_func_mean,
        xdata,
        ydata.take(indices=0, axis=data_axis),
        cov_matrix_inv,
    )
    dof = len(xdata) - len(guess)
    χ2ν = χ2 / dof if dof > 0 else None
    return fit_params, χ2ν, dof


def ensemble_fit_params_to_func(fit_func, x, params, data_axis=0):
    length = params.shape[data_axis]
    result = np.empty((length, len(x)))
    try:
        for n, func in enumerate(fit_func):
            result[n] = func(x, *params[n])
    except TypeError:
        result = np.array([fit_func(x, *params[n]) for n in range(length)])
    return result


def filter_main(xdata, ydata, xlims, stdmax=np.inf, data_axis=0, return_mask=False):
    xlim_mask = np.logical_and((xdata >= xlims[0]), (xdata <= xlims[1]))
    std_mask = np.std(ydata, axis=data_axis) <= stdmax
    keep_mask = np.logical_and(xlim_mask, std_mask)
    # fade out points outside xlims, remove points outside stdmax
    fade_mask = np.logical_and(~xlim_mask, std_mask)
    if return_mask:
        return (
            xdata[keep_mask],
            ydata[:, keep_mask],
            xdata[fade_mask],
            ydata[:, fade_mask],
            keep_mask,
        )
    else:
        return (
            xdata[keep_mask],
            ydata[:, keep_mask],
            xdata[fade_mask],
            ydata[:, fade_mask],
        )
