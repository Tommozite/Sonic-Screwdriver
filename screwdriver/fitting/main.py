import numpy as np
import dill as pickle
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from .chi2 import χ2_weighted


def save_fit(
    path,
    x,
    data,
    xfiltered,
    xrej,
    xfit,
    datafit_mean,
    datafit_ens,
    fitfunc,
    paramsfit_mean,
    paramsfit_ens,
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
            "values": datafit_ens,
            "avg": datafit_mean,
            # "avg": fitfunc(xfit, *paramsfit.mean(axis=0)),
            "std": np.std(datafit_ens, axis=0),
            "fit_func": fitfunc,
            "params": paramsfit_ens,
            "params_mean": paramsfit_mean,
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
            curve_fit(fit_func, xdata, ydata[data], sigma=ydata.std(axis=data_axis),)[0]
            for data in range(Ndata)
        ]
    )
    fit_mean = np.array(
        curve_fit(
            fit_func,
            xdata,
            ydata.take(indices=0, axis=data_axis),
            sigma=ydata.std(axis=data_axis),
        )[0]
    )

    cov_matrix = np.cov(ydata.T)
    cov_matrix_inv = np.linalg.pinv(cov_matrix)

    χ2 = χ2_weighted(
        fit_mean, fit_func, xdata, ydata.take(indices=0, axis=data_axis), cov_matrix_inv
    )
    dof = len(xdata) - len(guess)

    return fit_mean, fit_params, χ2 / dof, dof


def ensemble_fit(fit_func, xdata, ydata, guess, bounds=None, data_axis=0, **kwargs):
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    Ndata = ydata.shape[data_axis]

    weight_matrix = np.diag(np.var(ydata, axis=data_axis) ** (-1))

    cov_matrix = np.cov(ydata.T)
    cov_matrix_inv = np.linalg.pinv(cov_matrix)

    proper_guess = minimize(
        χ2_weighted,
        guess,
        args=(fit_func, xdata, ydata.mean(axis=data_axis), weight_matrix),
        method="Nelder-Mead",
        bounds=bounds,
    ).x
    fit_data = [
        minimize(
            χ2_weighted,
            proper_guess,
            args=(fit_func, xdata, ydata[data], weight_matrix),
            method="Nelder-Mead",
            bounds=bounds,
        )
        for data in range(Ndata)
    ]
    fit_params = np.array([fit.x for fit in fit_data], dtype=fit_data[0].x.dtype)
    fit_mean = minimize(
        χ2_weighted,
        proper_guess,
        args=(fit_func, xdata, ydata.take(indices=0, axis=data_axis), weight_matrix),
        method="Nelder-Mead",
        bounds=bounds,
    )
    χ2 = χ2_weighted(
        fit_mean.x,
        fit_func,
        xdata,
        ydata.take(indices=0, axis=data_axis),
        cov_matrix_inv,
    )
    dof = len(xdata) - len(guess)
    χ2ν = χ2 / dof if dof > 0 else None
    return fit_mean.x, fit_params, χ2ν, dof


def ensemble_fit_funcarray(
    fit_func_mean,
    fit_func_array,
    xdata,
    ydata,
    guess,
    bounds=None,
    data_axis=0,
    **kwargs,
):
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    Ndata = ydata.shape[data_axis]
    weight_matrix = np.linalg.inv(np.cov(ydata.T))
    proper_guess = minimize(
        χ2_weighted,
        guess,
        args=(fit_func_mean, xdata, ydata.mean(axis=data_axis), weight_matrix),
        method="Nelder-Mead",
        bounds=bounds,
    ).x
    fit_data = [
        minimize(
            χ2_weighted,
            proper_guess,
            args=(fit_func_array[data], xdata, ydata[data], weight_matrix),
            method="Nelder-Mead",
            bounds=bounds,
        )
        for data in range(Ndata)
    ]
    fit_params = np.array([fit.x for fit in fit_data], dtype=fit_data[0].x.dtype)
    # print([fit.x for fit in fit_data])
    fit_mean = minimize(
        χ2_weighted,
        proper_guess,
        args=(fit_func_mean, xdata, ydata.mean(axis=data_axis), weight_matrix),
        method="Nelder-Mead",
        bounds=bounds,
    )
    dof = len(xdata) - len(guess)
    χ2ν = fit_mean.fun / dof if dof > 0 else None
    return fit_mean.x, fit_params, χ2ν, dof


def ensemble_fit_params_to_func(fit_func, x, params_mean, params, data_axis=0):
    length = params.shape[data_axis]
    result_mean = np.array(fit_func(x, *params_mean))
    result = np.array([fit_func(x, *params[n]) for n in range(length)])
    return result_mean, result


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
