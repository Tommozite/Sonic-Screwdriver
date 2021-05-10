import numpy as np
from scipy.optimize import curve_fit
from functools import partial


def χ2(func, params, xdata, ydata):
    # axis 0 is bootstraps
    cov_mat = np.cov(ydata.T)
    cov_mat_inv = np.linalg.inv(cov_mat)
    diff = func(xdata, *params.T) - np.mean(ydata.T, axis=1)
    result = np.dot(diff, np.dot(cov_mat_inv, diff))
    return result


def ensemble_fit_params(
    fit_func, xdata, ydata, bounds=(-np.inf, np.inf), guess=None, data_axis=0
):
    Ndata = ydata.shape[data_axis]
    try:
        fit_params = np.array(
            [
                curve_fit(
                    fit_func,
                    xdata,
                    ydata[data],
                    sigma=ydata.std(axis=data_axis),
                    bounds=bounds,
                    p0=guess,
                )[0]
                for data in range(Ndata)
            ]
        )
        dof = len(xdata) - len(fit_params[0])
        fit_xdata = np.array(
            [fit_func(xdata, *fit_params[data]) for data in range(Ndata)]
        ).mean(axis=0)
    except TypeError:
        # Support for array of functions for each data slice (i.e. input of other
        # parameters obtained for each slice)
        fit_params = np.array(
            [
                curve_fit(
                    fit_func[data],
                    xdata,
                    ydata[data],
                    sigma=ydata.std(axis=data_axis),
                    bounds=bounds,
                    p0=guess,
                )[0]
                for data in range(Ndata)
            ]
        )
        dof = len(xdata) - len(fit_params[0])
        fit_xdata = np.array(
            [fit_func(xdata, *fit_params[data]) for data in range(Ndata)]
        ).mean(axis=0)
    except:
        raise
    χ2 = sum(
        [
            ((ydata.mean(axis=data_axis)[i] - fit_xdata[i]) ** 2)
            / (ydata.std(axis=data_axis)[i] ** 2)
            for i in range(len(xdata))
        ]
    )

    χ2ν = χ2 / dof

    return fit_params, χ2ν


def ensemble_fit_params_to_fit(fit_params, xlinspace, fit_func, data_axis=0):
    Ndata = fit_params.shape[data_axis]
    try:
        fit = np.array(
            [fit_func(xlinspace, *fit_params[data]) for data in range(Ndata)]
        )
    except TypeError:
        fit = np.array(
            [fit_func[data](xlinspace, *fit_params[data]) for data in range(Ndata)]
        )
    return fit


def ensemble_fit(
    fit_func, xdata, ydata, xlinspace, bounds=None, guess=None, data_axis=0
):
    fit_params, χ2ν = ensemble_fit_params(
        fit_func, xdata, ydata, bounds, guess, data_axis=data_axis
    )
    fit = ensemble_fit_params_to_fit(fit_params, xlinspace, fit_func, data_axis)
    return fit, fit_params, χ2ν


def filter_main(xdata, ydata, xlims, stdmax, data_axis=0):
    xlim_mask = np.logical_and((xdata >= xlims[0]), (xdata <= xlims[1]))
    std_mask = np.std(ydata, axis=data_axis) <= stdmax
    keep_mask = np.logical_and(xlim_mask, std_mask)
    # fade out points outside xlims, remove points outside stdmax
    fade_mask = np.logical_and(~xlim_mask, std_mask)
    return (
        xdata[keep_mask],
        ydata[:, keep_mask],
        xdata[fade_mask],
        ydata[:, fade_mask],
    )


def chiral_extrapolation_dict(mass_array, data, fit_func):
    result = []
    χ_fit_params = []
    χ2ν = []
    mass_array = np.array(mass_array)
    try:
        for p in range(len(data[list(data.keys())[0]][0, :])):
            y_array = np.array([data[k][:, p] for k in data.keys()]).transpose()
            x_linspace = np.linspace(
                min(np.min(mass_array), 0), max(np.max(mass_array), 0)
            )
            χ_fit, χ_fit_params_temp, χ2ν_temp = ensemble_fit(
                fit_func, mass_array, y_array, x_linspace
            )
            result.append(χ_fit[:, 0])
            χ_fit_params.append(χ_fit_params_temp)
            χ2ν.append(χ2ν_temp)
        result = np.array(result).transpose()
    except TypeError:
        result = np.array(data[list(data.keys())[0]])
        χ_fit_params = None
        χ2ν = None
    except:
        raise
    return result, χ_fit_params, χ2ν


def G2_filter(G2_time, G2, limits, κ, **kwargs):
    G2_time_filtered = {}
    G2_filtered = {}
    G2_linspace = {}
    for k in κ:
        G2_filter_mask = np.logical_and(
            G2_time[k] >= limits[k][0], G2_time[k] <= limits[k][1]
        )
        G2_time_filtered[k] = np.array(G2_time[k][G2_filter_mask])
        G2_filtered[k] = np.array(G2[k][:, G2_filter_mask])
        res = 5
        linnum = res * (limits[k][1] - limits[k][0])
        G2_linspace[k] = np.linspace(limits[k][0], limits[k][1], num=linnum)
    return G2_time_filtered, G2_filtered, G2_linspace


def trapezoid_dict(t_range, vertices):
    """Vertices go (t,τ), returns {t: [τ_low, τ_high]} for t in _range"""
    bounds = {}
    vertices = np.array(vertices)
    slope_bottom = (
        (vertices[1, 1] - vertices[0, 1]) / (vertices[1, 0] - vertices[0, 0])
        if vertices[1, 0] - vertices[0, 0]
        else 0
    )
    slope_top = (
        (vertices[2, 1] - vertices[3, 1]) / (vertices[2, 0] - vertices[3, 0])
        if vertices[2, 0] - vertices[3, 0]
        else 0
    )
    for t in t_range:
        if t < vertices[0, 0] or t > vertices[1, 0]:
            bounds[t] = [-1, -1]
        else:
            if vertices[0, 1] > vertices[1, 1]:
                limit_low = slope_bottom * (t - vertices[0, 0]) + vertices[0, 1]
            else:
                limit_low = slope_bottom * (t - vertices[1, 0]) + vertices[1, 1]
            if vertices[2, 1] < vertices[3, 1]:
                limit_high = slope_top * (t - vertices[2, 0]) + vertices[2, 1]
            else:
                limit_high = slope_top * (t - vertices[3, 0]) + vertices[3, 1]
            bounds[t] = [round(limit_low), round(limit_high)]
    return bounds


def G3_filter(sink_times, G3_time, G3, G3_filter_vertices, κ, **kwargs):
    G3_time_filtered = {}
    G3_filtered = {}
    G3_linspace = {}
    for k in κ:
        G3_filter_limits = trapezoid_dict(G3_time[k], G3_filter_vertices[k])
        G3_filter_mask = {
            t: np.logical_and(
                G3_time[k][t] >= G3_filter_limits[t][0],
                G3_time[k][t] <= G3_filter_limits[t][1],
            )
            for t in sink_times[k]
        }
        G3_time_filtered[k] = {
            t: G3_time[k][t][G3_filter_mask[t]] for t in sink_times[k]
        }

        G3_filtered[k] = {t: G3[k][t][:, G3_filter_mask[t]] for t in sink_times[k]}
        G3_linspace[k] = {
            t: np.linspace(G3_filter_limits[t][0], G3_filter_limits[t][1])
            for t in sink_times[k]
        }
    return G3_time_filtered, G3_filtered, G3_linspace


def G2_fit(
    G2_fit_func, G2_time_filtered, G2_filtered, G2_linspace, bounds, κ, Nboot, **kwargs
):
    G2_fit_params = {}
    G2_fit = {}
    for k in κ:
        G2_sigma = np.std(G2_filtered[k], axis=0)
        G2_fit_params[k] = np.array(
            [
                curve_fit(
                    G2_fit_func,
                    G2_time_filtered[k],
                    G2_filtered[k][n],
                    bounds=bounds[k],
                    sigma=G2_sigma,
                )[0]
                for n in range(Nboot)
            ]
        )
        G2_fit[k] = np.array(
            [G2_fit_func(G2_linspace[k], *G2_fit_params[k][n]) for n in range(Nboot)]
        )
    return G2_fit_params, G2_fit


def G3_fit(
    G3_fit_func,
    sink_times,
    G3_time_filtered,
    G3_filtered,
    G3_linspace,
    G2_fit_params,
    bounds,
    κ,
    Nboot,
    **kwargs
):
    G3_fit_params = {}
    G3_fit = {}
    for k in κ:
        G3_sigma = np.concatenate(
            [G3_filtered[k][t].std(axis=0) for t in sink_times[k]]
        )
        G3_fit_params[k] = np.array(
            [
                curve_fit(
                    partial(G3_fit_func, *G2_fit_params[k][n]),
                    np.array(
                        [[t, τ] for t in sink_times[k] for τ in G3_time_filtered[k][t]]
                    ),
                    np.concatenate([G3_filtered[k][t][n] for t in sink_times[k]]),
                    sigma=G3_sigma,
                    bounds=bounds[k],
                )[0]
                for n in range(Nboot)
            ]
        )

        G3_fit[k] = {
            t: np.array(
                [
                    G3_fit_func(
                        *G2_fit_params[k][n],
                        np.array(
                            [
                                [t, G3_linspace[k][t][i]]
                                for i in range(len(G3_linspace[k][t]))
                            ]
                        ),
                        *G3_fit_params[k][n]
                    )
                    for n in range(Nboot)
                ]
            )
            for t in sink_times[k]
        }

    return G3_fit_params, G3_fit
