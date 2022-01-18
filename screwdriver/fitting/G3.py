import numpy as np
from ..ansatz import G2, G3, R
from functools import partial
from .main import ensemble_fit_funcarray


def filter_G3(xdata, ydata, xlims, stdmax=np.inf, data_axis=0):
    if len(xlims[xdata[0, 1]]) == 2:
        xlim_mask = np.logical_and(
            (xdata[:, 0] >= np.array([xlims[x][0] for x in xdata[:, 1]])),
            (xdata[:, 0] <= np.array([xlims[x][1] for x in xdata[:, 1]])),
        )
    elif len(xlims[xdata[0, 1]]) == 4:
        xlim_mask_0 = np.logical_and(
            (xdata[:, 0] >= np.array([xlims[x][0] for x in xdata[:, 1]])),
            (xdata[:, 0] <= np.array([xlims[x][1] for x in xdata[:, 1]])),
        )
        xlim_mask_1 = np.logical_and(
            (xdata[:, 0] >= np.array([xlims[x][2] for x in xdata[:, 1]])),
            (xdata[:, 0] <= np.array([xlims[x][3] for x in xdata[:, 1]])),
        )
        xlim_mask = np.logical_or(xlim_mask_0, xlim_mask_1)
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


def fit_G3(
    G3_time,
    G3_data,
    G2_data,
    G2_fit_mean,
    G2_fit_params,
    guess,
    G3_tlim=None,
    data_axis=0,
    data_normalise=True,
    **kwargs,
):
    # Get fitting functions
    G2_fit_func = G2.G2_get_fit_func(**kwargs)
    G3_fit_func = G3.G3_get_fit_func(**kwargs)

    # Construct the function array
    G3_func_mean_dict = {
        k: G2_fit_mean[G2_fit_func.args.index(k)] for k in G3_fit_func.G2_args
    }

    G3_func_array_dict = [
        {k: G2_fit_params[n, G2_fit_func.args.index(k)] for k in G3_fit_func.G2_args}
        for n in range(kwargs["Nboot"])
    ]
    G3_func_mean = partial(G3_fit_func.eval, **G3_func_mean_dict)
    G3_func_array = [
        partial(G3_fit_func.eval, **G3_func_array_dict[n])
        for n in range(kwargs["Nboot"])
    ]

    G3_tlin = np.array(
        [
            [τ, t]
            for t in kwargs["t_sink"]
            for τ in np.linspace(G3_tlim[t][0], G3_tlim[t][1])
        ]
    )

    G3_time_fit, G3_data_fit, G3_time_rej, G3_data_rej = filter_G3(
        G3_time, G3_data, G3_tlim
    )

    G3_fit_mean, G3_fit_params, G3_fit_χ2ν, dof = ensemble_fit_funcarray(
        G3_func_mean, G3_func_array, G3_time_fit, G3_data_fit, guess
    )

    G3_func_mean_dict = dict(
        {k: G2_fit_mean[G2_fit_func.args.index(k)] for k in G3_fit_func.G2_args},
        **{k: G3_fit_mean[G3_fit_func.args.index(k)] for k in G3_fit_func.args},
    )
    G3_func_array_dict = [
        dict(
            {
                k: G2_fit_params[n, G2_fit_func.args.index(k)]
                for k in G3_fit_func.G2_args
            },
            **{
                k: G3_fit_params[n, G3_fit_func.args.index(k)] for k in G3_fit_func.args
            },
        )
        for n in range(kwargs["Nboot"])
    ]
    G3_func_mean = partial(G3_fit_func.eval, **G3_func_mean_dict)
    G3_func_array = [
        partial(G3_fit_func.eval, **G3_func_array_dict[n])
        for n in range(kwargs["Nboot"])
    ]

    G3_fit = np.array([G3_func_array[n](G3_tlin) for n in range(kwargs["Nboot"])])
    G3_mean = G3_fit.mean(axis=data_axis)
    G3_std = G3_fit.std(axis=data_axis)

    return (
        G3_data,
        G3_time_fit,
        G3_time_rej,
        G3_tlin,
        G3_mean,
        G3_std,
        G3_fit,
        G3_fit_func,
        G3_fit_mean,
        G3_fit_params,
        G3_fit_χ2ν,
    )


def fit_G3_ratio_func(
    G3_time,
    G3_data,
    G2_data,
    G2_fit_mean,
    G2_fit_params,
    guess,
    G3_tlim=None,
    data_axis=0,
    data_normalise=True,
    **kwargs,
):
    # Get fitting functions
    G2_fit_func = G2.G2_get_fit_func(**kwargs)
    G3_fit_func = G3.G3_get_fit_func(**kwargs)
    R_fit_func = R.R_get_fit_func(**kwargs)

    # Construct the function array
    G3_func_mean_dict = {
        k: G2_fit_mean[G2_fit_func.args.index(k)] for k in G3_fit_func.G2_args
    }

    G3_func_array_dict = [
        {k: G2_fit_params[n, G2_fit_func.args.index(k)] for k in G3_fit_func.G2_args}
        for n in range(kwargs["Nboot"])
    ]
    G3_func_mean = partial(G3_fit_func.eval, **G3_func_mean_dict)
    G3_func_array = [
        partial(G3_fit_func.eval, **G3_func_array_dict[n])
        for n in range(kwargs["Nboot"])
    ]

    G3_tlin = np.array(
        [
            [τ, t]
            for t in kwargs["t_sink"]
            for τ in np.linspace(G3_tlim[t][0], G3_tlim[t][1])
        ]
    )

    G3_time_fit, G3_data_fit, G3_time_rej, G3_data_rej = filter_G3(
        G3_time, G3_data, G3_tlim
    )

    G3_fit_mean, G3_fit_params, G3_fit_χ2ν, dof = ensemble_fit_funcarray(
        G3_func_mean, G3_func_array, G3_time_fit, G3_data_fit, guess
    )

    R_data = G3_data / G2_data[:, G3_time[:, 1]]

    if data_normalise:
        G3_func_mean_dict = dict(
            {k: G2_fit_mean[G2_fit_func.args.index(k)] for k in G3_fit_func.G2_args},
            **{k: G3_fit_mean[G3_fit_func.args.index(k)] for k in G3_fit_func.args},
        )
        G3_func_array_dict = [
            dict(
                {
                    k: G2_fit_params[n, G2_fit_func.args.index(k)]
                    for k in G3_fit_func.G2_args
                },
                **{
                    k: G3_fit_params[n, G3_fit_func.args.index(k)]
                    for k in G3_fit_func.args
                },
            )
            for n in range(kwargs["Nboot"])
        ]
        G3_func_mean = partial(G3_fit_func.eval, **G3_func_mean_dict)
        G3_func_array = [
            partial(G3_fit_func.eval, **G3_func_array_dict[n])
            for n in range(kwargs["Nboot"])
        ]

        R_fit = (
            np.array([G3_func_array[n](G3_tlin) for n in range(kwargs["Nboot"])])
            / np.array([G2_data[:, round(t[1])] for t in G3_tlin]).T
        )
        R_mean = R_fit.mean(axis=data_axis)
        R_std = R_fit.std(axis=data_axis)
    else:
        R_func_mean_dict = dict(
            {k: G2_fit_mean[G2_fit_func.args.index(k)] for k in R_fit_func.G2_args},
            **{k: G3_fit_mean[G3_fit_func.args.index(k)] for k in R_fit_func.G3_args},
        )
        R_func_array_dict = [
            dict(
                {
                    k: G2_fit_params[n, G2_fit_func.args.index(k)]
                    for k in R_fit_func.G2_args
                },
                **{
                    k: G3_fit_params[n, G3_fit_func.args.index(k)]
                    for k in R_fit_func.G3_args
                },
            )
            for n in range(kwargs["Nboot"])
        ]
        R_func_mean = partial(R_fit_func.eval, **R_func_mean_dict)
        R_func_array = [
            partial(R_fit_func.eval, **R_func_array_dict[n])
            for n in range(kwargs["Nboot"])
        ]

        R_mean = np.array(R_func_mean(G3_tlin))
        R_fit = np.array([R_func_array[n](G3_tlin) for n in range(kwargs["Nboot"])])
        R_std = R_fit.std(axis=data_axis)

    return (
        R_data,
        G3_time_fit,
        G3_time_rej,
        G3_tlin,
        R_mean,
        R_std,
        R_fit,
        R_fit_func,
        G3_fit_mean,
        G3_fit_params,
        G3_fit_χ2ν,
    )
