import numpy as np
from ..ansatz import G2, G3, R
from ..ansatz import simple as sa
from functools import partial
from .G3 import filter_G3
from .main import ensemble_fit, ensemble_fit_params_to_func


def fit_R(
    G3_time, G3_data, G2_data, G2_fit_mean, G2_fit_params, guess, G3_tlim, **kwargs
):
    # Get fitting functions
    G2_fit_func = G2.G2_get_fit_func(**kwargs)
    R_fit_func = R.R_get_fit_func(**kwargs)

    R_data = G3_data / G2_data[:, G3_time[:, 1]]

    # Construct the function array
    R_func_mean_dict = {
        k: G2_fit_mean[G2_fit_func.args.index(k)] for k in R_fit_func.G2_args
    }

    R_func_array_dict = [
        {k: G2_fit_params[n, G2_fit_func.args.index(k)] for k in R_fit_func.G2_args}
        for n in range(kwargs["Nboot"])
    ]
    R_func_mean = partial(R_fit_func.eval, **R_func_mean_dict)
    R_func_array = [
        partial(R_fit_func.eval, **R_func_array_dict[n]) for n in range(kwargs["Nboot"])
    ]
    G3_tlin = np.array(
        [
            [τ, t]
            for t in kwargs["t_sink"]
            for τ in np.linspace(G3_tlim[t][0], G3_tlim[t][1])
        ]
    )

    G3_time_fit, R_data_fit, G3_time_rej, R_data_rej = filter_G3(
        G3_time, R_data, G3_tlim
    )

    R_fit_mean, R_fit_params, R_fit_χ2ν, dof = ensemble_fit(
        R_func_mean, R_func_array, G3_time_fit, R_data_fit, guess
    )

    R_mean = np.array(R_func_mean(G3_tlin, *R_fit_mean))
    R_fit = np.array(
        [R_func_array[n](G3_tlin, *R_fit_params[n]) for n in range(kwargs["Nboot"])]
    )
    R_std = R_fit.std(axis=0)

    return (
        R_data,
        G3_time_fit,
        G3_time_rej,
        G3_tlin,
        R_mean,
        R_std,
        R_fit,
        R_fit_func,
        R_fit_mean,
        R_fit_params,
        R_fit_χ2ν,
    )


def fit_R_plateau(G3_time, G3_data, G2_data, guess, G3_tlim, **kwargs):
    # Get fitting functions
    R_fit_func = lambda x, c: sa.constant(x[:, 0], c)

    R_data = G3_data / G2_data[:, G3_time[:, 1]]

    G3_tlin = np.array(
        [
            [τ, t]
            for t in kwargs["t_sink"]
            for τ in np.linspace(G3_tlim[t][0], G3_tlim[t][1])
        ]
    )

    G3_time_fit, R_data_fit, G3_time_rej, R_data_rej = filter_G3(
        G3_time, R_data, G3_tlim
    )

    R_fit_mean, R_fit_params, R_fit_χ2ν, dof = ensemble_fit(
        R_fit_func, G3_time_fit, R_data_fit, guess
    )

    R_mean, R_fit = ensemble_fit_params_to_func(
        R_fit_func, G3_tlin, R_fit_mean, R_fit_params
    )
    R_std = R_fit.std(axis=0)
    return (
        R_data,
        G3_time_fit,
        G3_time_rej,
        G3_tlin,
        R_mean,
        R_std,
        R_fit,
        R_fit_func,
        R_fit_mean,
        R_fit_params,
        R_fit_χ2ν,
    )
