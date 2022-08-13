import numpy as np
from functools import partial
from ..lattice import effective_mass


def G2_fh_get_fit_func(G2_fh_fit_func, **kwargs):
    if G2_fh_fit_func == "1exp_ratio_fh":
        return G2_fh_1exp_ratio(**kwargs)
    elif G2_fh_fit_func == "1exp_centered_ratio_fh":
        return G2_fh_1exp_centered_ratio(**kwargs)
    elif G2_fh_fit_func == "1exp_fh":
        return G2_fh_1exp(**kwargs)
    elif G2_fh_fit_func == "2exp_fh":
        return G2_fh_2exp(**kwargs)
    elif G2_fh_fit_func == "1exp_centered_fh":
        return G2_fh_1exp_centered(**kwargs)
    else:
        raise ValueError(f"G2 fh fit func '{G2_fh_fit_func}' not implemented")


def G2_fh_1exp_func_centered_ratio_template(
    t, A, Δm, A0, m0, Nt, odd=False, *args, **kwargs
):
    return (
        A
        * (np.exp(-(m0 + Δm) * t) + np.exp(-(m0 + Δm * (-1) ** (int(odd))) * (Nt - t)))
        / (np.exp(-(m0) * t) + np.exp(-(m0) * (Nt - t)))
    )


def G2_fh_1exp_func_centered_ratio(Nt, odd=False):
    return partial(G2_fh_1exp_func_centered_ratio_template, Nt=Nt, odd=odd)


def G2_fh_1exp_guess_centered_ratio(G2_time, G2_data, **kwargs):
    return [
        G2_data[0, 0],
        effective_mass(G2_time, G2_data, kwargs["eff_mass_a"], kwargs["Nt"])[
            0, kwargs["Nt"] // 4
        ],
    ]


class G2_fh_1exp_centered_ratio:
    def __init__(self, odd=False, **kwargs):

        self.eval = G2_fh_1exp_func_centered_ratio(kwargs["Nt"], odd)
        self.npar = 2
        self.λ_args = ["A", "Δm"]
        self.G2_args = ["A0", "m0"]
        self.guess = G2_fh_1exp_guess_centered_ratio


def G2_fh_1exp_func_ratio(t, A, Δm, A0, m0, *args, **kwargs):
    return A * np.exp(-Δm * t)


def G2_fh_1exp_guess_ratio(G2_time, G2_data, **kwargs):
    return [
        G2_data[0, 0],
        effective_mass(G2_time, G2_data, kwargs["eff_mass_a"], kwargs["Nt"])[
            0, kwargs["Nt"] // 4
        ],
    ]


class G2_fh_1exp_ratio:
    def __init__(self, **kwargs):

        self.eval = G2_fh_1exp_func_ratio
        self.npar = 2
        self.λ_args = ["A", "Δm"]
        self.G2_args = ["A0", "m0"]
        self.guess = G2_fh_1exp_guess_ratio


def G2_fh_1exp_func(t, A, mλ, A0, m0, *args, **kwargs):
    return A * np.exp(-mλ * t)


def G2_fh_1exp_guess(G2_time, G2_data, **kwargs):
    return [
        G2_data[0, 0],
        effective_mass(G2_time, G2_data, kwargs["eff_mass_a"], kwargs["Nt"])[
            0, kwargs["Nt"] // 4
        ],
    ]


class G2_fh_1exp:
    def __init__(self, **kwargs):

        self.eval = G2_fh_1exp_func
        self.npar = 2
        self.λ_args = ["A", "mλ"]
        self.G2_args = ["A0", "m0"]
        self.guess = G2_fh_1exp_guess


def G2_fh_1exp_func_centered_template(t, A, mλ, A0, m0, Nt, *args, **kwargs):
    return A * (np.exp(-mλ * t) + np.exp(-mλ * (Nt - t)))


def G2_fh_1exp_func_centered(Nt):
    return partial(G2_fh_1exp_func_centered_template, Nt=Nt)


def G2_fh_1exp_guess_centered(G2_time, G2_data, **kwargs):
    return [
        G2_data[0, 0],
        effective_mass(G2_time, G2_data, kwargs["eff_mass_a"], kwargs["Nt"])[
            0, kwargs["Nt"] // 4
        ],
    ]


class G2_fh_1exp_centered:
    def __init__(self, **kwargs):

        self.eval = G2_fh_1exp_func_centered(kwargs["Nt"])
        self.npar = 2
        self.λ_args = ["A", "mλ"]
        self.G2_args = ["A0", "m0"]
        self.guess = G2_fh_1exp_guess_centered


def G2_fh_2exp_func(t, A, mλ, Ap, mλp, A0, m0, *args, **kwargs):
    return A * np.exp(-mλ * t) + Ap * np.exp(-mλp * t)


def G2_fh_2exp_guess(G2_time, G2_data, **kwargs):
    return [
        G2_data[0, 0],
        effective_mass(G2_time, G2_data, kwargs["eff_mass_a"], kwargs["Nt"])[
            0,
            kwargs["Nt"] // 4,
        ],
        G2_data[0, 0],
        effective_mass(G2_time, G2_data, kwargs["eff_mass_a"], kwargs["Nt"])[
            0,
            kwargs["Nt"] // 4,
        ]
        + 1,
    ]


class G2_fh_2exp:
    def __init__(self, **kwargs):

        self.eval = G2_fh_2exp_func
        self.npar = 2
        self.λ_args = ["A", "mλ", "Ap", "mλp"]
        self.G2_args = ["A0", "m0"]
        self.guess = G2_fh_2exp_guess
