import numpy as np
from functools import partial
from ..lattice import effective_mass


def G2_get_fit_func(G2_fit_func, **kwargs):
    if G2_fit_func == "1exp":
        return G2_1exp(**kwargs)
    elif G2_fit_func == "1exp_centered":
        return G2_1exp_centered(**kwargs)
    elif G2_fit_func == "1exp_centered_fh":
        return G2_1exp_centered_fh(**kwargs)
    elif G2_fit_func == "2exp":
        return G2_2exp(**kwargs)
    else:
        raise ValueError(f"G2 fit func '{G2_fit_func}' not implemented")


def G2_1exp_func(t, A0, m0, *args, **kwargs):
    return A0 * np.exp(-m0 * t)


def G2_1exp_guess(G2_time, G2_data, **kwargs):
    return [
        G2_data[0, 0],
        effective_mass(G2_time, G2_data, kwargs["eff_mass_a"], kwargs["Nt"])[
            0, kwargs["Nt"] // 4
        ],
    ]


class G2_1exp:
    def __init__(self, **kwargs):

        self.eval = G2_1exp_func
        self.npar = 2
        self.args = ["A0", "m0"]
        self.guess = G2_1exp_guess


def G2_1exp_func_centered_template(t, A0, m0, Nt, *args, **kwargs):
    return A0 * (np.exp(-m0 * t) + np.exp(-m0 * (Nt - t)))


def G2_1exp_func_centered(Nt):
    return partial(G2_1exp_func_centered_template, Nt=Nt)


def G2_1exp_guess_centered(G2_time, G2_data, **kwargs):
    return [
        G2_data[0, 0],
        effective_mass(G2_time, G2_data, kwargs["eff_mass_a"], kwargs["Nt"])[
            0, kwargs["Nt"] // 4
        ],
    ]


class G2_1exp_centered:
    def __init__(self, **kwargs):

        self.eval = G2_1exp_func_centered(kwargs["Nt"])
        self.npar = 2
        self.args = ["A0", "m0"]
        self.guess = G2_1exp_guess_centered


def G2_2exp_func(t, A0, m0, A1, m1, *args, **kwargs):
    return A0 * np.exp(-m0 * t) + A1 * np.exp(-m1 * t)


def G2_2exp_guess(G2_time, G2_data, **kwargs):
    return [
        G2_data[0, 0],
        effective_mass(G2_time, G2_data, kwargs["eff_mass_a"], kwargs["Nt"])[
            0, kwargs["Nt"] // 4
        ],
        G2_data[0, 0],
        1
        + effective_mass(G2_time, G2_data, kwargs["eff_mass_a"], kwargs["Nt"])[
            0, kwargs["Nt"] // 4
        ],
    ]


class G2_2exp:
    def __init__(self, **kwargs):

        self.eval = G2_2exp_func
        self.npar = 4
        self.args = ["A0", "m0", "A1", "m1"]
        self.guess = G2_2exp_guess

