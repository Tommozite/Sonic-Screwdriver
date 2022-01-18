import numpy as np
from functools import partial


def G3_get_fit_func(G3_fit_func, **kwargs):
    if G3_fit_func == "1exp":
        return G3_1exp(**kwargs)
    elif G3_fit_func == "1exp_centered":
        return G3_1exp_centered(**kwargs)
    elif G3_fit_func == "2exp":
        return G3_2exp(**kwargs)
    elif G3_fit_func == "2exp_tsym":
        return G3_2exp_tsym(**kwargs)
    else:
        raise ValueError(f"G3 fit func '{G3_fit_func}' not implemented")


def G3_1exp_func(x, B00, A0, m0, *args, **kwargs):
    t = x[:, 1]
    result = A0 * B00 * np.exp(-m0 * t)
    return result


def G3_1exp_guess(G3_time, G3_data, G2_time, G2_data, **kwargs):
    return [G3_data[0, 0] / G2_data[0, G3_time[0, 1]]]


class G3_1exp:
    def __init__(self, **kwargs):

        self.eval = G3_1exp_func
        self.npar = 1
        self.args = ["B00"]
        self.G2_args = ["A0", "m0"]
        self.guess = G3_1exp_guess


def G3_1exp_func_centered_template(x, B00, Nt, A0, m0, *args, **kwargs):
    t = x[:, 1]
    result = A0 * B00 * (np.exp(-m0 * t) + np.exp(-m0 * (Nt - t)))
    return result


def G3_1exp_func_centered(Nt):
    return partial(G3_1exp_func_centered_template, Nt=Nt)


def G3_1exp_guess_centered(G3_time, G3_data, G2_time, G2_data, **kwargs):
    return [G3_data[0, 0] / G2_data[0, G3_time[0, 1]]]


class G3_1exp_centered:
    def __init__(self, **kwargs):

        self.eval = G3_1exp_func_centered(kwargs["Nt"])
        self.npar = 1
        self.args = ["B00"]
        self.G2_args = ["A0", "m0"]
        self.guess = G3_1exp_guess_centered


def G3_2exp_func(x, B00, B01, B10, B11, A0, m0, A1, m1, *args, **kwargs):
    τ = x[:, 0]
    t = x[:, 1]
    result = (
        A0 * B00 * np.exp(-m0 * t)
        + A1 * B11 * np.exp(-m1 * t)
        + np.sqrt(A0 * A1) * B01 * np.exp(-m0 * τ) * np.exp(-m1 * (t - τ))
        + np.sqrt(A0 * A1) * B10 * np.exp(-m1 * τ) * np.exp(-m0 * (t - τ))
    )
    return result


def G3_2exp_guess(G3_time, G3_data, G2_time, G2_data, **kwargs):
    return [G3_data[0, 0] / G2_data[0, G3_time[0, 1]], 0, 0, 0]


class G3_2exp:
    def __init__(self, **kwargs):

        self.eval = G3_2exp_func
        self.npar = 4
        self.args = ["B00", "B01", "B10", "B11"]
        self.G2_args = ["A0", "m0", "A1", "m1"]
        self.guess = G3_2exp_guess


def G3_2exp_func_tsym(x, B00, B01, B11, A0, m0, A1, m1, *args, **kwargs):
    τ = x[:, 0]
    t = x[:, 1]
    Δm = m1 - m0
    result = (
        A0
        * np.exp(-m0 * t)
        * (
            B00
            + B01 * (np.exp(-Δm * τ) + np.exp(-Δm * (t - τ)))
            + B11 * np.exp(-Δm * t)
        )
    )
    return result


def G3_2exp_guess_tsym(G3_time, G3_data, G2_time, G2_data, **kwargs):
    return [
        G3_data[0, 0] / G2_data[0, G3_time[0, 1]],
        G3_data[0, 0] / G2_data[0, G3_time[0, 1]] / 100,
        G3_data[0, 0] / G2_data[0, G3_time[0, 1]] / 100,
    ]


class G3_2exp_tsym:
    def __init__(self, **kwargs):

        self.eval = G3_2exp_func_tsym
        self.npar = 3
        self.args = ["B00", "B01", "B11"]
        self.G2_args = ["A0", "m0", "A1", "m1"]
        self.guess = G3_2exp_guess_tsym

