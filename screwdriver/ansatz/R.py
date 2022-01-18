from .G2 import *
from .G3 import *


def R_get_fit_func(G2_fit_func, G3_fit_func, **kwargs):
    if G2_fit_func == "1exp" and G3_fit_func == "1exp":
        return R_1exp_1exp(**kwargs)
    elif G2_fit_func == "1exp_centered" and G3_fit_func == "1exp_centered":
        return R_1exp_centered_1exp_centered(**kwargs)
    elif G2_fit_func == "2exp" and G3_fit_func == "2exp_tsym":
        return R_2exp_2exp_tsym(**kwargs)
    else:
        raise ValueError(f"R fit func '{G2_fit_func}-{G3_fit_func} not implemented")


def R_1exp_1exp_func(x, B00, *args, **kwargs):
    τ = x[:, 0]
    result = 0 * τ + B00
    return result


class R_1exp_1exp:
    def __init__(self, **kwargs):
        self.eval = R_1exp_1exp_func
        self.npar = 1
        self.G3_args = ["B00"]
        self.G2_args = []


def R_1exp_centered_1exp_centered_func(x, B00, *args, **kwargs):
    τ = x[:, 0]
    result = 0 * τ + B00
    return result


class R_1exp_centered_1exp_centered:
    def __init__(self, **kwargs):
        self.eval = R_1exp_centered_1exp_centered_func
        self.npar = 1
        self.G3_args = ["B00"]
        self.G2_args = []


def R_2exp_2exp_tsym_func(x, B00, B01, B11, A0, m0, A1, m1, *args, **kwargs):
    return G3_2exp_func_tsym(
        x, B00, B01, B11, A0, m0, A1, m1, *args, **kwargs
    ) / G2_2exp_func(x[:, 1], A0, m0, A1, m1, *args, **kwargs)


class R_2exp_2exp_tsym:
    def __init__(self, **kwargs):
        self.eval = R_2exp_2exp_tsym_func
        self.npar = 7
        self.G3_args = ["B00", "B01", "B11"]
        self.G2_args = ["A0", "m0", "A1", "m1"]
