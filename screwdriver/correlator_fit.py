import numpy as np


def G2_1exp_func(x, A0, m0):
    return A0 * np.exp(-m0 * x)


def G2_2exp_func(x, A0, m0, A1, m1):
    return A0 * np.exp(-m0 * x) + A1 * np.exp(-m1 * x)


def G2_1exp_func_middle(x, A0, m0):
    return A0 * (np.exp(-m0 * x) + np.exp(-m0 * (48 - x)))


def G3_1exp_func(x, B00, A0, m0):
    t = x[1]
    result = A0 * B00 * np.exp(-m0 * t)
    return result


def G3_1exp_func_middle(x, B00, A0, m0):
    t = x[1]
    result = A0 * B00 * (np.exp(-m0 * t) + np.exp(-m0 * (48 - t)))
    return result


def G3_2exp_func(x, B00, B10, B11, A0, m0, A1, m1):
    Δm = m1 - m0
    τ = x[0] + x[1] / 2
    t = x[1]
    result = (
        A0
        * np.exp(-m0 * t)
        * (
            B00
            + B10 * (np.exp(-Δm * τ) + np.exp(-Δm * (t - τ)))
            + B11 * np.exp(-Δm * t)
        )
    )
    return result
