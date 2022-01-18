import numpy as np


def get_simple_ansatz(fit_func, **kwargs):
    if fit_func == "linear":
        return linear
    elif fit_func == "quadratic":
        return quadratic


def constant(x, a0, *args, **kwargs):
    return a0 * np.ones_like(x)


def linear(x, a0, a1, *args, **kwargs):
    return a0 + a1 * x


def quadratic(x, a0, a1, a2, *args, **kwargs):
    return a0 + a1 * x + a2 * x ** 2


def sine(x, A, f):
    return A * np.sin(f * x)

