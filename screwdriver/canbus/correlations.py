import numpy as np

from .capture import Capture


def pearson(X, Y):
    cov = np.mean((X - np.mean(X, axis=0)) * (Y - np.mean(Y, axis=0)), axis=0)
    std_X = np.std(X, axis=0) + 1e-6
    std_Y = np.std(Y, axis=0) + 1e-6
    coeff = cov / (std_X * std_Y)
    return coeff


def point_biserial(X, Y):
    M1 = np.mean(X[Y], axis=0)
    M0 = np.mean(X[~Y], axis=0)
    sn = np.std(X)
    n = len(Y)
    n1 = np.sum(Y)
    n0 = n - n1
    coeff = ((M1 - M0) / sn) * np.sqrt(n1 * n0 / n**2)
    return coeff


def smc(X, Y):
    M11 = np.sum(np.logical_and(X, Y))
    M01 = np.sum(np.logical_and(~X, Y))
    M10 = np.sum(np.logical_and(X, ~Y))
    M00 = np.sum(np.logical_and(~X, ~Y))
    coeff = (M00 + M11) / (M00 + M10 + M01 + M11)
    return coeff
