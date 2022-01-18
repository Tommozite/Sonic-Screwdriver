import numpy as np


def χ2(params, func, xdata, ydata):

    cov_mat = np.cov(ydata, rowvar=False)
    cov_mat_inv = np.linalg.pinv(cov_mat)
    res = ydata.mean(axis=0) - func(xdata, *params)
    result = np.einsum("i,ij,j", res, cov_mat_inv, res)
    return result


def χ2_weighted(params, func, xdata, ydata, weight_matrix):
    res = ydata - func(xdata, *params)
    return np.einsum("i,ij,j", res, weight_matrix, res)
