import numpy as np

startseed = 8


class BootstrapClass:
    def __init__(self, data, nboot, cfg_axis=0):
        self.boot_array = bootstrap(data, nboot, cfg_axis, 0)
        self.avg = None
        self.std = None

    def stats(self):
        self.avg = np.mean(self.boot_array, axis=0)
        self.std = np.std(self.boot_array, axis=0)

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):

        ...


def bootstrap(array, nboot, cfg_axis=0, boot_axis=0):
    array = np.moveaxis(array, cfg_axis, 0)
    length = np.shape(array)[0]
    result = np.empty([nboot, *np.shape(array)[1:]], dtype=array.dtype)
    result[0] = np.mean(array, axis=0)
    myseed = int(startseed * nboot)
    np.random.seed(myseed)

    for iboot in range(1, nboot):
        rint = np.random.randint(0, length, size=length)
        result[iboot] = array[rint].mean(axis=0)
    result = np.moveaxis(result, 0, boot_axis)
    return result
