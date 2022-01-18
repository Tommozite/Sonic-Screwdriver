import numpy as np

startseed = 8


def bootstrap(array, Nboot, cfg_axis=0, boot_axis=0):
    array = np.moveaxis(array, cfg_axis, 0)
    length = np.shape(array)[0]
    result = np.empty([Nboot, *np.shape(array)[1:]], dtype=array.dtype)
    result[0] = np.mean(array, axis=0)  # Use the sample mean as the first bootstrap
    myseed = int(startseed * Nboot)
    np.random.seed(myseed)

    for iboot in range(1, Nboot):
        rint = np.random.randint(0, length, size=length)
        result[iboot] = np.mean(array[rint], axis=0)
    result = np.moveaxis(result, 0, boot_axis)
    return result
