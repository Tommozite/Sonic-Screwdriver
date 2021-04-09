import numpy as np


def eff_mass(G2, δt=2, res=1, t_axis=1):
    """
    Calculates the effective mass from in input 2-point correlator.
    res is the number of points between each integer time step,
    δt is the number of integer time steps to take the ratio across.
    """
    eff_mass = np.apply_along_axis(
        lambda x: (1 / δt) * np.log(np.abs(x[: -(res * δt)] / x[(res * δt) :])),
        axis=t_axis,
        array=G2,
    )

    return eff_mass
