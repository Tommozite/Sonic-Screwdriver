import numpy as np

from ..lattice import effective_mass
from .main import filter_main, ensemble_fit


def fit_G2_effective_mass(
    t, G2, fitfunc, guess, eff_mass_a, G2_tlim, data_axis=0, **kwargs
):
    Nt = len(t)
    num_data = G2.shape[data_axis]
    tlin = np.linspace(min(t), max(t) + 1, 4 * (len(t)), endpoint=False)
    tfit, datafit, _, _ = filter_main(t, G2, G2_tlim)

    fit_params, fit_χ2ν, dof = ensemble_fit(fitfunc, tfit, datafit, guess)

    fit = np.array([fitfunc(tlin, *fit_params[n]) for n in range(num_data)])

    effmass_data = effective_mass(t, G2, eff_mass_a, Nt)
    effmass_fit = effective_mass(tlin, fit, eff_mass_a, Nt)

    tfit, _, trej, _ = filter_main(t, effmass_data, G2_tlim)
    tlin_filtered, effmass_fit_filtered, _, _, mask = filter_main(
        tlin, effmass_fit, G2_tlim, return_mask=True
    )

    effmass_mean_filtered = effmass_fit_filtered.mean(axis=data_axis)
    effmass_std_filtered = effmass_fit_filtered.std(axis=data_axis)

    return (
        effmass_data,
        tfit,
        trej,
        tlin_filtered,
        effmass_mean_filtered,
        effmass_std_filtered,
        effmass_fit_filtered,
        fit_params,
        fit_χ2ν,
    )
