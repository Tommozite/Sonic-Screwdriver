import numpy as np
from scipy.linalg import eig

from . import misc_functions as mf


def momentum_lattice(nμ, Ns, Nt, Nd, **params):
    pμ = 2 * np.pi * nμ * np.array([1 / Ns] * (Nd - 1) + [1 / Nt])

    p2 = np.einsum("pm->p", pμ**2)
    return pμ, p2


def momentum_lattice_sine_fermion(nμ, Ns, Nt, Nd, **params):
    pμ = 2 * np.pi * nμ * np.array([1 / Ns] * (Nd - 1) + [1 / Nt])

    sinpμ = np.sin(pμ)
    sinp2 = np.einsum("pm->p", sinpμ**2)
    return sinpμ, sinp2


def momentum_lattice_sine_boson(nμ, Ns, Nt, Nd, **params):
    pμ = 2 * np.pi * nμ * np.array([1 / Ns] * (Nd - 1) + [1 / Nt])
    sinpμ = 2 * np.sin(pμ / 2)
    sinp2 = np.einsum("pm->p", sinpμ**2)
    return sinpμ, sinp2


def effective_mass(t, G2, eff_mass_a, Nt, time_axis=1, **kwargs):
    t_adv = mf.wherein((t + eff_mass_a) % Nt, t)
    if np.any(t_adv.mask):
        raise ValueError(
            "Correlator must be defined at 't+a % Nt' for every point in 't'"
        )
    result = (1 / eff_mass_a) * np.log(np.abs(G2 / G2.take(t_adv, axis=time_axis)))
    return result


def cosh_effective_mass(t, G2, eff_mass_a, Nt, time_axis=1, **kwargs):
    t_adv = mf.wherein((t + eff_mass_a) % Nt, t)
    t_ret = mf.wherein((t - eff_mass_a) % Nt, t)
    if np.any(t_adv.mask):
        raise ValueError(
            "Correlator must be defined at 't+a % Nt' and 't-a % Nt' for every point in 't'"
        )
    ratio = (G2.take(t_adv, axis=time_axis) + G2.take(t_ret, axis=time_axis)) / (2 * G2)
    ratio = np.max(np.array([ratio, 1 / ratio]), axis=0)
    result = (1 / eff_mass_a) * np.arccosh(np.abs(ratio))
    return result


def G2_variational_pencil_of_function_matrix(t, G2, δ, Nt, time_axis=1, **kwargs):
    t_adv_1δ = mf.wherein((t + δ) % Nt, t)
    t_adv_2δ = mf.wherein((t + 2 * δ) % Nt, t)

    result = np.array(
        [
            [G2, G2.take(t_adv_1δ, axis=time_axis)],
            [G2.take(t_adv_1δ, axis=time_axis), G2.take(t_adv_2δ, axis=time_axis)],
        ]
    )
    result = np.einsum("ijnt->ntij", result)
    return result


def G3_variational_pencil_of_function_matrix(
    x, G3, sink_times, δ, Nt, time_axis=1, **kwargs
):
    G3_sep = [G3[:, x[:, 1] == t] for t in sink_times]
    τ = [x[x[:, 1] == t, 0] for t in sink_times][0]

    τ_adv = mf.wherein((τ + δ) % Nt, τ)

    result = np.array(
        [
            [G3_sep[0], G3_sep[1]],
            [
                G3_sep[1].take(τ_adv, axis=time_axis),
                G3_sep[2].take(τ_adv, axis=time_axis),
            ],
        ]
    )
    result = np.einsum("ijnt->ntij", result)
    return result


def variational_pencil_of_function(
    G2, G2_time, G3, G3_time, δ, t0, Δ, Nt, t_sink, time_axis=1, **kwargs
):
    G2_matrix = G2_variational_pencil_of_function_matrix(G2_time, G2, δ, Nt, **kwargs)
    G3_matrix = G3_variational_pencil_of_function_matrix(
        G3_time, G3, t_sink, δ, Nt, **kwargs
    )
    G2_inv = np.linalg.inv(G2_matrix[0, t0])
    wv, v = eig(G2_matrix[0, t0 + Δ] @ G2_inv, left=True, right=False)
    wu, u = eig(G2_inv @ G2_matrix[0, t0 + Δ], left=False, right=True)
    G2_rot = [np.einsum("i,ntij,j->nt", v[:, β], G2_matrix, u[:, β]) for β in range(2)]
    G3_rot = [np.einsum("i,ntij,j->nt", v[:, β], G3_matrix, u[:, β]) for β in range(2)]
    return G2_rot, G3_rot
