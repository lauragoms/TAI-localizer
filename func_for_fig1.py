import numpy as np
from scipy import linalg as la

from koala import graph_utils as gu
from tai_localizer.perulizer import (
    bhz_ham,
    z2_spec_loc,
    bhz_trs_operator,
    spin_chern_marker,
)


def param_to_observables(
    lattice,
    A,
    B,
    delta,
    alpha,
    w,
    disorder_average=1,
    unitary=None,
    trs_operator=bhz_trs_operator,
):

    b_list = np.zeros(disorder_average)
    g_list = np.zeros(disorder_average)
    s_list = np.zeros(disorder_average)
    if unitary is not None:
        scaling = (lattice.n_vertices * 4) // trs_operator.shape[0]
        trs_operator = np.kron(np.eye(scaling), trs_operator)
        trs_operator = unitary @ trs_operator @ unitary.conj().T

    for j in range(disorder_average):
        # unpack parameters
        ws = (np.random.rand(lattice.n_vertices) * 2 - 1) * w / 2
        wp = (np.random.rand(lattice.n_vertices) * 2 - 1) * w / 2

        l_open = gu.cut_boundaries(lattice)

        ham_params = (
            A,
            B,
            alpha,
            delta,
            ws,
            wp,
        )

        hamiltonian = bhz_ham(lattice, *ham_params)
        hamiltonian_open = bhz_ham(l_open, *ham_params)

        if unitary is not None:
            hamiltonian = unitary @ hamiltonian @ unitary.conj().T
            hamiltonian_open = unitary @ hamiltonian_open @ unitary.conj().T

        # find gap and bott index
        e, v = la.eigh(hamiltonian)
        full_proj = v @ np.diag(e < 0) @ v.conj().T

        b_out = spin_chern_marker(lattice, full_proj, fix=True)

        gap_out = np.min(np.abs(e))
        b_list[j] = b_out
        g_list[j] = gap_out

        # find spectral localiser
        spec_loc = z2_spec_loc(l_open, hamiltonian_open, 0, trs_operator)
        s_list[j] = 0.5 * (1 - spec_loc)

    b = np.mean(b_list)
    g = np.mean(g_list)
    s = np.mean(s_list)
    return b + g + s, b, g, s


# def param_obs_2b(sig, Delta, dis_onsite, system_size, A, B, dis_hadamard):

#     # for seed in seed_range? average over disorder?
#     rng = np.random.default_rng()

#     par_dict = {
#         "norbs": 4,
#         "rng_hdmd": rng,
#         "Delta": Delta,
#         "dis_hadamard": dis_hadamard * 100,
#         "mu": 0,
#         "A": A,
#         "B": B,
#         "rng_W": rng,
#         "dis_onsite": dis_onsite,
#     }

#     # create koala lattice with a diff sigma
#     L = system_size
#     radius = np.sqrt(2) / L - 0.0001
#     sigma = sig * 1 / L

#     points = pointsets.grid(L, L)
#     points = pointsets.move_all_points(points, sigma, sigma, 8)

#     e, c = proximity_bonds(points, radius)
#     not_crossing = np.abs(c).sum(axis=1) == 0

#     # create kwant system
#     system = amorph_BHZ(
#         L * points, e[not_crossing]
#     )  # if you change L, you should change kappa ~ 1/L
#     fsyst = system.finalized()
#     ham = fsyst.hamiltonian_submatrix(params=par_dict, sparse=True)
#     positions = [site.pos for site in fsyst.sites]

#     # compute localizer
#     loc_rotated = spectral_localizer_AII2D(np.array(positions), ham, E0=0, kappa=0.1)
#     inv_localizer = pfaff_sign(loc_rotated.todense())

#     return inv_localizer
