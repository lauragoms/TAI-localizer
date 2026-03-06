import numpy as np
from scipy import linalg as la

from koala import graph_utils as gu
from tai_localiser.perulizer import (
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
    s_gap_list = np.zeros(disorder_average)

    if unitary is not None:
        scaling = (lattice.n_vertices * 4) // trs_operator.shape[0]
        trs_operator = np.kron(np.eye(scaling), trs_operator)
        trs_operator = unitary @ trs_operator @ unitary.conj().T

    rng = np.random.default_rng()

    for j in range(disorder_average):
        # unpack parameters
        ws = (rng.random(lattice.n_vertices) * 2 - 1) * w / 2
        wp = (rng.random(lattice.n_vertices) * 2 - 1) * w / 2

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

        b_out, spin_gap = spin_chern_marker(lattice, full_proj, fix=True, return_spin_gap=True)

        gap_out = np.min(np.abs(e))
        b_list[j] = b_out
        g_list[j] = gap_out
        s_gap_list[j] = spin_gap

        # find spectral localiser
        spec_loc = z2_spec_loc(l_open, hamiltonian_open, 0, trs_operator)
        s_list[j] = 0.5 * (1 - spec_loc)

    bott = np.mean(b_list)
    gap = np.mean(g_list)
    spec = np.mean(s_list)
    s_gap = np.mean(spin_gap)
    # b_std = np.std(b_list)
    # g_std = np.std(g_list)
    # s_std = np.std(s_list)

 
    return bott + spec, bott, gap, spec, spin_gap