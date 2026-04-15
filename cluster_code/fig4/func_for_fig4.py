import numpy as np
import scipy.sparse as sp

from koala import pointsets
from koala import graph_utils as gu


from tai_localiser.perulizer import (
    proximity_bonds,
    randomly_rotate,
    sigma_y,
    proximity_lattice,
    bhz_ham,
    bhz_trs_operator,
    z2_spec_loc,
)

from tai_localiser.lauralizer.amorphous_model_BHZ_2D import amorph_BHZ
from tai_localiser.lauralizer.localizer import (
    spectral_localizer_AII2D,
    pfaff_sign,
)


def param_obs_2d_benchmark_peru(
    system_size: int,
    sigma: float,
    bond_distance: float,
    A: float,
    B: float,
    Delta: float,
    onsite_disorder: float,
    alpha: float = 0,
    hadamard_disorder: float = 0,
    kappa=1,
    disorder_average=1,
    beta=5,
    **kwargs
) -> tuple:

    # make the points
    rng = np.random.default_rng()
    points = pointsets.grid(system_size, system_size)

    s_list = np.zeros(disorder_average)
    for j in range(disorder_average):
        points = pointsets.move_all_points(points, sigma, sigma, beta)
        lattice = proximity_lattice(points, bond_distance)
        lattice = gu.cut_boundaries(lattice)

        ws = (rng.random(lattice.n_vertices) * 2 - 1) * onsite_disorder / 2
        wp = (rng.random(lattice.n_vertices) * 2 - 1) * onsite_disorder / 2
        ham_params = (
            A,
            B,
            alpha,
            Delta,
            ws,
            wp,
        )
        hamiltonian = bhz_ham(lattice, *ham_params, **kwargs)
        spec_loc = z2_spec_loc(lattice, hamiltonian, 0, bhz_trs_operator)
        s_list[j] = spec_loc
    return np.average(s_list)


def param_obs_2b(
    system_size: int,
    sigma: float,
    kappa_shift: float,
    bond_distance: float,
    A: float,
    B: float,
    Delta: float,
    onsite_disorder: float,
    hadamard_disorder: float = 0,
    kappa_spec=1,
    disorder_average=1,
    beta=5,
    bond_power=1,
    bond_lengthscale=.1,
) -> tuple:

    # make the points
    init_points = pointsets.grid(system_size, system_size)

    loc_out = np.zeros(disorder_average)
    # ptslists = []
    for j in range(disorder_average):
        rng = np.random.default_rng(int(j))
        points = pointsets.move_all_points(
            init_points, sigma, kappa_shift, beta, rng=rng
            )

        # make the lattice
        edges, c = proximity_bonds(points, bond_distance)
        not_crossing = np.abs(c).sum(axis=1) == 0
        edges = edges[not_crossing]  # get rid of boundary crossing bonds

        # hamiltonian parameters
        parameters = {
            "norbs": 4,
            "rng_W": rng,
            "Delta": Delta,
            "A": A,
            "B": B,
            "dis_onsite": onsite_disorder,
            "mu": 0,
            "bond_lengthscale": bond_lengthscale,
            "bond_power": bond_power,
        }
        # make the system in kwant
        system = amorph_BHZ(points, edges)
        fsyst = system.finalized()
        ham = fsyst.hamiltonian_submatrix(params=parameters, sparse=True)
        positions = [site.pos for site in fsyst.sites]

        # add a rotation, note that you also have to rotate the time reversal operator
        if hadamard_disorder > 0:
            random_unitary = randomly_rotate(
                len(positions), hadamard_disorder, sparse=True
            )
            ham = random_unitary.conj().T @ ham @ random_unitary
            trs_operator = sp.kron(sp.eye(len(positions) * 2), sigma_y)
            trs_operator = random_unitary @ trs_operator @ random_unitary.conj().T
        else:
            trs_operator = None


        # compute localizer
        loc_rotated = spectral_localizer_AII2D(
            np.array(positions),
            ham,
            E0=0,
            kappa=kappa_spec,
            time_reversal_operator=trs_operator,
        )

        inv_localizer = pfaff_sign(loc_rotated.todense())

        loc_out[j] = inv_localizer
        # ptslists.append(positions)
    return np.mean(loc_out)  # w, ptslists
