import numpy as np
import scipy.sparse as sp

from koala import pointsets
from tai_localizer.perulizer import proximity_bonds, randomly_rotate, sigma_y

from tai_localizer.lauralizer.amorphous_model_BHZ_2D import amorph_BHZ
from tai_localizer.lauralizer.localizer import (
    spectral_localizer_AII2D,
    pfaff_sign,
)


def param_obs_2b(
    system_size: int,
    sigma: float,
    bond_distance: float,
    A: float,
    B: float,
    Delta: float,
    onsite_disorder: float,
    hadamard_disorder: float = 0,
    kappa=1,
    disorder_average=1,
) -> tuple:

    # make the points
    rng = np.random.default_rng()
    points = pointsets.grid(system_size, system_size)

    loc_out = np.zeros(disorder_average)

    for j in range(disorder_average):
        points = pointsets.move_all_points(points, sigma, sigma, 8)

        # make the lattice
        edges, c = proximity_bonds(points, bond_distance)
        not_crossing = np.abs(c).sum(axis=1) == 0
        edges = edges[not_crossing]  # get rid of boundary crossing bonds

        # hamiltonian parameters
        parameters = {
            "norbs": 4,
            "rng_hdmd": rng,
            "rng_W": rng,
            "Delta": Delta,
            "A": A,
            "B": B,
            "dis_hadamard": 0,
            "dis_onsite": onsite_disorder,
            "mu": 0,
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
            trs_operator =  sp.kron(sp.eye(len(positions)*2), sigma_y)
            trs_operator = random_unitary @ trs_operator @ random_unitary.conj().T
        else:
            trs_operator = None


        # compute localizer
        loc_rotated = spectral_localizer_AII2D(
            np.array(positions), ham, E0=0, kappa=kappa, time_reversal_operator=trs_operator)
        
        inv_localizer = pfaff_sign(loc_rotated.todense())

        loc_out[j] = inv_localizer
    return np.mean(loc_out)
