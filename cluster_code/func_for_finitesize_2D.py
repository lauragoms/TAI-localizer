import gc
import numpy as np
import scipy.sparse as sp
from koala import pointsets
from tai_localiser.perulizer import (
    proximity_bonds,
    randomly_rotate,
    sigma_y,
)

from tai_localiser.lauralizer.amorphous_model_BHZ_2D import amorph_BHZ
from tai_localiser.lauralizer.localizer import (
    spectral_localizer_AII2D,
    pfaff_sign,
)


def param_obs_2b(
    points: np.ndarray,
    system_size: int,
    sigma: float,
    kappa_shift: float,
    bond_distance: float,
    A: float,
    B: float,
    Delta: float,
    onsite_disorder: float,
    seed: float,
    hadamard_disorder: float = 0,
    kappa_spec=1,
    beta=5,
    bond_power=1,
    bond_lengthscale=.1,
) -> tuple:
    """
    Computes the Z2 invariant for one disorder realisation.

    Points are moved incrementally (random walk in configuration space),
    so the caller must pass in the current points and use the returned
    points for the next call.

    Returns
    -------
    inv_localizer : float
        Pfaffian sign (±1).
    points : np.ndarray
        Updated point positions after this step.
    """

    rng = np.random.default_rng(int(seed))
    points = pointsets.move_all_points(
        points, sigma, kappa_shift, beta, rng=rng
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

    # add a rotation, note that you also have to rotate
    # the time reversal operator
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
    del loc_rotated
    gc.collect()

    return inv_localizer, points
