from koala import pointsets
import numpy as np


from tai_localizer.lauralizer.amorphous_model_BHZ_2D import amorph_BHZ
from tai_localizer.lauralizer.localizer import (
    spectral_localizer_AII2D,
    local_gap_localizer,
    # pfaff_sign
)
from tai_localizer.perulizer import (
    proximity_bonds
)


def localgap_sys(
    system_size: int,
    Delta: float,
    A: float,
    B: float,
    bond_power: float,
    onsite_disorder: float,
    E0: float,
    kappa: float,
    rng: np.random.Generator = np.random.default_rng(),
):
    # Parameters Hamiltonian

    bond_distance = 1.1 / system_size

    parameters = {
            "norbs": 4,
            "rng_W": rng,
            "Delta": Delta,
            "A": A,
            "B": B,
            "dis_onsite": onsite_disorder,
            "mu": 0,
            "bond_lengthscale": 1 / system_size,
            "bond_power": bond_power,
        }

    # Make the points
    points = pointsets.grid(system_size, system_size)

    # Make the lattice
    edges, c = proximity_bonds(points, bond_distance)
    not_crossing = np.abs(c).sum(axis=1) == 0
    edges = edges[not_crossing]

    # Make the system in kwant
    system = amorph_BHZ(points, edges)
    fsyst = system.finalized()

    # Make the Hamiltonian
    ham = fsyst.hamiltonian_submatrix(params=parameters, sparse=True)
    positions = [site.pos for site in fsyst.sites]

    loc = spectral_localizer_AII2D(
        np.array(positions),
        ham,
        E0=E0,
        kappa=kappa,
        time_reversal_operator=None,
        rotated=True
    )

    locgap = np.abs(local_gap_localizer(loc))[0]
    # locgap = pfaff_sign(loc.todense())
    return locgap
