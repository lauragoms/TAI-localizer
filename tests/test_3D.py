import numpy as np
import matplotlib.pyplot as plt

from tai_localiser.lauralizer.amorphous_model_3D import (
    amorph_hopping,
    amorph_3DTI
    )
from tai_localiser.lauralizer.functions import bonds_func
from tai_localiser.lauralizer.localizer import (
    spectral_localizer_AII3D,
    sign_det
    )


def grid_3D(nx: int, ny: int, nz: int) -> np.ndarray:
    """Generates a uniformly spaced grid of points in 3D

    Args:
        nx (int): Number of points in x direction
        ny (int): Number of points in y direction
        nz (int): Number of points in z direction

    Returns:
        np.ndarray: List of all the positions.
    """

    pos_x = np.linspace(0, 1, nx, endpoint=False)
    pos_y = np.linspace(0, 1, ny, endpoint=False)
    pos_z = np.linspace(0, 1, nz, endpoint=False)
    pos_x += 0.5*pos_x[0]
    pos_y += 0.5*pos_y[0]
    pos_z += 0.5*pos_z[0]
    g_out = np.reshape(np.meshgrid(pos_x, pos_y, pos_z), [3, -1]).T
    return g_out


system_size = 5
bond_distance = 1.3 / system_size
sites = grid_3D(system_size, system_size, system_size)
bonds = bonds_func(sites, bond_distance)
syst = amorph_3DTI(sites, bonds)
sys_sites = syst.finalized().sites
positions = [site.pos for site in sys_sites]


def test_hoppings():
    site1 = sys_sites[0]
    site2 = sys_sites[1]

    lambdaJ = 1
    bond_lengthscale = 1.0 / 3
    bond_power = 1.0
    t = amorph_hopping(site1, site2, lambdaJ, bond_lengthscale, bond_power)
    t2 = amorph_hopping(site2, site1, lambdaJ, bond_lengthscale, bond_power)

    assert np.allclose(t.conj().T, t2), "Hopping is not Hermitian!"


def test_topology():

    rng = np.random.default_rng()
    new_params = {
        'MJ': 2.3,
        'A': 1.0,
        'bond_lengthscale': 1/system_size,
        'bond_power': 1/system_size,
        'dis_onsite': 0.0,
        'rng_W': rng,
    }

    ham = syst.finalized().hamiltonian_submatrix(
        params=new_params,
        sparse=True
        )
    M_linspace = np.linspace(-4, 4, 50)
    z2_M = []
    for M in M_linspace:
        new_params.update({'MJ': M})
        ham = syst.finalized().hamiltonian_submatrix(
            params=new_params,
            sparse=True
            )

        L = spectral_localizer_AII3D(
            np.array(positions),
            ham,
            0,
            kappa=2,
        )
        z2_M.append(sign_det(L))
    plt.plot(M_linspace, z2_M, '-o')


# test_hoppings()
test_topology()
