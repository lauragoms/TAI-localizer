import numpy as np
from tai_localizer.lauralizer.amorphous_model_3D import (
    amorph_3DTI
    )
from tai_localizer.lauralizer.functions import bonds_func
from tai_localizer.lauralizer.localizer import (
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

# sites = grid_3D(system_size, system_size, system_size)


def params_obs_3D(
    MJ: float,
    A: float,
    onsite_disorder: float,
    seed: float,
    sites,
    kappa,
    E0,
    bond_distance: float,
    bond_power: float,
    bond_lengthscale: float,
):

    bonds = bonds_func(sites, bond_distance)
    syst = amorph_3DTI(sites, bonds)
    sys_sites = syst.finalized().sites
    positions = [site.pos for site in sys_sites]

    rng = np.random.default_rng(seed)
    new_params = {
        'MJ': MJ,
        'A': A,
        'bond_lengthscale': bond_lengthscale,
        'bond_power': bond_power,
        'dis_onsite': onsite_disorder,
        'rng_W': rng,
    }

    ham = syst.finalized().hamiltonian_submatrix(
        params=new_params,
        sparse=True
        )

    L = spectral_localizer_AII3D(
        np.array(positions),
        ham,
        E0=E0,
        kappa=kappa,
        norbs=4,
        whole_localizer=False,
    )
    return sign_det(L)
