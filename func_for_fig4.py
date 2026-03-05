import kwant
import scipy.sparse as sp
import numpy as np
from tai_localiser.lauralizer.amorphous_model_3D import (
    amorph_3DTI
    )
from tai_localiser.lauralizer.functions import bonds_func
from tai_localiser.lauralizer.localizer import (
    spectral_localizer_AII3D,
    sign_det
    )
from koala import pointsets
import pickle as pi

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
    disorder_average: int,
    system_size: int,
    kappa_spec,
    E0,
    bond_power: float,
    bond_lengthscale: float,
    sigma: float,
    kappa_shift: float,
    beta: float,
    resolution: int,
    provide_sites: bool,
    **kwargs,
):
    # create lattice:
    sites = grid_3D(system_size, system_size, system_size)

    idx_dis = []

    for seed in range(disorder_average):
        rng = np.random.default_rng(seed)
        # structural disorder
        sites = pointsets.move_all_points(sites, sigma, kappa_shift, beta, resolution=resolution)

        # create bonds
        bond_distance = 1.3 / system_size
        bonds = bonds_func(sites, bond_distance)

        # create system
        syst = amorph_3DTI(sites, bonds)
        sys_sites = syst.finalized().sites
        positions = [site.pos for site in sys_sites]
        # kwant.plot(syst.finalized(), site_size=1, hop_lw=0.1)

        # create hamiltonian with system params
        new_params = {
            'MJ': MJ,
            'A': A,
            'bond_lengthscale': bond_lengthscale,
            'bond_power': bond_power,
            'dis_onsite': 0,  # we add disorder later to the Hamiltonian, so we set this to zero here
            'rng_W': rng,  # not used, but we need to provide it to create the system
        }

        ham = syst.finalized().hamiltonian_submatrix(
            params=new_params,
            sparse=True
            )
        # onsite disorder
        ham_W = ham + sp.diags(rng.uniform(
            -onsite_disorder/2, onsite_disorder/2, ham.shape[0]))
        # compute localizer and index
        L = spectral_localizer_AII3D(
            np.array(positions),
            ham_W,
            E0=E0,
            kappa=kappa_spec,
            norbs=4,
            whole_localizer=False,
        )
        idx_dis.append(sign_det(L, **kwargs))
    return np.mean(np.array(idx_dis))  # average over disorder realizations
