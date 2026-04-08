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


def params_obs_3D(
    MJ: float,
    A: float,
    onsite_disorder: float,
    system_size: int,
    points,
    kappa_spec,
    E0,
    bond_power: float,
    bond_lengthscale: float,
    sigma: float,
    kappa_shift: float,
    beta: float,
    resolution: int,
    seed,
    **kwargs,
):

    rng = np.random.default_rng(seed)
    # structural disorder
    points = pointsets.move_all_points(
        points, sigma, kappa_shift, beta, resolution=resolution, rng=rng
        )

    # create bonds
    bond_distance = 1.3 / system_size
    bonds = bonds_func(points, bond_distance)

    # create system
    syst = amorph_3DTI(points, bonds)
    sys_sites = syst.finalized().sites
    positions = [site.pos for site in sys_sites]
    # kwant.plot(syst.finalized(), site_size=1, hop_lw=0.1)

    # create hamiltonian with system params
    new_params = {
        'MJ': MJ,
        'A': A,
        'bond_lengthscale': bond_lengthscale,
        'bond_power': bond_power,
        'dis_onsite': onsite_disorder,
        'rng_W': rng,  # not used, but we need to provide it to create the system
    }

    ham = syst.finalized().hamiltonian_submatrix(
        params=new_params,
        sparse=True
        )

    # compute localizer and index
    L = spectral_localizer_AII3D(
        np.array(positions),
        ham,
        E0=E0,
        kappa=kappa_spec,
        norbs=4,
        whole_localizer=False,
    )
    idx_dis = sign_det(L, **kwargs)

    return idx_dis, points
