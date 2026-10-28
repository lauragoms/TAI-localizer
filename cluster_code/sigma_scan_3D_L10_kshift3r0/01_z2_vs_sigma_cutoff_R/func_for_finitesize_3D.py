import gc
import numpy as np
from tai_localiser.lauralizer.amorphous_model_3D import (
    amorph_3DTI
    )
from tai_localiser.lauralizer.functions import bonds_func
from tai_localiser.lauralizer.localizer import (
    spectral_localizer_AII3D,
    sign_det
    )
from tai_localiser.perulizer.lattices import iterative_move_point


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
    bond_distance: float,
    sigma: float,
    kappa_shift: float,
    beta: float,
    resolution: int,
    seed,
    **kwargs,
):
    """
    Computes the Z2 invariant (sign of det of the 3D spectral localizer
    block) for one disorder realisation.

    Same as cluster_code/finite-size/func_for_finitesize_3D.py, except that
    the bond cutoff is passed as `bond_distance` (absolute units) instead of
    being fixed to 1.3 / system_size.
    """

    rng = np.random.default_rng(int(seed))
    # structural disorder
    points = iterative_move_point(
        points, sigma, kappa_shift, resolution=resolution, rng=rng
        )

    # create bonds (open boundaries)
    bonds = bonds_func(points, bond_distance)

    # create system
    syst = amorph_3DTI(points, bonds)
    fsyst = syst.finalized()
    positions = [site.pos for site in fsyst.sites]

    # create hamiltonian with system params; the onsite Anderson term is
    # uniform in [-W/2, W/2] times identity, drawn from rng after the points
    new_params = {
        'MJ': MJ,
        'A': A,
        'bond_lengthscale': bond_lengthscale,
        'bond_power': bond_power,
        'dis_onsite': onsite_disorder,
        'rng_W': rng,
    }

    ham = fsyst.hamiltonian_submatrix(
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
    del L
    gc.collect()

    return idx_dis
