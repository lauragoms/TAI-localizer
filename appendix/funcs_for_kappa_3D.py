import scipy.sparse as sp
import numpy as np
from tai_localiser.lauralizer.amorphous_model_3D import (
    amorph_3DTI
    )
from tai_localiser.lauralizer.functions import bonds_func
from tai_localiser.lauralizer.localizer import (
    spectral_localizer_AII3D,
    local_gap_localizer,
    )
from koala import pointsets
from koala.pointsets import grid


def localgap_sys_3D(
    MJ: float,
    A: float,
    onsite_disorder: float,
    seed: int,
    system_size: int,
    kappa_spec: float,
    E0: float,
    bond_power: float,
    bond_lengthscale: float,
    sigma: float,
    kappa_shift: float,
    beta: float,
    resolution: int,
    **kwargs,
):
    rng = np.random.default_rng(seed)

    # create lattice:
    sites = grid(system_size, system_size, system_size)

    # structural disorder
    sites = pointsets.move_all_points(
        sites, sigma, kappa_shift, beta, resolution=resolution
        )

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
    locgap = np.abs(local_gap_localizer(L))[0]
    return locgap
