from .functions import (
    Amorphous,
    polar_coords,
    bonds_func,
    sigma_0,
    sigma_x,
    sigma_y,
    sigma_z)

import numpy as np
import kwant


params = dict(
    norbs=4,
    Lx=5,
    Ly=5,
    Delta=4.0,
    B=1,
    A=1,
    mu=0,
    mu_leads=0,
    dis_onsite=0,
    seed_onsite=0,
    dis_hadamard=0,
    seed_hadamard=0,
    R=1.1,
    name="a",
    dis_amorph=0,
    seed_amorph=0,
)

"""helper functions"""


def onsite(
    site: kwant.builder.SiteFamily,
    rng_W: np.random.default_rng,
    dis_onsite: float,
    Delta: float,
    B: float,
    norbs: int,
    mu: float,
):
    W = dis_onsite
    disorder = rng_W.uniform(-W / 2, W / 2)
    return (
        (Delta - 4 * B) * np.kron(sigma_z, sigma_0)
        + disorder * np.eye(norbs)
        + mu * np.kron(sigma_0, sigma_0)
    )


def amorph_hopping(
    site1: kwant.builder.SiteFamily,
    site2: kwant.builder.SiteFamily,
    A: float,
    B: float,
    bond_lengthscale: float,
    bond_power: float,
):

    vec = np.array(np.array(site2.pos) - np.array(site1.pos))

    rho, phi = polar_coords(vec[0], vec[1])

    hop_strength = B * np.kron(sigma_z, sigma_0)
    hop_x_SOC = np.cos(phi) * (A / (2j) * np.kron(sigma_x, sigma_x))
    hop_y_SOC = -np.sin(phi) * A / (2j) * np.kron(sigma_x, sigma_y)

    rescaled_distance = (rho - bond_lengthscale) / bond_lengthscale
    hopping_multiplier = np.exp(- rescaled_distance * bond_power)

    return hopping_multiplier*(hop_strength + hop_x_SOC + hop_y_SOC)


def amorph_BHZ(sites, bonds):
    """
    Returns an amorphous 2D BHZ model system with given parameters.

    Parameters
    ----------
    sites : list
        List of position of the amorphous system.
    p : dict
        Dictionary containing the model parameters
        {A,B,Delta,dis_onsite,dis_hadamard}

    Returns
    -------
    syst : kwant.builder.Builder
    """

    lat = Amorphous(sites)
    syst = kwant.Builder()

    for i in range(len(sites)):
        syst[lat(i)] = onsite

    for i, j in bonds:
        syst[lat(i), lat(j)] = amorph_hopping

    return syst
