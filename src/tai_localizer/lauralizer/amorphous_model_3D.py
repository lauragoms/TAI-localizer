import kwant
import numpy as np
from numpy import kron
from math import sqrt


from .functions import (
    Amorphous,
    spherical_coord_general,
    sigma_0,
    sigma_x,
    sigma_y,
    sigma_z)

params = dict(
    pbc=False,
    R=1.01,
    MJ=2.3,
    lambdaJ=1,
    norbs=4,
    name='a',
    dis_onsite=0,
    )

tau_0 = s_0 = sigma_0
tau_x = s_x = sigma_x
tau_y = s_y = sigma_y
tau_z = s_z = sigma_z


def onsite(
        site: kwant.builder.SiteFamily,
        MJ: float,
        # BJ: float,
        dis_onsite: float,
        rng_W: np.random.default_rng,
):
    # break_PH_term  = break_PH*kron(tau_0,sigma_0)
    W = dis_onsite
    disorder = rng_W.uniform(-W / 2, W / 2)

    return (
        MJ*kron(tau_z, sigma_0)
        + disorder*kron(tau_0, sigma_0)
        # + kron(tau_0, (BJ/sqrt(3))*(sigma_x + sigma_y + sigma_z))
        )


def amorph_hopping(
        site1: kwant.builder.SiteFamily,
        site2: kwant.builder.SiteFamily,
        lambdaJ: float,
        bond_lengthscale: float,
        bond_power: float,
        ):

    vec = np.array(np.array(site2.pos) - np.array(site1.pos))
    rho, theta, phi = spherical_coord_general(*vec)
    tx = np.sin(theta)*np.cos(phi) * np.kron(tau_x, sigma_x)
    ty = np.sin(theta)*np.sin(phi) * np.kron(tau_x, sigma_y)
    tz = np.cos(theta) * np.kron(tau_x, sigma_z)
    t0 = 1j*kron(tau_z, sigma_0)

    rescaled_distance = (rho - bond_lengthscale) / bond_lengthscale
    hopping_multiplier = np.exp(- rescaled_distance * bond_power)
    # print(vec,np.round((-1j*(lambdaJ/2)*(tx + ty + tz + t0)) * hopping_multiplier, 5))
    # print(vec, hopping_multiplier)
    return (-1j*(lambdaJ/2)*(tx + ty + tz + t0)) * hopping_multiplier


# sites = [(x,y,z) for x in range(-Lx,Lx) for y in range(-Ly,Ly) for z in range (-Lz,Lz)] 


def amorph_3DTI(
        sites: list,
        bonds: list
):

    lat = Amorphous(sites)
    syst = kwant.Builder()

    # Onsite terms
    for i in range(len(sites)):
        syst[lat(i)] = onsite

    # Hopping terms
    for i, j in bonds:
        syst[lat(i), lat(j)] = amorph_hopping

    return syst


'''Bloch Hamiltonian'''


def Bloch_H(
        kx: float,
        ky: float,
        kz: float,
        MJ: float,
        lambdaJ: float,
        BJ: float,
        break_PH: float,
):
    """
    Constructs the Bloch Hamiltonian for a cubic lattice with a given k-vector.
    """

    H1 = (MJ + np.cos(kx) + np.cos(ky) + np.cos(kz)) * kron(tau_z, sigma_0)
    H2 = (lambdaJ*(np.sin(kx)*kron(tau_x, sigma_x)
                   + np.sin(ky)*kron(tau_x, sigma_y)
                   + np.sin(kz)*kron(tau_x, sigma_z)))
    H3 = ((BJ/sqrt(3))*(kron(tau_0, sigma_x)
                        + kron(tau_0, sigma_y)
                        + kron(tau_0, sigma_z)))
    H_break_PH = break_PH*kron(tau_0, sigma_0)

    return H1 + H2 + H3 + H_break_PH
