import kwant
import numpy as np
from numpy import kron
from math import sqrt

from scipy import spatial
import copy

from .functions import Amorphous, spherical_coord_general, sigma_0, sigma_x, sigma_y, sigma_z

params = dict(
    pbc=False,
    R=1.01,
    MJ=2.3,
    lambdaJ=1,
    BJ=0.,
    break_PH=0.05,
    norbs=4,
    name='a',
    dis_onsite=0,
    mu=0
    )

tau_0 = s_0 = sigma_0
tau_x = s_x = sigma_x
tau_y = s_y = sigma_y
tau_z = s_z = sigma_z


def Bond_3D(
    lat: list,
    D: float
):

    """ Return bonds in 3D lattice

    Parameters:
    ----------
    lat : list
        List of lattice site positions
    D : float
        Cut-off distance for bonding
    """

    og_size = len(lat)  # how many sites in OBC
    out = copy.deepcopy(lat)

    tree = spatial.KDTree(out)
    # Find points in lat that are within distance r of out
    bonds = tree.query_ball_point(x=lat, r=D)
    info_bond = list()

    for i in range(len(bonds)):
        b = bonds[i]
        b.remove(i)  # sremove onsite
        a = list()
        for item in b:
            new_index = item
            if item >= og_size:  # not in OBC
                # Readjusts from a PBC copy to the index in the OBC
                new_index = item - int(item/og_size)*og_size
            if i < new_index:
                info_bond.append([i, new_index])
                # print(i,new_index,dist,out[i],out[item])
                a.append(new_index)

    return info_bond


def onsite(
        site: kwant.builder.SiteFamily,
        MJ: float,
        BJ: float,
        dis_onsite: float,
        rng_W: np.random.default_rng,
):
    # break_PH_term  = break_PH*kron(tau_0,sigma_0)
    W = dis_onsite
    disorder = rng_W.uniform(-W / 2, W / 2)

    return (
        MJ*kron(tau_z, sigma_0)
        + kron(tau_0, (BJ/sqrt(3))*(sigma_x + sigma_y + sigma_z))
        + disorder*kron(tau_0, sigma_0)
        )


def amorph_hopping(
        site1: kwant.builder.SiteFamily,
        site2: kwant.builder.SiteFamily,
        norbs: int,
        lambdaJ: float,
        ):

    vec = np.array(site1.pos - site2.pos)
    rho, theta, phi = spherical_coord_general(*vec)
    
    if np.allclose(vec, np.zeros(3), atol=1e-5):
        return np.zeros((norbs, norbs), dtype=complex)

    else:
        SIGMA = -(vec[0]*sigma_x + vec[1]*sigma_y + vec[2]*sigma_z)
        return (kron(tau_z, sigma_0)*(1/2)
                - 1j*lambdaJ*kron(tau_x, SIGMA)*(1/2))


# sites = [(x,y,z) for x in range(-Lx,Lx) for y in range(-Ly,Ly) for z in range (-Lz,Lz)] 


def create_cubic_TI(
        sites: list,
        bonds: list
):

    lat = Amorphous(sites)
    syst = kwant.Builder()

    # Onsite terms
    for i in range(len(sites)):
        syst[lat(i)] = onsite

    # Hopping terms
    for i, j, vec in bonds:
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
