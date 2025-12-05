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
)


sigma_0 = np.eye(2)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

"""Crystalline system"""


def onsite(site, rng_W, dis_onsite, Delta, B, norbs, mu):
    W = dis_onsite
    disorder = rng_W.uniform(-W / 2, W / 2)
    return (
        (Delta - 4 * B) * np.kron(sigma_z, sigma_0)
        + disorder * np.eye(norbs)
        + mu * np.kron(sigma_0, sigma_0)
    )  # peru's code is -1 because of ws, wp = -1


def hop_x(site1, site2, dis_hadamard, rng_hdmd, A, B):
    spin = sigma_z
    # print(rng_hdmd.choice([0, 1], p=[1 - hadamard/100, hadamard/100])==1)

    if (
        dis_hadamard != 0
        and rng_hdmd.choice([0, 1], p=[1 - dis_hadamard / 100, dis_hadamard / 100]) == 1
    ):
        spin = sigma_x

    return B * np.kron(sigma_z, sigma_0) + A / (2j) * np.kron(sigma_x, spin)


def hop_y(site0, site1, A, B):
    return B * np.kron(sigma_z, sigma_0) - A / (2j) * np.kron(sigma_y, sigma_0)


def system_2D_BHZ(Lx, Ly):
    """
    Returns a 2D BHZ model system with given parameters.

    Parameters
    ----------
    params : dict
        Dictionary containing the model parameters

    Returns
    -------
    sys : kwant.builder.Builder
    """

    lat = kwant.lattice.square(norbs=4)


    sys = kwant.Builder()

    sys[(lat(x, y) for x in range(-Lx, Lx) for y in range(-Ly, Ly))] = onsite
    sys[kwant.builder.HoppingKind((1, 0), lat, lat)] = hop_x
    sys[kwant.builder.HoppingKind((0, 1), lat, lat)] = hop_y

    return sys, lat


def onsite_lead(site, Delta, B, mu_leads):
    return (Delta - 4 * B) * np.kron(sigma_z, sigma_0) + mu_leads * np.kron(
        sigma_0, sigma_0
    )  # peru's code is -1 because of ws, wp = -1


def lead_BHZ(Ly, lat):

    sym = kwant.TranslationalSymmetry(lat.vec((1, 0)))
    lead = kwant.Builder(sym)

    lead[(lat(0, y) for y in range(-Ly, Ly))] = onsite_lead
    lead[kwant.builder.HoppingKind((1, 0), lat, lat)] = hop_x
    lead[kwant.builder.HoppingKind((0, 1), lat, lat)] = hop_y

    return lead


def BHZ_with_leads(Lx, Ly, params_syst, params_lead):

    syst, lat = system_2D_BHZ(Lx, Ly)
    lead_plus = lead_BHZ(Ly, lat)

    lead_minus = lead_plus.reversed()

    syst.attach_lead(lead_plus)  # to layer k=0
    syst.attach_lead(lead_minus)

    return syst.finalized(), lead_plus.finalized(), lead_minus.finalized()


def TR_op(len_sites):
    return np.kron(np.kron(np.identity(len_sites), sigma_0), sigma_y)
