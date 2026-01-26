from . import crystalline_model_BHZ_2D as bhz

import numpy as np
import scipy.sparse as sp
from scipy.linalg import qr
import kwant
import sys

import mumps
from scipy import spatial
import copy

import pfapack.ctypes as cpf
import ctypes

ctx = mumps.Context()  # Provided by python-mumps package


def zero_params(system):
    return {parameter: 0 for parameter in system.parameters}


norbs = 4
class Amorphous(kwant.builder.SiteFamily):
    """Creates a lattice from the positions of sites."""

    def __init__(self, coords):
        n = "a"
        self.coords = coords
        super(Amorphous, self).__init__(str(n + n), str(n), norbs)

    def normalize_tag(self, tag):
        try:
            tag = int(tag[0])
        except:
            raise KeyError
        if 0 <= tag < len(self.coords):
            return tag
        else:
            raise KeyError

    def pos(self, tag):
        return self.coords[tag]

    def family(self):
        n = "a"
        return str(n)


def _fast_pfaffian(K):
    if np.iscomplex(K).any():
        raise Exception("Matrix should be real >:|")
    skpf10_d = cpf._init("skpf10_d")
    matrix_f = np.asarray(K, dtype=np.float64, order="F")
    result_array = (ctypes.c_double * 2)(0.0, 0.0)
    uplo_bytes = "U".encode()
    method_bytes = "P".encode()
    skpf10_d(K.shape[0], matrix_f, result_array, uplo_bytes, method_bytes)
    return (result_array[0], result_array[1])


def update_progress(progress, decimalpoints=0):
    """Make an interactive progress bar as described on:
    https://stackoverflow.com/questions/3160699/python-progress-bar
    """
    barLength = 20  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    text = "\rPercent: [{0}] {1}% {2}".format(
        "#" * block + "-" * (barLength - block),
        round(progress * 100, decimalpoints),
        status,
    )
    sys.stdout.write(text)
    sys.stdout.flush()


"""Eigenvalues with mumps"""


class LuInv(sp.linalg.LinearOperator):
    def __init__(self, A):
        # Kwant mumps only wraps complex dtype
        A = A.astype(complex, copy=False)

        inst = ctx
        inst.analyze(A, ordering="metis")
        inst.factor(A)
        self.solve = inst.solve
        sp.linalg.LinearOperator.__init__(self, A.dtype, A.shape)

    def _matvec(self, x):
        return self.solve(x.astype(self.dtype))


def eigsh(
    A,
    k,
    sigma,
    return_eigenvectors=False,
    **kwargs,
):
    """Call sla.eigsh with mumps support and sorting.

    Please see scipy.sparse.linalg.eigsh for documentation.
    """

    opinv = LuInv(A - sigma * sp.identity(A.shape[0]))
    out = sp.linalg.eigsh(
        A,
        k,
        sigma=sigma,
        OPinv=opinv,
        return_eigenvectors=return_eigenvectors,
        tol=1e-4,
        **kwargs,
    )

    if not return_eigenvectors:
        return np.sort(out)

    # Scipy sparse.eigsh does not sort the eigenvalues nor orthonormalize the
    # eigenvectors. To fix that we do QR on groups of eigenvectors with the
    # numerically same eigenvalue. Because this is done on a few vectors at a
    # time, the numerical cost is negligible.
    evals, evecs = out
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]
    # We can set a relatively loose bound on degeneracy because different
    # subspaces are also orthogonal.
    eps = 1e-7 * (evals[-1] - evals[0]) + 1e-7
    groups = np.split(evecs, np.where(np.diff(evals) > eps)[0] + 1, axis=1)
    for group in groups:
        if group.shape[1] > 1:
            group[:, 1:] = qr(group[:, 1:], mode="economic")[0]
    evecs = np.concatenate(groups, axis=1)
    return evals, evecs


"""Defining Pauli matrices"""
sigma_0 = np.array([[1, 0], [0, 1]])
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
sigma_01 = sp.csr_matrix(([1], ([0], [1])), shape=(2, 2))
sigma_10 = sp.csr_matrix(([1], ([1], [0])), shape=(2, 2))


"""Conductance"""


def conductance_E(E, syst):
    S = kwant.smatrix(syst, energy=E)
    G = S.transmission(1, 0)
    return G


def average_conductance_W(E, Wr, Lx=4, Ly=4, num_reals=50, params=bhz.params):

    G_W = []
    G_W_reals = []

    for W in Wr:

        G_reals = []
        seed_range = np.arange(num_reals)

        if W == 0:
            seed_range = [0]
            num_reals = 1

        print("Averaging over realizations...")

        for ind, val in enumerate(seed_range):
            update_progress((ind + 1) / len(seed_range))
            params["W"] = W
            params["seed_W"] = val
            # syst = mcryst.hexagonal_syst_with_leads(p=p).finalized()
            syst, _, _ = bhz.BHZ_with_leads(Lx, Ly, params=params)
            G = conductance_E(E, syst)
            G_reals.append(G)

        if W == 0:
            G_avg = G
        else:
            G_avg = np.mean(np.array(G_reals), axis=0)

        G_W.append(G_avg)
        G_W_reals.append(G_reals)

        print("W:", W, "G:", G_avg)

    return G_W, G_W_reals


def polar_coords(x0, y0):
    """Determines spherical coordinates of a point with
    respect to 0.


    Parameters
    -----------------
    x0, y0 : floats,
        2D position of the point
    Returns
    ----------------
    rho,  phi : floats,
        spherical coordinates"""

    rho = np.sqrt((x0) ** 2 + (y0) ** 2)

    phi = np.sign(y0) * np.arccos(x0 / np.sqrt(x0**2 + y0**2))
    return rho, phi


def spherical_coord_general(x0, y0, z0):
    ''' Determines spherical coordinates of a point with
    respect to 0.

    Parameters
    -----------------
    x0, y0, z0: floats,
        3D position of the point (or vector)

    Returns
    ----------------
    rho, theta, phi: floats,
        spherical coordinates'''

    rho = np.sqrt((x0)**2+(y0)**2+(z0)**2)
    phi = np.arctan2(y0, x0)
    theta = np.arccos(((z0)/rho))
    return rho, theta, phi


def bonds_func(
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
