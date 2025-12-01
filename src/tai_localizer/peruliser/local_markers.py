import ctypes
import numpy as np
from jax.numpy import einsum

import pfapack.ctypes as cpf
from koala.lattice import Lattice

from tai_localizer.peruliser.misc import sigma_x,sigma_y,sigma_z,bhz_trs_operator

def chern_marker(l, P, fix=False):
    positions = l.vertices.positions
    n_orbitals = P.shape[0] // l.n_vertices

    # make x and y arrays
    x = positions[:, 0]
    y = positions[:, 1]
    x = np.kron(x, np.ones(n_orbitals))
    y = np.kron(y, np.ones(n_orbitals))

    # make shift matrices
    shifts_x = np.outer(np.ones(len(x)), x) - np.outer(
        x, np.ones(len(x))
    )
    shifts_y = np.outer(np.ones(len(y)), y) - np.outer(
        y, np.ones(len(y))
    )

    if fix:
        shifts_x = (shifts_x + 0.5) % 1 - 0.5
        shifts_y = (shifts_y + 0.5) % 1 - 0.5

    marker2 = einsum("ij,jk,kl,li -> i", P, P * shifts_x, P * shifts_y, P).imag

    # sum over orbitals
    m_out  = marker2.reshape(l.n_vertices, n_orbitals).sum(axis=1)
    return m_out * 4 * np.pi * l.n_vertices 


def _fast_pfaffian(K):
    if np.iscomplex(K).any():
        raise AttributeError("Matrix should be real >:|")
    skpf10_d = cpf._init("skpf10_d")
    matrix_f = np.asarray(K, dtype=np.float64, order="F")
    result_array = (ctypes.c_double * 2)(0.0, 0.0)
    uplo_bytes = "U".encode()
    method_bytes = "P".encode()
    skpf10_d(K.shape[0], matrix_f, result_array, uplo_bytes, method_bytes)
    return (result_array[0], result_array[1])




def z2_spec_loc(
    lattice: Lattice, hamiltonian: np.ndarray, energy: float, trs_operator: np.ndarray, kappa = 1
) -> int:

    pos = lattice.vertices.positions
    h_size = hamiltonian.shape[0]
    n_dof = h_size // len(pos)
    full_trs = np.kron(np.eye(lattice.n_vertices), trs_operator)

    # this operation could be faster if we used sparse...
    h_trs = einsum("ji,jk,kl -> il", full_trs, hamiltonian, full_trs)

    assert trs_operator.shape == (n_dof, n_dof), "TRS operator has wrong shape"
    assert np.allclose(h_trs, hamiltonian.conj()), "System doesn't have TRS symmetry"

    # make the q operator:
    q = np.eye(2 * h_size) + np.kron(sigma_y, full_trs)

    X = np.kron(np.diag(pos[:, 0] - 0.5), np.eye(n_dof))
    Y = np.kron(np.diag(pos[:, 1] - 0.5), np.eye(n_dof))

    naive_localiser = (
        np.kron(sigma_x, X)*kappa
        + np.kron(sigma_y, Y)*kappa
        + np.kron(sigma_z, hamiltonian - energy * np.eye(h_size))
    )

    rotated_localiser = -einsum("ji,jk,kl -> il", q.conj(), naive_localiser, q)
    assert np.allclose(rotated_localiser.real, 0), "Rotation matrix doesn't work"
    loc = rotated_localiser.imag
    assert np.allclose(loc + loc.T, 0), "Localiser not antisymmetric"

    pf = _fast_pfaffian(loc)
    return np.sign(pf[0])
