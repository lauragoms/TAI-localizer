from .functions import (
    _fast_pfaffian,
    eigsh,
    find_dos_gap,
    sigma_x,
    sigma_y,
    sigma_z,
    sigma_0
    )

import numpy as np
import scipy.sparse as sp
import kwant
import mumps



sigma_01 = sp.csr_matrix(([1], ([0], [1])), shape=(2, 2))
sigma_10 = sp.csr_matrix(([1], ([1], [0])), shape=(2, 2))


def spectral_localizer_AII2D(
    positions: list,
    ham: sp.csr_matrix,
    E0: float,
    kappa: float,
    norbs: int = 4,
    rotated: bool = True,
    X0=None,
    time_reversal_operator=None,
) -> sp.csr_matrix:
    """
    Spectral Localizer for class AII2D
    
    :param positions: Position of the sites
    :type positions: list
    :param ham: Sparse Hamiltonian of the system
    :type ham: sp.csr_matrix
    :param E0: Energy to compute the spectral localizer
    :type E0: float
    :param kappa: Parameter Kappa for the spectral localizer
    :type kappa: float
    :param norbs: Number of orbitals per site
    :type norbs: int
    :param X0: Reference position for the localizer
    :type X0: tuple or None
    :param rotated: True for computing the pfaffian
    :type rotated: bool
    """

    if X0 is None:
        x0, y0 = np.average(positions, axis=0)
    else:
        x0, y0 = X0

    Ls = ham.shape[0] // norbs

    X = sp.diags(np.kron(positions[:, 0], [1] * norbs))
    Y = sp.diags(np.kron(positions[:, 1], [1] * norbs))

    if time_reversal_operator is None:
        TR = sp.kron(sp.kron(sp.eye(Ls), sigma_0), sigma_y)
    else:
        TR = time_reversal_operator

    # h_trs = einsum("ji,jk,kl -> il", (-1j*TR).todense(),
    #                ham.todense(), (-1j*TR).todense())
    h_trs = TR @ ham @ TR

    assert np.allclose(sp.linalg.norm(h_trs-ham.conj()).max(), 0), "System doesn't have TRS symmetry"

    D = (X - (x0) * sp.eye(norbs * Ls)) + 1j * (Y - (y0) * sp.eye(norbs * Ls))
    Q = (1 / np.sqrt(2)) * sp.bmat(
        [[sp.eye(norbs * Ls), TR], [-TR, sp.eye(norbs * Ls)]]
    )
    h = ham - E0 * sp.eye(norbs * Ls)

    localizer = sp.bmat([[h, kappa * D.conj()], [kappa * D, -h]])
    loc_rotated = 1j * Q.conj() @ localizer @ Q

    out = loc_rotated if rotated else localizer
    # make sure its real
    assert np.allclose(abs(out.imag).max(), 0), "Localizer is not real"

    return out.real


def pfaff_sign(loc_rotated: np.array):
    invariant = np.sign(_fast_pfaffian(loc_rotated)[0])
    return invariant


def spectral_localizer_AII3D(
    positions: np.array,
    ham: sp.csr_matrix,
    E0: float,
    kappa: float,
    norbs: int = 4,
    X0=None,
    whole_localizer: bool = False,
) -> sp.csr_matrix:

    """
    Spectral Localizer for class AII3D

    Parameters
    ----------
    positions : np.array
        List with positions of the sites
    ham : sp.csr_matrix
        Sparse Hamiltonian of the system
    E0 : float
        Energy to compute the spectral localizer
    kappa : float
        Parameter Kappa for the spectral localizer
    norbs : int, optional
        Number of orbitals per site, by default 4
    X0 : tuple, optional
        Reference position for the localizer, by default Center
    whole_localizer : bool, optional
        If True, returns the whole localizer, by default False
    Returns
    -------
    sp.csr_matrix
        Sparse spectral localizer matrix
    """
    if X0 is None:
        x0, y0, z0 = np.average(positions, axis=0)
    else:
        x0, y0, z0 = X0

    X = sp.diags(
        np.kron(positions[:, 0], [1] * norbs)
        ) - (x0) * sp.eye(ham.shape[0])
    Y = sp.diags(
        np.kron(positions[:, 1], [1] * norbs)
        ) - (y0) * sp.eye(ham.shape[0])
    Z = sp.diags(
        np.kron(positions[:, 2], [1] * norbs)
        ) - (z0) * sp.eye(ham.shape[0])

    D = (
        sp.kron(sigma_x, X)
        + sp.kron(sigma_y, Y)
        + sp.kron(sigma_z, Z)
    )
    # print(sp.kron(Z, sigma_z))

    h = sp.kron(sigma_0, (ham - E0 * sp.eye(ham.shape[0])))
    # Block of Localizer, just need this for Z2
    block_1 = h - 1j * kappa * D

    if whole_localizer:
        localizer = (
            sp.kron(sigma_01, block_1)
            + sp.kron(sigma_10, block_1.getH())
            )

    return localizer if whole_localizer else block_1


def sign_det(matrix: sp.csr_matrix, **kwargs):
    ctx = mumps.Context(**kwargs)
    return np.real(ctx.slogdet(matrix, ordering="scotch")[0])


def local_gap_localizer(localizer: sp.csr_matrix, k=1, **kwargs):
    local_gap = eigsh(localizer, k=k, sigma=0, **kwargs)
    return local_gap


def dos_kpm(L: sp.csr_matrix,
            bounds: tuple = (-1, 1),
            number_points: int = 200,
            energy_resol: float = 0.01):

    es = np.linspace(*bounds, number_points)
    spectrum = kwant.kpm.SpectralDensity(hamiltonian=L)
    spectrum.add_moments(energy_resolution=energy_resol)
    density_subset = spectrum(es)
    DOS = np.real(density_subset) / np.max(np.real(density_subset))

    gap = find_dos_gap(es, DOS)
    return DOS, gap
