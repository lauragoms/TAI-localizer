

import numpy as np
from numpy import einsum
import scipy.sparse as sp
import kwant

from tai_localizer.perulizer.misc import sigma_x, sigma_y, sigma_z, sigma_0
from .functions import _fast_pfaffian, eigsh, find_dos_gap


def spectral_localizer_AII2D(
    positions: list,
    ham: sp.csr_matrix,
    E0: float,
    kappa: float,
    norbs: int = 4,
    rotated: bool = True,
    X0 = None,
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

    TR = sp.kron(sp.kron(sp.eye(Ls), sigma_0), sigma_y)

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

    return loc_rotated if rotated else localizer


def pfaff_sign(loc_rotated: np.array):
    invariant = np.sign(_fast_pfaffian(loc_rotated)[0])
    return invariant


def local_gap_localizer(localizer: sp.csr_matrix):
        local_gap = eigsh(localizer, k=1, sigma=0)
        return np.abs(local_gap)


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

