from . import model_BHZ_2D as bhz
from . import model_TI_3D as m3d
# from . import pfaffian as pff

import numpy as np
import scipy.sparse as sp
import kwant
import numpy as np
import numpy.linalg as npl
import time
import sys
import pylab as py
import scipy.sparse as sp
from scipy.linalg import qr
import mumps

import pfapack.ctypes as cpf
import ctypes

ctx = mumps.Context()  # Provided by python-mumps package


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


params = dict(
    kappa=0.1, 
    W=0, 
    num_reals=50, 
    maxiter=2000, 
    tol_locgap=1e-6, 
)

"""Defining Pauli matrices"""
sigma_0 = np.array([[1, 0], [0, 1]])
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
sigma_01 = sp.csr_matrix(([1], ([0], [1])), shape=(2, 2))
sigma_10 = sp.csr_matrix(([1], ([1], [0])), shape=(2, 2))


"""Spectral Localizer BHZ model"""


def pos_H(fsyst, coord=0):
    """Calculate the position operator in the 'coord' direction of the
    Hamiltonian of fsyst. It also computes the position operator for a given position with index=index_origin in fsyst.sites
    """
    H, ton, fon = fsyst.hamiltonian_submatrix(return_norb=True)
    x = np.zeros(H.shape)
    ind = 0

    for i in range(len(fsyst.sites)):
        for j in range(ind, ind + ton[i]):

            x[j, j] = fsyst.sites[i].pos[coord]
            # x0[j, j] = fsyst.sites[index_origin].pos[coord]
        ind += ton[i]

    return x


def get_center(fsyst_sites):
    sites = {}
    d = len(fsyst_sites[0].pos)

    for i in range(d):
        sites[i] = []

    for site in fsyst_sites:
        site = site.pos
        for coord in range(d):
            sites[coord].append(site[coord])
    if d == 1:
        return (np.max(np.array(sites[0])) + np.min(np.array(sites[0]))) / 2
    if d == 2:
        return (np.max(np.array(sites[0])) + np.min(np.array(sites[0]))) / 2, (
            np.max(np.array(sites[1])) + np.min(np.array(sites[1]))
        ) / 2
    if d == 3:
        return (
            (np.max(np.array(sites[0])) + np.min(np.array(sites[0]))) / 2,
            (np.max(np.array(sites[1])) + np.min(np.array(sites[1]))) / 2,
            (np.max(np.array(sites[2])) + np.min(np.array(sites[2]))) / 2,
        )


def spectral_localizer_AII2D(
    syst,
    W,
    E0,
    kappa,
    X0=np.array(["None"]),
    num_reals=50,
    p=params,
    compute_inv=True,
    compute_localgap=False,
    compute_DOS=False,
):
    """Computes the spectral localizer for a 2D AII system.
    ----------------
    syst: kwant object non-finalized,

    Returns
    ----------------
    pfaff_sign: Topological invariant as the sign of the pfaffian."""



    fsyst = syst.finalized()
    Ls = len(fsyst.sites)
    ham = fsyst.hamiltonian_submatrix()

    if X0[0] == "None":
        X0 = get_center(fsyst.sites)
    # if flatten==True:
    #     ham = flattened_H(ham)

    x0, y0 = X0

    X = pos_H(fsyst, coord=0)
    Y = pos_H(fsyst, coord=1)
    id = np.identity(np.shape(ham)[0])
    print("E0:", E0, "x0,y0:", x0, y0, "kappa:", kappa, "W:", W)

    TR = np.kron(np.kron(np.identity(Ls), sigma_0), sigma_y)
    D = (X - (x0) * id) + 1j * (Y - (y0) * id)

    Q = (1 / np.sqrt(2)) * np.block([[id, TR], [-TR, id]])

    if W == 0:
        num_reals = 1
    seed_range = np.arange(num_reals)
    localgap_realization = 0
    pfaffian_realizations = 0
    list_pf_reals = []
    list_localgap_realizations = []
    density_subset_realizations = np.zeros(200, dtype=np.complex128)

    print("Averaging over realizations...")
    for ind, val in enumerate(seed_range):
        update_progress((ind + 1) / len(seed_range))

        seed = val
        np.random.seed(int(seed))
        # ANDERSON'S DISORDER
        if W != 0:
            disp = np.diag(
                np.random.uniform(low=-W / 2, high=W / 2, size=len(fsyst.sites))
            )
            AND_disorder = np.kron(disp, np.eye(4))

        if W == 0:
            h = ham - E0 * id
        else:
            h = ham - E0 * id + AND_disorder

        if compute_inv:
            L = np.block([[h, kappa * np.conjugate(D)], [kappa * D, -h]])
            Hp = 1j * np.conjugate(Q) @ L @ Q  # lorings section 5.4 i*conj(Q).H.Q

            # pfaff_sign = np.real(pff.pfaffian(Hp,sign_only=True))
    
            pfaff_sign = np.sign(_fast_pfaffian(Hp, sign_only=True))
            pfaffian_realizations = pfaffian_realizations + pfaff_sign
            list_pf_reals.append(pfaff_sign)
            print("Realization Pfaffian:", np.real(pfaff_sign))

        if compute_localgap:
            L_sparse = np.block([[h, kappa * np.conjugate(D)], [kappa * D, -h]])
            start_time2 = time.perf_counter()
            local_gap = eigsh(L_sparse, k=1, sigma=0)
            print("Local gap realization:", local_gap)
            end_time2 = time.perf_counter()
            print("Time:", end_time2 - start_time2)

            localgap_realization = localgap_realization + np.abs(local_gap)
            list_localgap_realizations.append(np.abs(local_gap))
        # Compute the Density of States of the Localizer with KPM
        if compute_DOS is True:
            start_time2 = time.perf_counter()
            bounds = (-1, 1)
            es = np.linspace(*bounds, 200)
            L_sparse = sp.block([[h, kappa * np.conjugate(D)], 
                                 [kappa * D, -h]])
            spectrum = kwant.kpm.SpectralDensity(hamiltonian=L_sparse)
            spectrum.add_moments(energy_resolution=0.01)
            energy_subset = es
            density_subset = spectrum(energy_subset)
            DOS = np.real(density_subset) / np.max(np.real(density_subset))
            density_subset_realizations += np.array(DOS)
            # You get local gap for free!
            localgap = find_dos_gap(energy_subset, DOS)
            localgap_realization += localgap
            list_localgap_realizations.append(localgap)
            end_time2 = time.perf_counter()
            print("Time:", end_time2 - start_time2)

    pfaffian_averaged = pfaffian_realizations / num_reals
    localgap_average = localgap_realization / num_reals
    density_subset_average = density_subset_realizations / num_reals

    if compute_inv:
        print("Pfaffian sign:", pfaffian_averaged)
        return pfaffian_averaged, list_pf_reals

    if compute_localgap:
        print("Local gap:", np.abs(localgap_average[0]))
        return localgap_average, list_localgap_realizations

    if compute_DOS is True:
        return (
            energy_subset,
            density_subset_average,
            localgap_average,
            list_localgap_realizations,
        )


"""Spectral Localizer 3D model"""


def sparse_pos_H(fsyst_sites, H, ton, coord=0):
    """Calculate the position operator in the 'coord' direction of the
    Hamiltonian of fsyst. It also computes the position operator for a given
    position with index=index_origin in fsyst.sites.
    """

    diag_values = np.zeros(H.shape[0])
    ind = 0
    for i in range(len(fsyst_sites)):
        for j in range(ind, ind + int(ton)):
            diag_values[j] = fsyst_sites[i].pos[coord]
        ind += int(ton)

    x_sparse = sp.diags(diag_values, format="csr")

    return x_sparse


def find_dos_gap(energies, dos, threshold=0.05):
    """
    Find the energy of the first significant peak starting from zero energy.
    Returns the energy distance from zero to the first peak above threshold.
    """
    # Find energy closest to zero
    zero_idx = len(energies) // 2

    # Filter out very small DOS values that might be numerical noise
    # significant_dos = dos/np.max(dos) > threshold
    significant_dos = dos > threshold  # the input is already normalized

    if not np.any(significant_dos):
        return np.inf  # No significant peaks found

    # Find all significant peak indices
    significant_indices = np.where(significant_dos)[0]

    # Calculate distances from zero energy index to all significant peaks
    distances_to_zero = np.abs(significant_indices - zero_idx)

    # Find the closest significant peak to zero
    closest_peak_relative_idx = np.argmin(distances_to_zero)
    closest_peak_idx = significant_indices[closest_peak_relative_idx]

    # Return the energy of this closest peak (absolute energy value)
    return np.abs(energies[closest_peak_idx])


def sparse_spectral_localizer_AII3D(
    ham,
    fsyst_sites,
    W,
    E0,
    kappa,
    X0=np.array(["None"]),
    num_reals=50,
    maxiter=2000,
    compute_inv=True,
    compute_DOS=False,
    compute_localgap=False,
):
    """
    Computes the sparse spectral localizer for any type of disorder

    Parameters
    ----------------
    ham : sparse Hamiltonian matrix of the system
    fsyst_sites : list of sites of the system
    W : Anderson disorder strength
    E0 : energy at which we compute the localizer
    X0 : position origin

    Returns
    ----------------
    compute_inv = True : invariant_average,list_invariant_realizations
    compute_DOS = True : energy_subset, density_subset_average
    compute_localgap = True : localgap_average,list_localgap_realizations"""

    if X0[0] == "None":
        X0 = get_center(fsyst_sites)
        
    print(
        "DISORDER. W:",
        W,
        " & num_reals:",
        num_reals,
    )

    print("LOCALIZER. kappa: ", kappa,
                        ", E0: ", E0,
                        ", X0:", X0, 
                        ", maxiter:", maxiter)

    norbs = ham.shape[0] / len(fsyst_sites)
    # Now we define position operators centered in X0 for crystalline:
    x0, y0, z0 = X0
    X = sparse_pos_H(fsyst_sites, H=ham, ton=norbs, coord=0)
    Y = sparse_pos_H(fsyst_sites, H=ham, ton=norbs, coord=1)
    Z = sparse_pos_H(fsyst_sites, H=ham, ton=norbs, coord=2)

    # print(len(fsyst_sites),X.shape,ham.shape)
    id = sp.eye(np.shape(X)[0])

    # We define D as the following:
    D = (
        sp.kron(X - x0 * id, sigma_x)
        + sp.kron(Y - y0 * id, sigma_y)
        + sp.kron(Z - z0 * id, sigma_z)
    )

    list_invariant_realizations = []
    invariant_realization = 0
    list_localgap_realizations = []
    localgap_realization = 0
    density_subset_realizations = np.zeros(200, dtype=np.complex128)

    if W == 0:
        num_reals = 1

    seed_range = np.arange(num_reals)

    for ind, val in enumerate(seed_range):

        print("Averaging over realizations...")

        update_progress((ind + 1) / len(seed_range))
        seed = val
        np.random.seed(seed)
        print("seed W:", seed)

        # ANDERSON'S DISORDER
        if W != 0:
            disp = sp.diags(
                np.random.uniform(
                    low=-W / 2, 
                    high=W / 2, 
                    size=len(fsyst_sites))
            )
            AND_disorder = sp.kron(disp, sp.eye(norbs))

        if W == 0:
            h = ham
        else:
            h = ham + AND_disorder

        # Total Hamiltonian
        H = sp.kron(h - E0 * id, sigma_0)

        # Block of Localizer
        block_1 = H + 1j * kappa * D

        # Compute the determinant with mumps
        if compute_inv is True:
            # invariant = find_signature(H+1j*kappa*D)

            invariant = np.real(ctx.slogdet(block_1, ordering="scotch")[0])

            invariant_realization = invariant_realization + invariant
            print("Invariant realization:", invariant)
            list_invariant_realizations.append(invariant)

        # Compute the local gap with mumps
        if compute_localgap is True:
            L_sparse = (sp.kron(sigma_01, block_1)
                        + sp.kron(sigma_10, block_1.getH()))
            start_time2 = time.perf_counter()
            local_gap = eigsh(L_sparse, k=1, sigma=0)
            print("Local gap realization:", local_gap)
            end_time2 = time.perf_counter()
            print("Time:", end_time2 - start_time2)

            localgap_realization = localgap_realization + np.abs(local_gap)
            list_localgap_realizations.append(np.abs(local_gap))

        # Compute the Density of States of the Localizer with KPM
        if compute_DOS is True:
            start_time2 = time.perf_counter()
            bounds = (-1, 1)
            es = np.linspace(*bounds, 200)
            L_sparse = (sp.kron(sigma_01, block_1)
                        + sp.kron(sigma_10, block_1.getH()))
            # L_sparse = block_1
            # print(L_sparse.shape[0])
            spectrum = kwant.kpm.SpectralDensity(hamiltonian=L_sparse)
            spectrum.add_moments(energy_resolution=0.01)
            energy_subset = es
            density_subset = spectrum(energy_subset)
            DOS = np.real(density_subset) / np.max(np.real(density_subset))
            density_subset_realizations += np.array(DOS)
            # You get local gap for free!
            # print(len(DOS),DOS)
            localgap = find_dos_gap(energy_subset, DOS)
            localgap_realization += localgap
            list_localgap_realizations.append(localgap)
            end_time2 = time.perf_counter()
            print("Time:", end_time2 - start_time2)

    # We average over all realizations

    invariant_average = invariant_realization / num_reals
    localgap_average = localgap_realization / num_reals
    density_subset_average = density_subset_realizations / num_reals

    if compute_inv is True:
        print("Invariant:", invariant_average)
        return invariant_average, list_invariant_realizations

    if compute_DOS is True:
        return (
            energy_subset,
            density_subset_average,
            localgap_average,
            list_localgap_realizations,
        )

    if compute_localgap is True:

        print("Local gap:", np.abs(localgap_average[0]))
        return localgap_average, list_localgap_realizations

    if compute_inv is True and compute_localgap is True:
        print("Invariant:", invariant_average)
        print("Local gap:", np.abs(localgap_average[0]))
        return (
            invariant_average,
            localgap_average,
            list_invariant_realizations,
            list_localgap_realizations,
        )


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
