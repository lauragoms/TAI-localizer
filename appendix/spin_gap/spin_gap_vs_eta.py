"""Spin gap of P s_z P as a function of the spin disorder eta (hadamard_disorder)."""
import numpy as np
from scipy import linalg as la
from koala import pointsets

from tai_localiser.perulizer import proximity_bonds, randomly_rotate
from tai_localiser.perulizer.lattices import iterative_move_point
from tai_localiser.lauralizer.amorphous_model_BHZ_2D import amorph_BHZ

# lattice parameters
system_size = 10
sigma = 0.5 / system_size
kappa_shift = 3 / system_size
bond_distance = 1.3 / system_size
bond_lengthscale = 1 / system_size
bond_power = 1

# model parameters
A = 1.0
B = 1.0
Delta = 2
W = 0

etas = np.linspace(0, 1, 21)
disorder_averages = 20


def spin_gaps(seed):
    """Spin gap of P s_z P for every eta, on one amorphous lattice."""
    rng = np.random.default_rng(seed)
    points = pointsets.grid(system_size, system_size)
    points = iterative_move_point(points, sigma, kappa_shift, rng=rng)

    edges, c = proximity_bonds(points, bond_distance)
    edges = edges[np.abs(c).sum(axis=1) == 0]  # open boundaries

    parameters = {
        "norbs": 4, "rng_W": rng, "Delta": Delta, "A": A, "B": B,
        "dis_onsite": W, "mu": 0,
        "bond_lengthscale": bond_lengthscale, "bond_power": bond_power,
    }
    fsyst = amorph_BHZ(points, edges).finalized()
    ham0 = fsyst.hamiltonian_submatrix(params=parameters)
    n_sites = len(fsyst.sites)

    # basis is site x orbital x spin; the conserved spin of amorph_BHZ
    # is tau_z sigma_z, and it is not rotated by the spin disorder
    s_z = np.kron(np.ones(n_sites), [1, -1, -1, 1])

    out = np.zeros((len(etas), 2))
    for i, eta in enumerate(etas):
        ham = ham0
        if eta > 0:
            u = randomly_rotate(n_sites, eta)
            ham = u.conj().T @ ham0 @ u

        e, v = la.eigh(ham)
        v_occ = v[:, e < 0]
        spin_spectrum = la.eigvalsh(v_occ.conj().T @ (s_z[:, None] * v_occ))
        out[i] = np.min(np.abs(spin_spectrum)), np.min(np.abs(e))
    return out


gaps = np.stack(
    [spin_gaps(seed) for seed in range(disorder_averages)], axis=1
)  # (eta, seed, [spin gap, energy gap])

np.savez(
    "spin_gap_vs_eta.npz", etas=etas, spin_gap=gaps[..., 0], energy_gap=gaps[..., 1],
    L=system_size, sigma=sigma, kappa_shift=kappa_shift, Delta=Delta, W=W,
)
for eta, g in zip(etas, gaps):
    print(f"eta={eta:.2f}  spin gap={g[:, 0].mean():.4f} ± {g[:, 0].std():.4f}"
          f"  min={g[:, 0].min():.4f}  E gap={g[:, 1].mean():.4f}")
