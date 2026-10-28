"""Spin Bott index (tau_z s_z) vs Delta for several spin disorders eta,
clean crystalline BHZ in the current (rotated) basis, W = 0."""
import numpy as np
from scipy import linalg as la
from koala import example_graphs as eg
from tai_localiser.perulizer import bhz_ham, spin_chern_marker, randomly_rotate

A = 1.0
B = 1.0
L = 10
deltas = np.linspace(-2, 10, 31)
etas = np.array([0, 0.4, 0.6, 0.8, 0.9, 1.0])
disorder_averages = 20
s_z = np.array([1, -1, -1, 1])  # tau_z s_z, conserved in this basis

lattice = eg.square_lattice(L, L)
bott = np.zeros((len(etas), len(deltas), disorder_averages))
gap = np.zeros_like(bott)

for i, eta in enumerate(etas):
    for j, delta in enumerate(deltas):
        ham0 = bhz_ham(lattice, A, B, 0.0, delta, 0.0, 0.0)
        for k in range(disorder_averages if eta > 0 else 1):
            u = randomly_rotate(lattice.n_vertices, eta)
            ham = u @ ham0 @ u.conj().T  # as in fig_2.ipynb
            e, v = la.eigh(ham)
            proj = v @ np.diag(e < 0) @ v.conj().T
            c, g = spin_chern_marker(lattice, proj, s_z=s_z, fix=True, return_spin_gap=True)
            bott[i, j, k], gap[i, j, k] = -c, g
        if eta == 0:  # no randomness, one realisation is enough
            bott[i, j, :], gap[i, j, :] = bott[i, j, 0], gap[i, j, 0]
    print(f"eta={eta:.1f} done", flush=True)

np.savez("spin_bott_vs_delta_eta_L10.npz", deltas=deltas, etas=etas, bott=bott, gap=gap)
