"""Spin Bott index vs Delta in the clean crystalline BHZ (current, rotated basis),
using either tau_0 s_z or tau_z s_z as the spin operator."""
import numpy as np
from scipy import linalg as la
from koala import example_graphs as eg
from tai_localiser.perulizer import bhz_ham, spin_chern_marker

A = 1.0
B = 1.0
deltas = np.linspace(-2, 10, 61)
system_sizes = [10, 20]
spin_operators = {
    "tau0_sz": np.array([1, -1, 1, -1]),
    "tauz_sz": np.array([1, -1, -1, 1]),
}

results = {}
for L in system_sizes:
    lattice = eg.square_lattice(L, L)
    for name, s_z in spin_operators.items():
        bott = np.zeros(len(deltas))
        gap = np.zeros(len(deltas))
        for i, delta in enumerate(deltas):
            ham = bhz_ham(lattice, A, B, 0.0, delta, 0.0, 0.0)
            e, v = la.eigh(ham)
            proj = v @ np.diag(e < 0) @ v.conj().T
            c, g = spin_chern_marker(lattice, proj, s_z=s_z, fix=True, return_spin_gap=True)
            bott[i], gap[i] = -c, g  # same sign convention as fig_2.ipynb
        results[f"bott_{name}_L{L}"] = bott
        results[f"gap_{name}_L{L}"] = gap

np.savez("spin_bott_vs_delta.npz", deltas=deltas, **results)
