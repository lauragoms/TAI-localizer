# %%
import kwant
import scipy.sparse as sp
import numpy as np

from tai_localiser.lauralizer.amorphous_model_3D import (
    amorph_3DTI
    )
from tai_localiser.lauralizer.functions import bonds_func
from tai_localiser.lauralizer.amorphous_model_BHZ_2D import amorph_BHZ
from koala import pointsets
from tai_localiser.lauralizer.functions import eigsh
import matplotlib.pyplot as plt

# %% lattice params
system_size = 10

bond_distance = 1.3 / system_size

points_2D = pointsets.grid(system_size, system_size)
bonds_2D = bonds_func(points_2D, bond_distance)

points_3D = pointsets.grid(system_size, system_size, system_size)
bonds_3D = bonds_func(points_3D, bond_distance)


# %% create system

sys_2D = amorph_BHZ(points_2D, bonds_2D)
sys_3D = amorph_3DTI(points_3D, bonds_3D)

# %% and plot it
kwant.plot(sys_2D.finalized(), site_size=0.1, hop_lw=0.05)
kwant.plot(sys_3D.finalized(), site_size=0.1, hop_lw=0.05) 

# %% sys params
A = 1
onsite_disorder = 0
rng = np.random.default_rng(0)

# 2D
B = 1
M = 1
Delta = M + 4*B
mu = 0

# 3D
MJ = 1.5

bond_lengthscale = 1 / system_size
bond_power_2D = 2
bond_power_3D = 1 / system_size

params_2D = {
    'norbs': 4,
    'Delta': Delta,
    'B': B,
    'A': A,
    'bond_lengthscale': bond_lengthscale,
    'bond_power': bond_power_2D,
    'dis_onsite': onsite_disorder,
    'rng_W': rng,  # not used when dis_onsite=0
    'mu': mu,
}

params_3D = {
    'MJ': MJ,
    'A': A,
    'bond_lengthscale': bond_lengthscale,
    'bond_power': bond_power_3D,
    'dis_onsite': onsite_disorder,
    'rng_W': rng,  # not used when dis_onsite=0
}

ham_2D = sys_2D.finalized().hamiltonian_submatrix(
    params=params_2D, sparse=True)
ham_3D = sys_3D.finalized().hamiltonian_submatrix(
    params=params_3D, sparse=True)

# %% compute spectrum
spec_2D = eigsh(ham_2D, k=100, sigma=0)
spec_3D = eigsh(ham_3D, k=100, sigma=0)


# %% plot to see bulk gap

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(spec_2D, bins=50, color='black')
plt.title('2D spectrum')
plt.subplot(1, 2, 2)
plt.hist(spec_3D, bins=50, color='black')
plt.title('3D spectrum')
plt.show()

# %%
