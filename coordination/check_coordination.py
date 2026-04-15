# %%
import numpy as np
from koala import pointsets
from tai_localiser.lauralizer.functions import bonds_func


# %%
system_size = {
    '2D': 70,
    '3D': 22,
}

kappa_shift = 0
beta = 1
resolution = 10

sigma_range = {
    '2D': np.linspace(0, 0.1, 50),
    '3D': np.linspace(0, 0.05, 50),
}

# %%
points = pointsets.grid(system_size, system_size)
points = pointsets.move_all_points(
            points, sigma, kappa_shift, beta,
            rng=__import__('numpy').random.default_rng(int(s))
        )
edges, c = proximity_bonds(points, bond_distance)
