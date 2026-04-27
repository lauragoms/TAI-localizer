import numpy as np
from koala.lattice import Lattice
from koala.pointsets import move_all_points

def iterative_move_point(points_in, sigma,  kappa, n_iterations=3, **kwargs):

    sigma_step = sigma / np.sqrt(n_iterations)
    kappa_step = kappa / n_iterations

    points = points_in.copy()
    for _ in range(n_iterations):
        points = move_all_points(points, sigma_step, kappa_step, **kwargs)

    return points

# same function as bonds_2D in laura's code
def proximity_bonds(positions: np.ndarray, distance_threshold: float) -> tuple:
    """Generates a lattice from a pointset by connecting every pair of vertices that is closer than a threshold

    Args:
        positions (np.ndarray): The positions of the vertices
        distance_threshold (float): The maximum distance to form a bond

    Returns:
        tuple: (bonds, crossing) for making a lattice
    """

    px = positions[:, 0]
    py = positions[:, 1]

    px_diff = px[None, :] - px[:, None]
    py_diff = py[None, :] - py[:, None]

    px_normed = (px_diff + 0.5) % 1 - 0.5
    py_normed = (py_diff + 0.5) % 1 - 0.5

    px_cross = (px_diff + 0.5) // 1
    py_cross = (py_diff + 0.5) // 1

    distances = px_normed**2 + py_normed**2
    valid = distances < distance_threshold**2
    valid = np.triu(valid, 1)

    edges = np.array(np.where(valid == 1)).T
    crossing_x = -px_cross[np.where(valid == 1)].astype(int)
    crossing_y = -py_cross[np.where(valid == 1)].astype(int)

    crossing = np.array([crossing_x, crossing_y]).T

    return edges, crossing


def proximity_lattice(positions, distance_threshold):
    """wrapper for proximity_bonds that returns a lattice object"""
    return Lattice(positions, *proximity_bonds(positions, distance_threshold))