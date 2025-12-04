import koala
from koala.pointsets import uniform, move_all_points, move_point
import numpy as np
from tqdm import tqdm


def koala_lattice_generator(L: int,
                            iters: int,
                            sigma: float = 0.03,
                            kappa: float = 0.03,
                            beta: float = 6):
    """
    Generates a lattice from iterating koala's function move_all_points
    
    :param L: Size of the crystalline square lattice
    :type L: int
    :param iters: Number of iterations for move_all_points
    :type iters: int
    :param sigma: Step size of the gaussian distribution around the original position
    :param kappa: Lengthscale of the repulsion between points
    :param beta: Temperature, controls the degree of randomness. Defaults to 6.
    :return: Each iteration of move_all_points
    :rtype: list
    """

    pos = np.linspace(0, 1, L, endpoint=False)
    pos += 0.5 * (pos[1] - pos[0])
    regular_lattice = np.reshape(np.meshgrid(pos, pos), [2, -1]).T
    pts_new = regular_lattice.copy()
    pts_history = [pts_new]
    for i in tqdm(range(iters)):
        pts_new = move_all_points(pts_new, sigma, kappa,
                                  beta, resolution=30)
        pts_history.append(pts_new)

    return pts_history