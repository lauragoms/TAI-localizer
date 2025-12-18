import numpy as np
from scipy import sparse as sp
from scipy.stats import unitary_group

sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
sigma_0 = np.array([[1, 0], [0, 1]])
hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

bhz_trs_operator = np.kron(np.eye(2), np.array([[0, 1], [-1, 0]]))
random_unitary = unitary_group(2)



def randomly_rotate(n_vertices, proportion, sparse = False):
    rng = np.random.default_rng()
    n = n_vertices * 2
    n_ones = np.linspace(0, 1, n) < proportion
    diag_rot = rng.permutation(n_ones)
    had_part = np.kron(np.diag(diag_rot), hadamard)
    eye_part = np.kron(np.diag(1 - diag_rot), np.eye(2))

    if sparse:
        had_part = sp.csr_matrix(had_part)
        eye_part = sp.csr_matrix(eye_part)
    
    return had_part + eye_part
