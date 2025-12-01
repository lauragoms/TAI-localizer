import numpy as np

sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

bhz_trs_operator = np.kron(np.eye(2), np.array([[0, 1], [-1, 0]]))