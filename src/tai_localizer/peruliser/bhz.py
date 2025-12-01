import numpy as np
from scipy import linalg as la
from misc import sigma_x,sigma_y, sigma_z


def _tx(A, B, alpha):
    b = B * np.kron(sigma_z, np.eye(2))
    a = -1j * A / 2 * np.kron(sigma_x, sigma_z)
    alp = 1j * alpha / 2 * np.kron(np.eye(2), sigma_y)

    return a + b + alp


def _ty(A, B, alpha):
    b = B * np.kron(sigma_z, np.eye(2))
    a = 1j * A / 2 * np.kron(sigma_y, np.eye(2))
    alp = -1j * alpha / 2 * np.kron(sigma_z, sigma_x)

    return a + b + alp


def _t_theta(A, B, alpha, theta):

    b = B * np.kron(sigma_z, np.eye(2))
    ax = -1j * A / 2 * np.kron(sigma_x, sigma_z)
    alpx = 1j * alpha / 2 * np.kron(np.eye(2), sigma_y)
    ay = 1j * A / 2 * np.kron(sigma_y, np.eye(2))
    alpy = -1j * alpha / 2 * np.kron(sigma_z, sigma_x)

    return b + np.cos(theta)*(ax + alpx) + np.sin(theta)*(ay + alpy)


def _onsite(Delta, B, ws, wp):
    a = (Delta - 4 * B) * np.kron(sigma_z, np.eye(2))
    b = np.kron(np.diag([ws, wp]), np.eye(2))

    return a + b



# basis for this is position \otimes orbital \otimes spin
def bhz_ham(square_lat, A, B, alpha, Delta, ws_vals, wp_vals):

    # check iterability
    try:
        iter(ws_vals)
    except TypeError:
        ws_vals = np.array([ws_vals] * square_lat.n_vertices)

    try:
        iter(wp_vals)
    except TypeError:
        wp_vals = np.array([wp_vals] * square_lat.n_vertices)

    # construct hopping parts
    tx = _tx(A, B, alpha)
    ty = _ty(A, B, alpha)

    angle_labels = (
        np.rint(2 * np.arctan2(*square_lat.edges.vectors.T) / np.pi).astype(int) % 4
    )

    edges = square_lat.edges.indices

    x_plus = edges[np.where(angle_labels == 0)]
    y_plus = edges[np.where(angle_labels == 1)]
    x_minus = edges[np.where(angle_labels == 2)]
    y_minus = edges[np.where(angle_labels == 3)]

    z = np.zeros([square_lat.n_vertices, square_lat.n_vertices])

    m_x_plus = z.copy()
    m_x_plus[x_plus[:, 0], x_plus[:, 1]] = 1
    m_x_minus = z.copy()
    m_x_minus[x_minus[:, 0], x_minus[:, 1]] = 1
    m_y_plus = z.copy()
    m_y_plus[y_plus[:, 0], y_plus[:, 1]] = 1
    m_y_minus = z.copy()
    m_y_minus[y_minus[:, 0], y_minus[:, 1]] = 1

    hopping = (
        np.kron(m_x_plus, tx)
        + np.kron(m_x_minus, tx.conj().T)
        + np.kron(m_y_plus, ty)
        + np.kron(m_y_minus, ty.conj().T)
    )
    hopping = hopping + hopping.conj().T

    onsite = np.zeros([square_lat.n_vertices * 4, square_lat.n_vertices * 4])
    for i in range(square_lat.n_vertices):
        onsite[i * 4 : (i + 1) * 4, i * 4 : (i + 1) * 4] = _onsite(
            Delta, B, ws_vals[i], wp_vals[i]
        )

    return onsite + hopping