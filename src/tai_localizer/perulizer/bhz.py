import numpy as np
from scipy import linalg as la
from .misc import sigma_x, sigma_y, sigma_z
from koala.lattice import Lattice


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
    a_x = -1j * A / 2 * np.kron(sigma_x, sigma_z)
    alp_x = 1j * alpha / 2 * np.kron(np.eye(2), sigma_y)
    a_y = 1j * A / 2 * np.kron(sigma_y, np.eye(2))
    alp_y = -1j * alpha / 2 * np.kron(sigma_z, sigma_x)

    return b + np.cos(theta) * (a_x + alp_x) + np.sin(theta) * (a_y + alp_y)


def _onsite(Delta, B, ws, wp):
    a = (Delta - 4 * B) * np.kron(sigma_z, np.eye(2))
    b = np.kron(np.diag([ws, wp]), np.eye(2))

    return a + b


def bhz_ham(
    lattice: Lattice,
    A: float,
    B: float,
    alpha: float,
    Delta: float,
    ws_vals: np.ndarray,
    wp_vals: np.ndarray,
    hopping_lengthscale:np.ndarray = None,
    hopping_power: int = 1,
) -> np.ndarray:
    # check iterability
    try:
        iter(ws_vals)
    except TypeError:
        ws_vals = np.array([ws_vals] * lattice.n_vertices)

    try:
        iter(wp_vals)
    except TypeError:
        wp_vals = np.array([wp_vals] * lattice.n_vertices)

    angles = np.arctan2(*lattice.edges.vectors.T)
    distances = la.norm(lattice.edges.vectors, axis=1)
    if hopping_lengthscale is not None:
        hopping_strengths = np.exp(-((distances-hopping_lengthscale)/hopping_lengthscale)**hopping_power)
    else:
        hopping_strengths = np.ones(len(distances))

    hopping = np.zeros(
        [lattice.n_vertices * 4, lattice.n_vertices * 4], dtype=complex
    )
    for n in range(lattice.n_edges):
        i, j = lattice.edges.indices[n]
        angle = angles[n]
        bond_op = _t_theta(A, B, alpha, angle)*hopping_strengths[n]
        hopping[i * 4 : (i + 1) * 4, j * 4 : (j + 1) * 4] += bond_op
        hopping[j * 4 : (j + 1) * 4, i * 4 : (i + 1) * 4] += bond_op.conj().T

    onsite = np.zeros([lattice.n_vertices * 4, lattice.n_vertices * 4])
    for i in range(lattice.n_vertices):
        onsite[i * 4 : (i + 1) * 4, i * 4 : (i + 1) * 4] = _onsite(
            Delta, B, ws_vals[i], wp_vals[i]
        )

    return onsite + hopping


# TODO - make this use
# basis for this is position \otimes orbital \otimes spin
def bhz_ham_regular(square_lat, A, B, alpha, Delta, ws_vals, wp_vals):

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
