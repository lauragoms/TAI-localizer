import numpy as np
from tai_localizer.perulizer.bhz import _tx, _ty, _t_theta

def test_hoppings():
    A= 0*1.12
    B = 0.43
    alpha = 0.347

    tx = _tx(A, B, alpha)
    tx2 = _t_theta(A, B, alpha, 0)
    tx2_dag = _t_theta(A, B, alpha, np.pi)

    ty = _ty(A, B, alpha)
    ty2 = _t_theta(A, B, alpha, np.pi/2)
    ty2_dag = _t_theta(A, B, alpha, 3*np.pi/2)

    assert np.allclose(tx , tx2)
    assert np.allclose(ty , ty2)
    assert np.allclose(tx.conj().T , tx2_dag)
    assert np.allclose(ty.conj().T , ty2_dag)