import numpy as np
from tai_localizer.lauralizer.amorphous_model_BHZ_2D import amorph_hopping, amorph_BHZ
from tai_localizer.lauralizer.functions import bonds_func
Lx = 3
Ly = Lx

sites = [(x,y) for x in range(-Lx,Lx) for y in range(-Ly,Ly)] 
bonds = bonds_func(sites, 1.1)
syst = amorph_BHZ(sites, bonds)
sys_sites = syst.finalized().sites

def test_hoppings():
    site1 = sys_sites[0]
    site2 = sys_sites[1]
    A = 1
    B = 1
    bond_lengthscale = 1.0
    bond_power = 1.0
    t = amorph_hopping(site1, site2, A, B, bond_lengthscale, bond_power)
    t2 = amorph_hopping(site2, site1, A, B, bond_lengthscale, bond_power)
    assert np.allclose(t.conj().T, t2), "Hopping is not Hermitian!"


test_hoppings()