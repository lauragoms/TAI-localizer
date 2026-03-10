import ast
import sys
import numpy as np
import pickle as pi
from funcs_for_kappa_3D import localgap_sys_3D


# if len(sys.argv) > 1:
#     arguments = sys.argv[1:]
#     print("Arguments received", sys.argv[1])

#     seed = ast.literal_eval(arguments[0])


# computational params
num_realizations = 1
kappa_values = np.linspace(0, 25, 50)
E_values = np.linspace(-2.5, 2.5, 50)


# lattice params
system_size = 10
bond_distance = 1.3 / system_size

# sys params
A = 1
MJ = 1.5
bond_lengthscale = 1 / system_size
bond_power = 1 / system_size

# localizer params
## kappa = 2
## E0 = 0

# onsite disorder
onsite_disorder = 0
seed = 0

# structural disorder params
sigma = 0
kappa_shift = 0
beta = 1
resolution = 10




## compute locgap
locgap_grid = {}
for kappa in kappa_values:
    locgap_grid[kappa] = []
    for E0 in E_values:
        locgap = localgap_sys_3D(
            MJ=MJ,
            A=A,
            onsite_disorder=onsite_disorder,
            seed=seed,
            bond_lengthscale=bond_lengthscale,
            bond_power=bond_power,
            bond_distance=bond_distance,
            system_size=system_size,
            kappa_spec=kappa,
            E0=E0,
            kappa=kappa_shift,
            sigma=sigma,
            kappa_shift=kappa_shift,
            beta=beta,
            resolution=resolution
        )
        locgap_grid[kappa].append(locgap)
        
        
## save
with open('locgap_grid_3D_kappa_E.pkl', 'wb') as f:
    pi.dump(locgap_grid, f)