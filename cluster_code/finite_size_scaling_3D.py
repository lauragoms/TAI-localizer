from tqdm import tqdm
from save_files_in_cluster import save_checkpoint
import sys
from sys import exit as sys_exit
import h5py
import signal
import ast
from pathlib import Path
folder = Path.cwd()
sys.path.append(str(folder.parent))
from func_for_fig5 import params_obs_3D


# parallel variable
if len(sys.argv) > 1:
    arguments = sys.argv[1:]
    print("Arguments received", sys.argv[1])
    sigma = ast.literal_eval(arguments[0])
parname = 'sigma'

# lattice params
system_size = 10
sigma_bounds = (0, 0.05 / system_size)
num_realizations = 100

# sys params
A = 1
MJ = 2
bond_lengthscale = 1 / system_size
bond_power = 1 / system_size
# localizer params
kappa = 2
E0 = 0

# DISORDER
kappa_shift = 0
beta = 1
onsite_disorder = 0
resolution = 10






params_obs_3D(
        system_size=system_size,
        MJ=MJ,
        A=A,
        onsite_disorder=onsite_disorder,
        disorder_average=num_realizations,
        bond_lengthscale=bond_lengthscale,
        bond_power=bond_power,
        kappa_spec=kappa,
        E0=E0,
        sigma=sigma,
        kappa_shift=kappa_shift,
        beta=beta,
        resolution=resolution,
    )