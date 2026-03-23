import ast
import sys
import numpy as np
import pickle as pi
from funcs_for_kappa_3D import localgap_sys_3D
from save_files_in_cluster import save_checkpoint
import h5py
from sys import exit as sys_exit
import signal
from tqdm import tqdm
from pathlib import Path


if len(sys.argv) > 1:
    arguments = sys.argv[1:]
    print("Arguments received", sys.argv[1])

    parallel_value = ast.literal_eval(arguments[0])

seed = parallel_value
parname = 'seed'

# computational params
M_values = np.linspace(0, 4, 50)
W_values = np.linspace(0, 10, 50)


# lattice params
system_size = 10
bond_distance = 1.3 / system_size

# sys params
A = 1

bond_lengthscale = 1 / system_size
bond_power = 1 / system_size

# localizer params
kappa = 2
E0 = 0

# structural disorder params
sigma = 0
kappa_shift = 0
beta = 1
resolution = 10


# save checkpoinmt function
results_dir = Path('results_3d_bc')
results_dir.mkdir(exist_ok=True)
fname = results_dir / f'results_{parname}{parallel_value}_kappa_{kappa}_L{system_size}.h5'

if fname.exists():
    with h5py.File(fname, 'r') as f:
        key = f'{parname}_{parallel_value}'
        if key in f:
            local_gap_grid = list(f[key]['local_gap_grid'])
            completed_MJ = len(local_gap_grid)
            M_values = M_values[completed_MJ:]
            print(f"Resuming from MJ index {completed_MJ}/{len(M_values)}")


def checkpoint(local_gap_grid):  # ADD VARIABLES HERE FOR MORE EXPECTED OUTPUTS
    save_checkpoint(
        fname=fname,
        parallelized_variable=parallel_value,
        parallelized_variable_name=parname,
        expected_output={'local_gap_grid': local_gap_grid},
        atributes={
            'Mmin': M_values[0],
            'Mmax': M_values[-1],
            'Wmin': W_values[0],
            'Wmax': W_values[-1],
            'system_size': system_size,
            'A': A,
            'bond_lengthscale': bond_lengthscale,
            'bond_power': bond_power,
            'seed': seed,
            'kappa_shift': kappa_shift,
            'beta': beta,
            'resolution': resolution,
            'kappa': kappa,
        }
    )


local_gap_grid = []


# ── signal handler ────────────────────────────────────────────
def handle_signal(signum, frame):
    print(f"\nSignal {signum} received — saving checkpoint and exiting...")
    if local_gap_grid:
        checkpoint(local_gap_grid)
    sys_exit(0)


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT,  handle_signal)


try:
    for i, MJ in tqdm(enumerate(M_values)):
        local_gap_grid.append([])
        for onsite_disorder in W_values:
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
            local_gap_grid[i].append(locgap)
        checkpoint(local_gap_grid)
        tqdm.write(
            f"Checkpoint saved for MJ={MJ} ({i + 1}/{len(M_values)} MJ values)")  
finally:
    if local_gap_grid:
        checkpoint(local_gap_grid)
    print("Final save completed.")
