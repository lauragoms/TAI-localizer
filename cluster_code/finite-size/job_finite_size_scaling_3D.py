from koala import pointsets
from tqdm import tqdm
from save_files_in_cluster import save_checkpoint
import sys
from sys import exit as sys_exit
import h5py
import signal
import ast
from pathlib import Path
from func_for_finitesize_3D import params_obs_3D


# parallel variable and outputs
if len(sys.argv) > 1:
    arguments = sys.argv[1:]
    print("Arguments received", sys.argv[1])
    parallel_value = ast.literal_eval(arguments[0])

sigma = parallel_value
parname = 'sigma'


# lattice params
system_size = 22
num_realizations = 100
sigma = sigma / system_size

# initial points
initial_points = pointsets.grid(system_size, system_size, system_size)

# sys params
A = 1
MJ = 2
bond_lengthscale = 1 / system_size
bond_power = 1
# localizer params
kappa = 2
E0 = 0

# DISORDER
kappa_shift = 0 #3*bond_lengthscale
beta = 1
onsite_disorder = 0
resolution = 10


# ── output folder and checkpoint config ──────────────────────
SAVE_EVERY = 1
results_dir = Path(f'results_3d_v_{kappa_shift}')
results_dir.mkdir(exist_ok=True)

fname = results_dir / f'results_{parname}{parallel_value}_num_reals_{num_realizations}_L{system_size}.h5'


start_seed = 0

# ── resume from checkpoint if it exists ──────────────────────
if fname.exists():
    with h5py.File(fname, 'r') as f:
        if f'{parname}_{parallel_value}' in f:
            z2 = list(f[f'{parname}_{parallel_value}']['z2'][:]) # ADD LINES HERE FOR MORE EXPECTED OUTPUTS
            start_seed = len(z2)
            print(
                f"Resuming {parname}={parallel_value} from seed {start_seed}/{num_realizations}")


def checkpoint(z2): # ADD VARIABLES HERE FOR MORE EXPECTED OUTPUTS
    save_checkpoint(
        fname=fname,
        parallelized_variable=sigma,
        parallelized_variable_name=parname,
        expected_output={'z2': z2},  # ADD VARIABLES HERE FOR MORE EXPECTED OUTPUTS
        atributes={
            'MJ': MJ, 'A': A, 'onsite_disorder': onsite_disorder,
            'sigma': sigma, 'L': system_size,
            'kappa_shift': kappa_shift, 'beta': beta, 'resolution': resolution,
            'bond_lengthscale': bond_lengthscale, 'bond_power': bond_power,
            'kappa': kappa, 'E0': E0, 'num_reals': num_realizations,
        }
    )


# ── signal handler if job is canceled ────────────────────────────────────────────
def handle_signal(signum, frame):
    print(f"\nSignal {signum} received — saving checkpoint and exiting...")
    if z2:
        checkpoint(z2)
    sys_exit(0)


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT,  handle_signal)


# ── main loop ─────────────────────────────────────────────────
z2 = []
try:
    for seed in tqdm(range(start_seed, num_realizations)):
        z2_seed = params_obs_3D(
            points=initial_points,
            system_size=system_size,
            MJ=MJ,
            A=A,
            onsite_disorder=onsite_disorder,
            bond_lengthscale=bond_lengthscale,
            bond_power=bond_power,
            kappa_spec=kappa,
            E0=E0,
            sigma=sigma,
            kappa_shift=kappa_shift,
            beta=beta,
            resolution=resolution,
            seed=seed,
        )

        z2.append(z2_seed)
        print(z2_seed)

        if (seed + 1) % SAVE_EVERY == 0:
            checkpoint(z2)
            tqdm.write(
                f"Checkpoint saved ({seed + 1}/{num_realizations} reals)")

finally:
    checkpoint(z2)
    print("Final save completed.")
