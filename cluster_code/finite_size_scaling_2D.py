# %%
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
from func_for_fig4 import param_obs_2b

# %%
# parallel variable
if len(sys.argv) > 1:
    arguments = sys.argv[1:]
    print("Arguments received", sys.argv[1])
    sigma = ast.literal_eval(arguments[0])
parname = 'sigma'

# lattice parameters
system_size = 30
sigma = sigma / system_size  # adjust to system size
bond_distance = 1.3 / system_size

# model parameters
A = 1.0
B = 1.0
Delta = 2
hadamard_disorder = 0.15
bond_power = 1
bond_lengthscale = 1 / system_size

# localiser parameter
kappa = 1

# DISORDER
kappa_shift = 0
beta = 1

W = 0

disorder_averages = 100

# ── output folder and checkpoint config ──────────────────────
SAVE_EVERY = 1
results_dir = Path(f'results_2d_v_{kappa_shift}')
results_dir.mkdir(exist_ok=True)

fname = results_dir / f'results_sigma{sigma}_num_reals_{disorder_averages}_L{system_size}.h5'

z2 = []
start_seed = 0
# %%
# ── resume from checkpoint if it exists ──────────────────────
if fname.exists():
    with h5py.File(fname, 'r') as f:
        if f'{parname}_{sigma}' in f:
            z2 = list(f[f'{parname}_{sigma}']['z2'][:])
            start_seed = len(z2)
            print(
                f"Resuming sigma={sigma} from seed {start_seed}/{disorder_averages}")


# this function calls a general save_checkpoint
# depending on the calculation we would change the expected_output
# and atributes passed to it


def checkpoint(z2):
    save_checkpoint(
        fname=fname,
        parallelized_variable=sigma,
        parallelized_variable_name=parname,
        expected_output={'z2': z2},
        atributes={
            'W': W, 'num_reals': disorder_averages, 'kappa': kappa,
            'L': system_size, 'bond_distance': bond_distance, 'A': A, 'B': B,
            'Delta': Delta, 'hadamard_disorder': hadamard_disorder,
            'bond_power': bond_power, 'bond_lengthscale': bond_lengthscale,
            'kappa_shift': kappa_shift, 'beta': beta, 'sigma': sigma,
        }
    )


# ── signal handler ────────────────────────────────────────────
def handle_signal(signum, frame):
    print(f"\nSignal {signum} received — saving checkpoint and exiting...")
    checkpoint(z2)
    sys_exit(0)


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT,  handle_signal)
# %%
# ── main loop ─────────────────────────────────────────────────
try:
    for seed in tqdm(range(start_seed, disorder_averages)):

        z2_seed = param_obs_2b(
            system_size=system_size,
            sigma=sigma,
            kappa_shift=kappa_shift,
            bond_distance=bond_distance,
            A=A,
            B=B,
            Delta=Delta,
            onsite_disorder=W,
            hadamard_disorder=hadamard_disorder,
            kappa_spec=kappa,
            disorder_average=1,  # we are averaging outside
            beta=beta,
            bond_power=bond_power,
            bond_lengthscale=bond_lengthscale,
        )  # just one disorder realisation at a time
        z2.append(z2_seed)

        if (seed + 1) % SAVE_EVERY == 0:
            checkpoint(z2)
            tqdm.write(
                f"Checkpoint saved ({seed + 1}/{disorder_averages} reals)")

finally:
    checkpoint(z2)
    print("Final save completed.")
