import ast
import sys
from pathlib import Path
import numpy as np
from funcs_for_kappa_3D import localgap_sys_3D
from save_files_in_cluster import save_checkpoint
import h5py
from sys import exit as sys_exit
import signal
from tqdm import tqdm


if len(sys.argv) > 1:
    arguments = sys.argv[1:]
    print("Arguments received", sys.argv[1])

    parallel_value = ast.literal_eval(arguments[0])

kappa = parallel_value
parname = 'kappa'
# computational params
E_values = np.linspace(-2.5, 2.5, 3)


# lattice params
system_size = 10
bond_distance = 1.3 / system_size

# sys params
A = 1
MJ = 1.5
bond_lengthscale = 1 / system_size
bond_power = 1 / system_size


# onsite disorder
onsite_disorder = 0
seed = 0

# structural disorder params
sigma = 0
kappa_shift = 0
beta = 1
resolution = 10


# ── output folder and checkpoint config ──────────────────────
SAVE_EVERY = 1
results_dir = Path('results_3d_a')
results_dir.mkdir(exist_ok=True)

fname = results_dir / f'results_{parname}{parallel_value}_E0_{len(E_values)}_L{system_size}.h5'


idx_start = 0
local_gap = []


# ── resume from checkpoint if it exists ──────────────────────
if fname.exists():
    with h5py.File(fname, 'r') as f:
        if f'{parname}_{parallel_value}' in f:
            local_gap = list(f[f'{parname}_{parallel_value}']['local_gap'][:]) # ADD LINES HERE FOR MORE EXPECTED OUTPUTS
            idx_start = len(local_gap)
            print(
                f"Resuming {parname}={parallel_value} from seed {idx_start}/{len(E_values)}")


def checkpoint(local_gap):  # ADD VARIABLES HERE FOR MORE EXPECTED OUTPUTS
    save_checkpoint(
        fname=fname,
        parallelized_variable=parallel_value,
        parallelized_variable_name=parname,
        expected_output={'local_gap': local_gap},
        atributes={
            'Emin': E_values[0],
            'Emax': E_values[-1],
            'num_reals': len(E_values),
            'system_size': system_size,
            'A': A,
            'MJ': MJ,
            'bond_lengthscale': bond_lengthscale,
            'bond_power': bond_power,
            'onsite_disorder': onsite_disorder,
            'seed': seed,
            'kappa_shift': kappa_shift,
            'beta': beta,
            'resolution': resolution,
            'kappa': kappa,
        }
    )


# ── signal handler ────────────────────────────────────────────
def handle_signal(signum, frame):
    print(f"\nSignal {signum} received — saving checkpoint and exiting...")
    if local_gap:
        checkpoint(local_gap)
    sys_exit(0)


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT,  handle_signal)

# ── main loop ─────────────────────────────────────────────────


try:
    for idx in tqdm(range(idx_start, len(E_values))):
        E0 = E_values[idx]
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
        local_gap.append(locgap)

        if (idx + 1) % SAVE_EVERY == 0:
            checkpoint(local_gap)
            tqdm.write(
                f"Checkpoint saved ({idx + 1}/{len(E_values)} reals)")
finally:
    if local_gap:
        checkpoint(local_gap)
    print("Final save completed.")
