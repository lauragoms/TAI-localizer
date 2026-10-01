# %%
import os
import argparse
from typing import Any
from pathlib import Path
import numpy as np
import h5py
from koala import pointsets
from func_for_finitesize_2D import param_obs_2b
from save_files_in_cluster import save_checkpoint


# ── command line arguments ───────────────────────────────────
# Z2 (Pfaffian sign of the 2D spectral localizer) vs structural disorder sigma
# for the amorphous BHZ model, at fixed bond cutoff R and Anderson disorder W.
#
# The SLURM array parallelizes sigma: each array task picks a single
# sigma = SIGMA_VALUES[sigma_index] (in units of the lattice spacing r0) and
# sweeps the disorder realizations. R and W are fixed for the whole array, so a
# run is launched with e.g.:
#     sbatch submit.sh --R 1.15          (cutoff scan, W=0)
#     sbatch submit.sh --W 4             (Anderson scan, R=1.3)
parser = argparse.ArgumentParser(
    description="Z2 of the 2D amorphous BHZ model vs sigma, parallelizing "
                "sigma over the SLURM array."
)
parser.add_argument("--L", type=int, default=20,
                    help="Linear system size (L x L grid, default 20).")
parser.add_argument("--R", type=float, default=1.3,
                    help="Bond cutoff in units of r0 = 1/L (default 1.3).")
parser.add_argument("--W", type=float, default=0.0,
                    help="Anderson disorder: onsite uniform in [-W/2, W/2] "
                         "times identity (default 0).")
parser.add_argument("--kshift", type=float, default=3.0,
                    help="Repulsion kappa_shift in units of r0 (default 3).")
parser.add_argument("--Delta", type=float, default=2.0,
                    help="BHZ mass parameter Delta (default 2).")
parser.add_argument("--reals", type=int, default=100,
                    help="Number of disorder realizations (default 100).")
parser.add_argument("--sigma-min", type=float, default=0.0,
                    help="Minimum sigma in units of r0 (default 0).")
parser.add_argument("--sigma-max", type=float, default=0.5,
                    help="Maximum sigma in units of r0 (default 0.5).")
parser.add_argument("--num-sigma", type=int, default=50,
                    help="Number of sigma values (default 50). "
                         "The SLURM array size must match this.")
parser.add_argument("--sigma-index", type=int, default=None,
                    help="Index into SIGMA_VALUES for this array task. "
                         "Defaults to $SLURM_ARRAY_TASK_ID.")
args = parser.parse_args()

sigma_index = args.sigma_index
if sigma_index is None:
    sigma_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

parname = "sigma"


# %%
L = args.L
r0 = 1 / L  # lattice spacing (the grid lives in the unit square)

SYSTEM_PARAMS = {
    "L": L,
    "R_cutoff": args.R,              # in units of r0
    "bond_distance": args.R * r0,    # absolute cutoff passed to proximity_bonds
}

MODEL_PARAMS = {
    "A": 1.0,
    "B": 1.0,
    "Delta": args.Delta,
    "W": args.W,
    "hadamard_disorder": 0.15,
    "bond_power": 1,
    "bond_lengthscale": r0,
}

DISORDER_PARAMS = {
    "kappa_shift_r0": args.kshift,         # in units of r0
    "kappa_shift": args.kshift * r0,       # absolute value
    "beta": 1,
    "n_iterations_move": 3,                # default of iterative_move_point
}

LOCALIZER_PARAMS = {
    "kappa": 1,
    "E0": 0.0,
}

SIMULATION_PARAMS = {
    "num_reals": args.reals,
    "sigma_bounds_r0": (args.sigma_min, args.sigma_max),
    "num_sigma": args.num_sigma,
}

# sigma grid in units of r0; the SLURM array indexes it, so the array size
# (see submit.sh) must match num_sigma.
SIGMA_VALUES = np.linspace(
    *SIMULATION_PARAMS["sigma_bounds_r0"],
    num=SIMULATION_PARAMS["num_sigma"],
)

sigma_r0 = float(SIGMA_VALUES[sigma_index])
sigma = sigma_r0 * r0  # absolute sigma, as in the old finite-size files
print(f"L={L}, R={args.R} r0, W={args.W}, kshift={args.kshift} r0, "
      f"sigma_index={sigma_index}, sigma={sigma_r0} r0")


def all_attrs() -> dict[str, Any]:
    attrs = {}
    attrs.update(SYSTEM_PARAMS)
    attrs.update(MODEL_PARAMS)
    attrs.update(DISORDER_PARAMS)
    attrs.update(LOCALIZER_PARAMS)
    attrs.update(SIMULATION_PARAMS)
    attrs["sigma"] = sigma
    attrs["sigma_r0"] = sigma_r0
    attrs["sigma_values_r0"] = SIGMA_VALUES
    attrs["seeds"] = "realization j uses np.random.default_rng(j), j=0..num_reals-1"
    return attrs


# %%
# ── output folder and checkpoint config ──────────────────────
results_dir = Path(
    f"results_2d_L{L}_kshift{args.kshift}r0_R{args.R}_W{args.W}_Delta{args.Delta}"
    f"_sigma{args.sigma_min}-{args.sigma_max}"
)
try:
    os.makedirs(results_dir, exist_ok=True)
except FileExistsError:
    pass  # concurrent array tasks race to create the dir on Lustre
fname = results_dir / f"sigma_{sigma_r0}.h5"


def checkpoint(z2):
    save_checkpoint(
        fname=fname,
        parallelized_variable=sigma,
        parallelized_variable_name=parname,
        expected_output={"z2": z2},
        atributes=all_attrs(),
    )


def load_existing_z2():
    """Return the z2 realizations already stored for this sigma, or [].

    Realization j always uses seed j, so the first n stored values can be
    reused and only seeds n..num_reals-1 are computed.
    """
    if not fname.exists():
        return []
    try:
        with h5py.File(fname, "r") as f:
            group_name = f"{parname}_{sigma}"
            if group_name in f and "z2" in f[group_name]:
                return list(f[group_name]["z2"][()])
    except (OSError, KeyError, RuntimeError) as e:
        print(f"Could not read existing {fname} ({type(e).__name__}: {e}); "
              "recomputing from scratch.")
    return []


# %%
initial_points = pointsets.grid(L, L)

n_target = SIMULATION_PARAMS["num_reals"]
z2 = load_existing_z2()
n_done = len(z2)

if n_done >= n_target:
    print(f"Already have {n_done} >= {n_target} realizations for "
          f"sigma={sigma_r0} r0; nothing to do.")
else:
    print(f"Reusing {n_done} realizations, computing seeds "
          f"{n_done}..{n_target - 1}.")
    for seed in range(n_done, n_target):
        z2_seed = param_obs_2b(
            points=initial_points,
            system_size=L,
            sigma=sigma,
            kappa_shift=DISORDER_PARAMS["kappa_shift"],
            bond_distance=SYSTEM_PARAMS["bond_distance"],
            A=MODEL_PARAMS["A"],
            B=MODEL_PARAMS["B"],
            Delta=MODEL_PARAMS["Delta"],
            onsite_disorder=MODEL_PARAMS["W"],
            seed=seed,
            hadamard_disorder=MODEL_PARAMS["hadamard_disorder"],
            kappa_spec=LOCALIZER_PARAMS["kappa"],
            beta=DISORDER_PARAMS["beta"],
            bond_power=MODEL_PARAMS["bond_power"],
            bond_lengthscale=MODEL_PARAMS["bond_lengthscale"],
        )
        z2.append(z2_seed)
        checkpoint(z2)  # per-realization checkpoint
        print(f"seed {seed}: z2={z2_seed}  ({len(z2)}/{n_target})", flush=True)

checkpoint(z2)
print(f"Saved {len(z2)} realizations -> {fname}")
