import h5py
import numpy as np
from pathlib import Path


def load_cluster_results(results_dir, fname_pattern):
    """
    Loads all HDF5 result files matching fname_pattern from results_dir.
    Returns a dict with all datasets sorted by the parallelized variable,
    plus metadata from attrs.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing the result files
    fname_pattern : str
        Glob pattern to match files, e.g. 'results_W*_Lx_10*.h5'

    Returns
    -------
    dict with:
        - '{par_name}_values' : np.array of parallelized variable values,
            sorted
        - one key per dataset found (e.g. 'dos', 'z2'), shape (n_W, ...)
        - 'attrs' : dict of metadata from the first file
    """
    results_dir = Path(results_dir)
    files = sorted(results_dir.glob(fname_pattern))

    if not files:
        raise FileNotFoundError(
            f"No files found matching {fname_pattern} in {results_dir}")

    par_values = []
    par_name = None
    data = {}
    rest = []
    non_completed_runs = []
    for j, f in enumerate(files):
        with h5py.File(f, 'r') as hf:
            # read group in the file
            key = list(hf.keys())[0]
            grp = hf[key]

            # read parallelized variable name from attrs
            par_name = grp.attrs['par_variable_name']
            # num_reals = grp.attrs['num_reals']
            num_reals = grp.attrs['num_reals']
            par_values.append(grp.attrs[par_name])
            # load all members of the group (outputs of calculation)
            for i, dataset_name in enumerate(grp.keys()):
                if dataset_name not in data:
                    data[dataset_name] = []
                # check if all realizations are present for the first parameter value
                if i==0:
                    if len(grp[dataset_name]) != num_reals:
                        rest.append(grp.attrs[par_name])
                        print(
                            f"Warning: Parameter {grp.attrs[par_name]} removed, it has only {len(grp[dataset_name])} / {num_reals} realizations")
                        non_completed_runs.append([j, grp.attrs[par_name]])
                data[dataset_name].append(grp[dataset_name][:])
    # remove datasets with non completed runs
    for idx, param in sorted(
        non_completed_runs,
        key=lambda x: x[0],
        reverse=True,  # remove from the end to avoid messing up indices
    ):
        par_values.remove(param)
        for _, value in data.items():
            value.pop(idx)
    # sort everything by parallelized variable

    idx = np.argsort(par_values)

    results = {f'{par_name}_values': np.array(par_values)[idx]}
    for dataset_name, values in data.items():
        # for elem in values:
            # print(len(elem))
        # for i, val in enumerate(values):
        #     num_reals = 50 

        #     if len(val) != num_reals:
        #         print(f"Parameter {np.array(par_values)[idx][i]} has averaged over only {len(val)} / {num_reals} realizations")
        results[dataset_name] = np.array(values)[idx]


    # metadata from first file
    with h5py.File(files[0], 'r') as hf:
        key = list(hf.keys())[0]
        results['attrs'] = dict(hf[key].attrs)

    print(f"Loaded {len(files)} files from {results_dir}")
    print(
        f"{par_name} range: [{np.min(par_values):.3f},{np.max(par_values):.3f}]")
    print("Parameters for the calculation:")
    for elem in results['attrs']:
        print(f"  {elem}: {results['attrs'][elem]}")
    print("Outputs obtained from the calculation:", list(data.keys()))

    return results, rest
