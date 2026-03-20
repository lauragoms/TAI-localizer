import h5py
import numpy as np
from pathlib import Path


def save_checkpoint(
        fname: Path,
        parallelized_variable: float,
        parallelized_variable_name: str,
        expected_output: dict,
        atributes: dict,
):
    par = parallelized_variable
    parname = parallelized_variable_name
    out = expected_output

    with h5py.File(fname, 'a') as f:
        if f'{parname}_{par}' in f:
            del f[f'{parname}_{par}']
        grp = f.create_group(f'{parname}_{par}')

        for key, value in out.items():
            grp.create_dataset(key, data=np.array(value))

        grp.attrs['par_variable_name'] = parname

        for key, value in atributes.items():
            grp.attrs[key] = value
