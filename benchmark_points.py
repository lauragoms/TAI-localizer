import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    from scipy import linalg as la
    from matplotlib import pyplot as plt
    import matplotlib

    import adaptive

    from tai_localiser.lauralizer import amorphous_model_BHZ_2D as am
    from tai_localiser.lauralizer.amorphous_model_BHZ_2D import amorph_BHZ
    from tai_localiser.lauralizer.localizer import (
        spectral_localizer_AII2D,
        pfaff_sign,
        local_gap_localizer,
    )
    from tai_localiser.perulizer import proximity_lattice, proximity_bonds

    from koala.lattice import Lattice
    from koala import pointsets
    from koala import plotting as pl
    import kwant as k

    # local imports
    from func_for_fig2 import param_obs_2b


    return (pointsets,)


@app.cell
def _(pointsets):
    # lattice parameters
    system_size = 10
    beta = 1
    initial_points = pointsets.grid(system_size, system_size, system_size)
    kappa = 0.3
    sigma = 0.1

    pointsets.move_all_points(initial_points, sigma, kappa, beta,resolution = 15, verbose=True)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
