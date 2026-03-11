# This file needs mumps 0.0.6 or later for comm variable to work
import adaptive
from func_for_fig4 import param_obs_2b
from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI


comm = MPI.COMM_WORLD
print(comm.Get_rank(), comm.Get_size())

# lattice parameters
system_size = 30
bond_distance = 1.3 / system_size

# model parameters
A = 1.0
B = 1.0
hadamard_disorder = 0.15
bond_power = 1
bond_lengthscale = 1 / system_size

# localiser parameter
kappa = 1

# disorder for fig b)
onsite_disorder_b = 0
kappa_shift_b = 0
beta = 1

# adaptive params
sigma_bounds = (0.0, 0.1 / system_size)
delta_bounds = (-2, 4)

disorder_averages = 100


def goal(ps):
    return ps.npoints > 5000


def f(delta_sig):
    return param_obs_2b(
        system_size=system_size,
        sigma=delta_sig[1],
        kappa_shift=kappa_shift_b,
        bond_distance=bond_distance,
        A=A,
        B=B,
        Delta=delta_sig[0],
        onsite_disorder=onsite_disorder_b,
        hadamard_disorder=hadamard_disorder,
        kappa_spec=kappa,
        disorder_average=disorder_averages,
        beta=beta,
        bond_power=bond_power,
        bond_lengthscale=bond_lengthscale,
    )


if __name__ == "__main__":

    fname = "data_fig4b.pkl"
    learner_dis = adaptive.Learner2D(
        f,
        bounds=[
            delta_bounds,
            sigma_bounds,
        ],
    )
    # learner_dis.load(fname)

    runner_dis = adaptive.Runner(
        learner_dis,
        executor=MPIPoolExecutor(max_workers=comm.Get_size()-1),
        ntasks=comm.Get_size()-1,
        shutdown_executor=True,
        goal=goal,
    )

    # periodically save the data (in case the job dies)
    runner_dis.start_periodic_saving(dict(fname=fname), interval=60)

    # block until runner_dis goal reached
    runner_dis.block_until_done()

    # save one final time before exiting
    learner_dis.save(fname)
