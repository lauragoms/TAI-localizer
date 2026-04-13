# This file needs mumps 0.0.6 or later for comm variable to work
import adaptive
from func_for_fig5 import params_obs_3D
from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI


comm = MPI.COMM_WORLD
print(comm.Get_rank(), comm.Get_size())

MJ_bounds = (0, 4)

# lattice params
system_size = 10
sigma_bounds = (0, 0.05 / system_size)
num_realizations = 100

# sys params
A = 1
bond_lengthscale = 1 / system_size
bond_power = 1 #/ system_size
onsite_disorder = 0

# localizer params
kappa = 2
E0 = 0

# structural disorder params
kappa_shift = 0.3
beta = 1
resolution = 10


def goal(ps):
    return ps.npoints > 5000


def f(sigma_MJ):

    return params_obs_3D(
        system_size=system_size,
        MJ=sigma_MJ[0],
        A=A,
        onsite_disorder=onsite_disorder,
        disorder_average=num_realizations,
        bond_lengthscale=bond_lengthscale,
        bond_power=bond_power,
        kappa_spec=kappa,
        E0=E0,
        sigma=sigma_MJ[1],
        kappa_shift=kappa_shift,
        beta=beta,
        resolution=resolution,
        comm=MPI.COMM_SELF,  # mumps parameter for multithreading
    )


if __name__ == "__main__":
    fname = "data_fig5c.pkl"
    learner_dis = adaptive.Learner2D(
        f,
        bounds=[MJ_bounds, sigma_bounds],
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
    runner_dis.start_periodic_saving(dict(fname=fname), interval=300)

    # block until runner_dis goal reached
    runner_dis.block_until_done()

    # save one final time before exiting
    learner_dis.save(fname)
