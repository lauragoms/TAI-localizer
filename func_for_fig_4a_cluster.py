from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
import adaptive
from func_for_fig4 import params_obs_3D

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

MJ_bounds = (0, 4)
disorder_bounds = (0, 10)
num_realizations = 30
system_size = 10
A = 1
bond_lengthscale = 1 / system_size
bond_power = 1 / system_size
kappa = 2
E0 = 0
sigma = 0
kappa_shift = 0
beta = 1
fname = "data_fig4a30reals.pkl"

def goal(ps):
    return ps.npoints > 5000

def f(dis_MJ):
    return params_obs_3D(
        system_size=system_size,
        MJ=dis_MJ[0],
        A=A,
        onsite_disorder=dis_MJ[1],
        disorder_average=num_realizations,
        bond_lengthscale=bond_lengthscale,
        bond_power=bond_power,
        kappa_spec=kappa,
        E0=E0,
        sigma=sigma,
        kappa_shift=kappa_shift,
        beta=beta,
        provide_sites=False,
        comm=MPI.COMM_SELF,
    )

if rank == 0:
    # Driver
    learner_dis = adaptive.Learner2D(f, bounds=[MJ_bounds, disorder_bounds])

    # Executor: todos los demás procesos serán automáticamente workers
    with MPIPoolExecutor(max_workers=size-1) as executor:
        runner_dis = adaptive.Runner(
            learner_dis,
            executor=executor,
            goal=goal
        )
        runner_dis.start_periodic_saving(dict(fname=fname), interval=60)
        runner_dis.block_until_done()
        learner_dis.save(fname)