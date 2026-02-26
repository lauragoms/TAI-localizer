import adaptive
from func_for_fig3D import params_obs_3D
from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.sparse import identity
import mumps

comm = MPI.COMM_WORLD
print(comm.Get_rank(), comm.Get_size())

MJ_bounds = (2, 4)
disorder_bounds = (2, 12)
num_realizations = 100


def goal(ps):
    return ps.npoints > 5000

# dummy function for testing mumps in adaptive:


def sparse_diag(matrix, k, sigma, **kwargs):
    """Call sla.eigsh with mumps support.

    See scipy.sparse.linalg.eigsh for documentation.
    """
    class LuInv(sla.LinearOperator):
        def __init__(self, A):
            inst = mumps.Context()
            inst.analyze(A, ordering='pord')
            inst.factor(A)
            self.solve = inst.solve
            sla.LinearOperator.__init__(self, A.dtype, A.shape)

        def _matvec(self, x):
            return self.solve(x.astype(self.dtype))

    opinv = LuInv(matrix - sigma * identity(matrix.shape[0]))
    return sla.eigsh(matrix, k, sigma=sigma, OPinv=opinv, **kwargs)

def mumps_test(A, B):
    matrix = sp.random(100, 100, density=0.01, format='csr') * A
    matrix += sp.eye(100) * B
    evals, evecs =sparse_diag(matrix, k=10, sigma=0)
    return evals[0] 

def f(dis_MJ):
    return mumps_test(A=dis_MJ[0], B= dis_MJ[1])

# def f(dis_MJ):
#     # lattice params
#     system_size = 10

#     # sys params
#     A = 1
#     num_realizations = 100
#     bond_lengthscale = 1 / system_size
#     bond_power = 1 / system_size

#     # localizer params
#     kappa = 2
#     E0 = 0
#     return params_obs_3D(
#         system_size=system_size,
#         MJ=dis_MJ[0],
#         A=A,
#         onsite_disorder=dis_MJ[1],
#         disorder_average=num_realizations,
#         bond_lengthscale=bond_lengthscale,
#         bond_power=bond_power,
#         kappa_spec=kappa,
#         E0=E0,
#         sigma=0.,
#         kappa_shift=0,
#         beta=1,
#     )



if __name__ == "__main__":
    fname = "data_fig3D_cluster_mpiexec.pkl"
    learner_dis = adaptive.Learner2D(
        f,
        bounds=[MJ_bounds, disorder_bounds],
    )
    # learner_dis.load(fname)

    runner_dis = adaptive.Runner(
        learner_dis,
        executor=MPIPoolExecutor(),
        shutdown_executor=True,
        goal=goal,
    )

    # periodically save the data (in case the job dies)
    runner_dis.start_periodic_saving(dict(fname=fname), interval=300)

    # block until runner_dis goal reached
    runner_dis.block_until_done()

    # save one final time before exiting
    learner_dis.save(fname)
