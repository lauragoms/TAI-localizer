import adaptive
from func_for_fig3D import params_obs_3D
from mpi4py.futures import MPIPoolExecutor

MJ_bounds = (0, 4)
disorder_bounds = (0, 12)
num_realizations = 100


def goal(ps):
    return ps.npoints > 5000


def f(dis_MJ):
    # lattice params
    system_size = 10

    # sys params
    A = 1
    num_realizations = 100
    bond_lengthscale = 1 / system_size
    bond_power = 1 / system_size

    # localizer params
    kappa = 2
    E0 = 0

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
        sigma=0.,
        kappa_shift=0,
        beta=1,
    )


learner_dis = adaptive.Learner2D(
    f,
    bounds=[MJ_bounds, disorder_bounds],
)

if __name__ == "__main__":

    runner_dis = adaptive.Runner(
        learner_dis,
        goal=goal,
        executor=MPIPoolExecutor(),
        shutdown_executor=True,
    )

    runner_dis.live_info()
    runner_dis.block_until_done()
