import gpflow as gpf
import matplotlib.pyplot as plt
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver

from core.acquisitions import get_acquisition
from core.models import GPRModule
from core.objectives import get_obj_func
from core.observers import mk_noisy_observer
from core.optimization import bayes_opt_loop_dist_robust
from core.utils import construct_grid_1d, cross_product, get_discrete_normal_dist_1d, get_discrete_uniform_dist_1d, \
    get_margin
from metrics.plotting import plot_function_2d, plot_bo_points_2d, plot_robust_regret, plot_gp_2d

ex = Experiment("DRBO")
ex.observers.append(FileStorageObserver('runs'))


@ex.named_config
def rand_func():
    acq_name = 'GP-UCB'
    obj_func_name = 'rand_func'
    divergence = 'MMD'  # 'MMD' or 'TV'
    dims = 2
    lowers = [0] * dims
    uppers = [1] * dims
    grid_density_per_dim = 20
    rand_func_num_points = 100
    ls = 0.1
    obs_variance = 0.001
    is_optimizing_gp = False
    opt_max_iter = 10
    num_bo_iters = 200
    num_init_points = 3
    beta_const = 1
    ref_mean = 0.5
    ref_var = 0.05
    true_mean = 0.2
    true_var = 0.05
    seed = 1


@ex.automain
def main(acq_name, obj_func_name, divergence, lowers, uppers, grid_density_per_dim, rand_func_num_points,
         dims, ls, obs_variance, is_optimizing_gp, num_bo_iters, opt_max_iter, num_init_points, beta_const,
         ref_mean, ref_var, true_mean, true_var, seed):
    np.random.seed(seed)

    f_kernel = gpf.kernels.SquaredExponential(lengthscales=[ls] * dims)
    if divergence == 'MMD':
        mmd_kernel = gpf.kernels.SquaredExponential(lengthscales=[ls])  # 1d for now
    else:
        mmd_kernel = None

    # Get objective function
    obj_func = get_obj_func(obj_func_name, lowers, uppers, f_kernel, rand_func_num_points, seed)

    # Action space (1d for now)
    action_points = construct_grid_1d(lowers[0], uppers[0], grid_density_per_dim)
    # Context space (1d for now)
    context_points = construct_grid_1d(lowers[1], uppers[1], grid_density_per_dim)
    search_points = cross_product(action_points, context_points)

    observer = mk_noisy_observer(obj_func, obs_variance)
    init_dataset = observer(search_points[np.random.randint(0, len(search_points), num_init_points)])

    # Model
    model = GPRModule(dims=dims,
                      kernel=f_kernel,
                      noise_variance=obs_variance,
                      dataset=init_dataset,
                      opt_max_iter=opt_max_iter)

    # Acquisition
    acquisition = get_acquisition(acq_name=acq_name,
                                  beta=lambda x: beta_const,
                                  divergence=divergence)  # TODO: Implement beta function

    # Distribution generating functions
    ref_dist_func = lambda x: get_discrete_normal_dist_1d(context_points, ref_mean, ref_var)
    # true_dist_func = lambda x: get_discrete_normal_dist_1d(context_points, true_mean, true_var)
    true_dist_func = lambda x: get_discrete_uniform_dist_1d(context_points)
    margin = get_margin(ref_dist_func(0), true_dist_func(0), mmd_kernel, context_points, divergence)
    margin_func = lambda x: margin  # Constant margin for now
    print("Using margin = {}".format(margin))

    # Main BO loop
    final_dataset, model_params = bayes_opt_loop_dist_robust(model=model,
                                                             init_dataset=init_dataset,
                                                             action_points=action_points,
                                                             context_points=context_points,
                                                             observer=observer,
                                                             acq=acquisition,
                                                             num_iters=num_bo_iters,
                                                             reference_dist_func=ref_dist_func,
                                                             true_dist_func=true_dist_func,
                                                             margin_func=margin_func,
                                                             divergence=divergence,
                                                             mmd_kernel=mmd_kernel,
                                                             optimize_gp=is_optimizing_gp)
    print("Final dataset: {}".format(final_dataset))
    # Plots
    query_points = final_dataset.query_points.numpy()
    maximizer = search_points[[np.argmax(obj_func(search_points))]]
    if dims == 2:
        title = obj_func_name + ", " + acq_name + " " + divergence + ", beta = {}".format(beta_const)
        _, ax = plot_function_2d(obj_func, lowers, uppers, grid_density_per_dim, contour=True,
                                 title=title, colorbar=True)
        plot_bo_points_2d(query_points, ax, num_init=num_init_points, maximizer=maximizer)

        _, ax = plot_gp_2d(model.gp, mins=lowers, maxs=uppers, grid_density=grid_density_per_dim)

    _, ax = plot_robust_regret(obj_func=obj_func,
                               query_points=query_points,
                               action_points=action_points,
                               context_points=context_points,
                               kernel=mmd_kernel,
                               ref_dist_func=ref_dist_func,
                               margin_func=margin_func,
                               divergence=divergence)

    plt.show()
