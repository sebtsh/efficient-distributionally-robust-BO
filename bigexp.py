import gpflow as gpf

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path
from sacred import Experiment
from sacred.observers import FileStorageObserver

from core.acquisitions import get_acquisition, get_beta_linear_schedule
from core.models import GPRModule
from core.objectives import get_obj_func
from core.observers import mk_noisy_observer
from core.optimization import bayes_opt_loop_dist_robust
from core.utils import construct_grid_1d, cross_product, get_discrete_normal_dist_1d, get_discrete_uniform_dist_1d, \
    get_margin, get_robust_expectation_and_action
from metrics.plotting import plot_function_2d, plot_bo_points_2d, plot_robust_regret, plot_gp_2d

matplotlib.use('Agg')

ex = Experiment("DRBO")
ex.observers.append(FileStorageObserver('runs'))


@ex.named_config
def rand_func():
    obj_func_name = 'rand_func'
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
    num_init_points = 10
    beta_schedule = 'constant'  # 'constant' or 'linear'
    ref_var = 0.05
    true_mean = 0.2
    true_var = 0.05
    seed = 0
    show_plots = False


@ex.automain
def main(obj_func_name, lowers, uppers, grid_density_per_dim, rand_func_num_points,
         dims, ls, obs_variance, is_optimizing_gp, num_bo_iters, opt_max_iter, num_init_points, beta_schedule,
         ref_var, true_mean, true_var, seed, show_plots):
    divergences = ['MMD', 'TV', 'modified_chi_squared']
    ref_means = [0, 0.25, 0.5]
    acquisitions = ['GP-UCB', 'DRBOGeneral', 'DRBOWorstCaseSens', 'DRBOCubicApprox', 'DRBOMidApprox', 'WorstCaseSensTS',
                    'CubicApproxTS', 'MidApproxTS']
    beta_consts = [0, 0.5, 1, 2, 3, 4]

    for divergence in divergences:
        for ref_mean in ref_means:
            # Calculate ground truth (robust exp) once to speed things up. Assumes constant dist and epsilon functions

            # Action space (1d for now)
            action_points = construct_grid_1d(lowers[0], uppers[0], grid_density_per_dim)
            # Context space (1d for now)
            context_points = construct_grid_1d(lowers[1], uppers[1], grid_density_per_dim)
            search_points = cross_product(action_points, context_points)

            # We can do this because not optimizing kernel
            f_kernel = gpf.kernels.SquaredExponential(lengthscales=[ls] * dims)
            if divergence == 'MMD':
                mmd_kernel = gpf.kernels.SquaredExponential(lengthscales=[ls])  # 1d for now
            else:
                mmd_kernel = None

            # Get objective function
            obj_func = get_obj_func(obj_func_name, lowers, uppers, f_kernel, rand_func_num_points, seed)

            # Distribution generating functions
            ref_dist_func = lambda x: get_discrete_normal_dist_1d(context_points, ref_mean, ref_var)
            # true_dist_func = lambda x: get_discrete_normal_dist_1d(context_points, true_mean, true_var)
            true_dist_func = lambda x: get_discrete_uniform_dist_1d(context_points)
            margin = get_margin(ref_dist_func(0), true_dist_func(0), mmd_kernel, context_points, divergence)
            margin_func = lambda x: margin  # Constant margin for now

            print("Calculating robust expectation")
            robust_expectation, robust_action = get_robust_expectation_and_action(action_points=action_points,
                                                                                  context_points=context_points,
                                                                                  kernel=mmd_kernel,
                                                                                  fvals_source='obj_func',
                                                                                  ref_dist=ref_dist_func(0),
                                                                                  divergence=divergence,
                                                                                  epsilon=margin_func(0),
                                                                                  obj_func=obj_func)

            for acq_name in acquisitions:
                for beta_const in beta_consts:
                    file_name = "{}-{}-{}-seed{}-beta{}{}-refmean{}".format(obj_func_name,
                                                                            divergence,
                                                                            acq_name,
                                                                            seed,
                                                                            beta_const,
                                                                            beta_schedule,
                                                                            ref_mean)
                    print("==========================")
                    print("Running experiment " + file_name)
                    print("Using margin = {}".format(margin))
                    np.random.seed(seed)

                    observer = mk_noisy_observer(obj_func, obs_variance)
                    init_dataset = observer(search_points[np.random.randint(0, len(search_points), num_init_points)])

                    # Model
                    model = GPRModule(dims=dims,
                                      kernel=f_kernel,
                                      noise_variance=obs_variance,
                                      dataset=init_dataset,
                                      opt_max_iter=opt_max_iter)

                    # Acquisition
                    # Create beta schedule
                    if beta_schedule == 'constant':
                        beta = lambda x: beta_const
                    elif beta_schedule == 'linear':
                        beta = get_beta_linear_schedule(2, 0, 100)
                    else:
                        raise Exception("Incorrect beta_schedule provided")
                    acquisition = get_acquisition(acq_name=acq_name,
                                                  beta=beta,
                                                  divergence=divergence)

                    # Main BO loop
                    final_dataset, model_params, average_acq_time = bayes_opt_loop_dist_robust(model=model,
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
                    print("Average acquisition time in seconds: {}".format(average_acq_time))
                    # Plots
                    query_points = final_dataset.query_points.numpy()
                    maximizer = search_points[[np.argmax(obj_func(search_points))]]
                    title = obj_func_name + "({}) ".format(seed) + acq_name + " " + divergence + ", b={}{}".format(
                        beta_const, beta_schedule)

                    if dims == 2:
                        Path("runs/plots").mkdir(parents=True, exist_ok=True)
                        fig, ax = plot_function_2d(obj_func, lowers, uppers, grid_density_per_dim, contour=True,
                                                   title=title, colorbar=True)
                        plot_bo_points_2d(query_points, ax, num_init=num_init_points, maximizer=maximizer)
                        fig.savefig("runs/plots/" + file_name + "-obj_func.png")
                        plt.close(fig)

                        fig, ax = plot_gp_2d(model.gp, mins=lowers, maxs=uppers, grid_density=grid_density_per_dim,
                                             save_location="runs/plots/" + file_name + "-gp.png")
                        plt.close(fig)

                    fig, ax, regrets, cumulative_regrets = plot_robust_regret(obj_func=obj_func,
                                                                              query_points=query_points,
                                                                              action_points=action_points,
                                                                              context_points=context_points,
                                                                              kernel=mmd_kernel,
                                                                              ref_dist_func=ref_dist_func,
                                                                              margin_func=margin_func,
                                                                              divergence=divergence,
                                                                              robust_expectation_action=(
                                                                                  robust_expectation, robust_action),
                                                                              title=title)
                    fig.savefig("runs/plots/" + file_name + "-regret.png")
                    plt.close(fig)

                    pickle.dump((regrets, cumulative_regrets, average_acq_time, query_points),
                                open("runs/" + file_name + ".p", "wb"))

                    if show_plots:
                        plt.show()
