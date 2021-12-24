import gpflow as gpf

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
from pathlib import Path
from sacred import Experiment
from sacred.observers import FileStorageObserver

sys.path.append(sys.path[0][:-len('experiments')])  # for imports to work
print(sys.path)

from core.acquisitions import get_acquisition
from core.models import GPRModule
from core.objectives import get_obj_func
from core.observers import mk_noisy_observer
from core.optimization import bayes_opt_loop_dist_robust
from core.utils import construct_grid_1d, cross_product, get_discrete_normal_dist_1d, get_discrete_uniform_dist, \
    get_margin, get_robust_expectation_and_action, normalize_dist
from metrics.plotting import plot_function_2d, plot_bo_points_2d, plot_robust_regret, plot_gp_2d

matplotlib.use('Agg')

ex = Experiment("DRBO")
ex.observers.append(FileStorageObserver('../runs'))


@ex.named_config
def rand_func():
    obj_func_name = 'rand_func'
    dims = 2
    lowers = [0] * dims
    uppers = [1] * dims
    grid_density_per_dim = 20
    rand_func_num_points = 100
    ls = 0.05
    obs_variance = 0.001
    is_optimizing_gp = False
    opt_max_iter = 10
    num_bo_iters = 400
    num_init_points = 10
    beta_const = 2
    ref_var = 0.02
    seed = 0


@ex.automain
def main(obj_func_name, lowers, uppers, grid_density_per_dim, rand_func_num_points,
         dims, ls, obs_variance, is_optimizing_gp, num_bo_iters, opt_max_iter, num_init_points, beta_const,
         ref_var, seed):
    dir = "runs/" + obj_func_name + "/"
    plot_dir = dir + "plots/"
    result_dir = dir + "indiv_results/"
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    divergences = ['modified_chi_squared']
    ref_means = [0, 0.5]
    acquisitions = ['GP-UCB', 'DRBOGeneral', 'DRBOWorstCaseSens', 'DRBOMidApprox']

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
            if divergence == 'MMD' or divergence == 'MMD_approx':
                mmd_kernel = gpf.kernels.SquaredExponential(lengthscales=[ls])  # 1d for now
            else:
                mmd_kernel = None

            # Get objective function
            obj_func = get_obj_func(obj_func_name, lowers, uppers, f_kernel, rand_func_num_points, seed)

            # Distribution generating functions
            if divergence == 'modified_chi_squared':  # Add small uniform everywhere for numeric reasons
                ref_dist_func = lambda x: normalize_dist(get_discrete_normal_dist_1d(context_points, ref_mean, ref_var) +
                                                         get_discrete_uniform_dist(context_points)/10)
            else:
                ref_dist_func = lambda x: get_discrete_normal_dist_1d(context_points, ref_mean, ref_var)
            true_dist_func = lambda x: get_discrete_uniform_dist(context_points)
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
                file_name = "{}-{}-{}-seed{}-beta{}-refmean{}".format(obj_func_name,
                                                                      divergence,
                                                                      acq_name,
                                                                      seed,
                                                                      beta_const,
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
                beta = lambda x: beta_const
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
                title = obj_func_name + "({}) ".format(seed) + acq_name + " " + divergence + ", b={}".format(
                    beta_const)

                if dims == 2:
                    fig, ax = plot_function_2d(obj_func, lowers, uppers, grid_density_per_dim, contour=True,
                                               title=title, colorbar=True)
                    plot_bo_points_2d(query_points, ax, num_init=num_init_points, maximizer=maximizer)
                    fig.savefig(plot_dir + file_name + "-obj_func.png")
                    plt.close(fig)

                    fig, ax = plot_gp_2d(model.gp, mins=lowers, maxs=uppers, grid_density=grid_density_per_dim,
                                         save_location=plot_dir + file_name + "-gp.png")
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
                fig.savefig(plot_dir + file_name + "-regret.png")
                plt.close(fig)

                pickle.dump((regrets, cumulative_regrets, average_acq_time, query_points),
                            open(result_dir + file_name + ".p", "wb"))
