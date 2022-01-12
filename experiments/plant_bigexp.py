import gpflow as gpf

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
from pathlib import Path
from sacred import Experiment
from sacred.observers import FileStorageObserver
from time import process_time

sys.path.append(sys.path[0][:-len('experiments')])  # for imports to work
print(sys.path)

from core.acquisitions import get_acquisition
from core.models import GPRModule
from core.objectives import get_obj_func
from core.observers import mk_noisy_observer
from core.optimization import bayes_opt_loop_dist_robust
from core.utils import construct_grid_1d, cross_product, get_discrete_normal_dist, get_discrete_uniform_dist, \
    get_margin, get_robust_expectation_and_action, normalize_dist
from metrics.plotting import plot_robust_regret

matplotlib.use('Agg')

ex = Experiment("DRBO")
ex.observers.append(FileStorageObserver('../runs'))


@ex.named_config
def plant():
    obj_func_name = 'plant'
    action_dims = 1
    context_dims = 1
    action_lowers = [0] * action_dims
    action_uppers = [1] * action_dims
    context_lowers = [0] * context_dims
    context_uppers = [1] * context_dims
    action_density_per_dim = 50
    context_density_per_dim = 50
    ls = [0.1] * (action_dims + context_dims)
    obs_variance = 0.001
    is_optimizing_gp = False
    opt_max_iter = 10
    num_bo_iters = 400
    num_init_points = 10
    beta_const = 2
    ref_var = 0.02
    seed = 0


@ex.automain
def main(obj_func_name, action_dims, context_dims, action_lowers, action_uppers, context_lowers, context_uppers,
         action_density_per_dim, context_density_per_dim, ls, obs_variance, is_optimizing_gp, opt_max_iter,
         num_bo_iters, num_init_points, beta_const, ref_var, seed):
    dir = "runs/" + obj_func_name + "/"
    plot_dir = dir + "plots/"
    result_dir = dir + "indiv_results/"
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    divergences = ['MMD_approx', 'TV', 'modified_chi_squared']
    acquisitions = ['GP-UCB', 'DRBOGeneral', 'DRBOWorstCaseSens', 'DRBOMidApprox']
    ref_means = np.array([[0.]])
    ref_cov = ref_var * np.eye(context_dims)

    all_dims = action_dims + context_dims
    all_lowers = action_lowers + context_lowers
    all_uppers = action_uppers + context_uppers
    lengthscales = np.array(ls)
    context_lengthscales = lengthscales[-context_dims:]

    for divergence in divergences:
        for ref_mean in ref_means:
            # Calculate ground truth (robust exp) once to speed things up. Assumes constant dist and epsilon functions

            # Action space
            action_points = construct_grid_1d(action_lowers[0], action_uppers[0], action_density_per_dim)
            for i in range(action_dims - 1):
                action_points = cross_product(action_points,
                                              construct_grid_1d(action_lowers[i + 1], action_uppers[i + 1],
                                                                action_density_per_dim))

            # Context space
            context_points = construct_grid_1d(context_lowers[0], context_uppers[0], context_density_per_dim)
            for i in range(context_dims - 1):
                context_points = cross_product(context_points,
                                               construct_grid_1d(context_lowers[i + 1], context_uppers[i + 1],
                                                                 context_density_per_dim))
            search_points = cross_product(action_points, context_points)

            # Warning: move kernels into inner loop if optimizing kernel
            f_kernel = gpf.kernels.SquaredExponential(lengthscales=lengthscales)
            if divergence == 'MMD' or divergence == 'MMD_approx':
                mmd_kernel = gpf.kernels.SquaredExponential(lengthscales=context_lengthscales)
            else:
                mmd_kernel = None

            # Get objective function
            obj_func = get_obj_func(obj_func_name, all_lowers, all_uppers, f_kernel)

            # Distribution generating functions
            if divergence == 'modified_chi_squared':
                ref_dist_func = lambda x: normalize_dist(get_discrete_normal_dist(context_points, ref_mean, ref_cov) +
                                                         get_discrete_uniform_dist(context_points)/100)
            else:
                ref_dist_func = lambda x: get_discrete_normal_dist(context_points, ref_mean, ref_cov)
            true_dist_func = lambda x: get_discrete_uniform_dist(context_points)
            margin = get_margin(ref_dist_func(0), true_dist_func(0), mmd_kernel, context_points, divergence)
            margin_func = lambda x: margin  # Constant margin for now

            print("Calculating robust expectation")
            start = process_time()
            robust_expectation, robust_action = get_robust_expectation_and_action(action_points=action_points,
                                                                                  context_points=context_points,
                                                                                  kernel=mmd_kernel,
                                                                                  fvals_source='obj_func',
                                                                                  ref_dist=ref_dist_func(0),
                                                                                  divergence=divergence,
                                                                                  epsilon=margin_func(0),
                                                                                  obj_func=obj_func)
            end = process_time()
            print(f"Robust expectation took {end-start} seconds")

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
                model = GPRModule(dims=all_dims,
                                  kernel=f_kernel,
                                  noise_variance=obs_variance,
                                  dataset=init_dataset,
                                  opt_max_iter=opt_max_iter)

                # Acquisition
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
                title = obj_func_name + "({}) ".format(seed) + acq_name + " " + divergence + ", b={}".format(
                    beta_const)

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
