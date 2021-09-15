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

from core.acquisitions import get_acquisition
from core.models import GPRModule
from core.objectives import get_obj_func
from core.observers import mk_noisy_observer
from core.optimization import bayes_opt_loop_dist_robust
from core.utils import construct_grid_1d, cross_product, get_margin, get_robust_expectation_and_action
from metrics.plotting import plot_function_2d, plot_bo_points_2d, plot_robust_regret, plot_gp_2d, \
    plot_cumulative_rewards

matplotlib.use('Agg')

ex = Experiment("DRBO")
ex.observers.append(FileStorageObserver('../runs'))


@ex.named_config
def wind():
    obj_func_name = 'wind'
    action_dims = 1
    context_dims = 1
    action_lowers = [0] * action_dims
    action_uppers = [1] * action_dims
    context_lowers = [0] * context_dims
    context_uppers = [1] * context_dims
    action_density_per_dim = 50
    context_density_per_dim = 50
    ls = 0.1
    obs_variance = 0.001
    is_optimizing_gp = False
    opt_max_iter = 10
    num_bo_iters = 200
    num_init_points = 10
    beta_const = 2
    seed = 0
    month = 0  # 0-23
    show_plots = False


@ex.automain
def main(obj_func_name, action_dims, context_dims, action_lowers, action_uppers, context_lowers, context_uppers,
         action_density_per_dim, context_density_per_dim, ls, obs_variance, is_optimizing_gp, opt_max_iter,
         num_bo_iters, num_init_points, beta_const, seed, month, show_plots):
    Path("runs/plots").mkdir(parents=True, exist_ok=True)
    Path("runs/indiv_results").mkdir(parents=True, exist_ok=True)

    divergences = ['MMD_approx', 'TV', 'modified_chi_squared']
    acquisitions = ['GP-UCB', 'DRBOGeneral', 'DRBOWorstCaseSens', 'DRBOMidApprox']

    all_dims = action_dims + context_dims
    all_lowers = action_lowers + context_lowers
    all_uppers = action_uppers + context_uppers
    lengthscales = np.array([ls] * all_dims)
    context_lengthscales = np.array([ls] * context_dims)

    # Load data
    ref_dist, power_in_months, true_dist_in_months = pickle.load(open("data/wind/wind_data.p", "rb"))

    for divergence in divergences:
        num_bo_iters = len(power_in_months[month])  # adapt num_bo_iters to number of data points in a month

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
        obj_func = get_obj_func(obj_func_name, all_lowers, all_uppers, f_kernel, seed)
        power_sequence = power_in_months[month][:, None]

        # Distribution generating functions
        ref_dist_func = lambda x: ref_dist
        true_dist = true_dist_in_months[month]
        true_dist_func = lambda x: true_dist
        margin = get_margin(ref_dist_func(0), true_dist_func(0), mmd_kernel, context_points, divergence)
        margin_func = lambda x: margin  # Constant margin for now

        # Calculate ground truth (robust exp) once to speed things up. Assumes constant dist and epsilon functions
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
            file_name = "{}-month{}-{}-{}-seed{}-beta{}".format(obj_func_name,
                                                                month,
                                                                divergence,
                                                                acq_name,
                                                                seed,
                                                                beta_const)
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
                                                                                       optimize_gp=is_optimizing_gp,
                                                                                       custom_sequence=power_sequence)
            print("Final dataset: {}".format(final_dataset))
            print("Average acquisition time in seconds: {}".format(average_acq_time))

            # Plots
            query_points = final_dataset.query_points.numpy()
            rewards = obj_func(query_points)
            cumulative_rewards = np.cumsum(rewards)

            maximizer = search_points[[np.argmax(obj_func(search_points))]]
            title = obj_func_name + "({}) ".format(seed) + acq_name + " " + divergence + ", b={}".format(
                beta_const)

            if all_dims == 2:
                fig, ax = plot_function_2d(obj_func, all_lowers, all_uppers,
                                           np.max([action_density_per_dim, context_density_per_dim]), contour=True,
                                           title=title, colorbar=True)
                plot_bo_points_2d(query_points, ax, num_init=num_init_points, maximizer=maximizer)
                fig.savefig("runs/plots/" + file_name + "-obj_func.png")
                plt.close(fig)

                fig, ax = plot_gp_2d(model.gp, mins=all_lowers, maxs=all_uppers,
                                     grid_density=np.max([action_density_per_dim, context_density_per_dim]),
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

            fig = plot_cumulative_rewards(cumulative_rewards=cumulative_rewards,
                                          title=title)
            fig.savefig("runs/plots/" + file_name + "-rewards.png")

            pickle.dump((regrets, cumulative_regrets, average_acq_time, query_points, rewards, cumulative_rewards),
                        open("runs/indiv_results" + file_name + ".p", "wb"))

            if show_plots:
                plt.show()
