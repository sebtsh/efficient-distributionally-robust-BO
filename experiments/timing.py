import gpflow as gpf
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
from sacred import Experiment
from sacred.observers import FileStorageObserver
from pathlib import Path

sys.path.append(sys.path[0][:-len('experiments')])  # for imports to work
print(sys.path)

from core.acquisitions import get_acquisition
from core.models import GPRModule
from core.objectives import get_obj_func
from core.observers import mk_noisy_observer
from core.optimization import bayes_opt_loop_dist_robust
from core.utils import construct_grid_1d, cross_product, get_discrete_normal_dist_1d, get_discrete_uniform_dist, \
    get_margin

ex = Experiment("DRBO_timing")
ex.observers.append(FileStorageObserver('../runs'))


@ex.named_config
def rand_func():
    obj_func_name = 'rand_func'
    dims = 2
    lowers = [0] * dims
    uppers = [1] * dims
    action_grid_density = 20
    rand_func_num_points = 100
    ls = 0.1
    obs_variance = 0.001
    is_optimizing_gp = False
    opt_max_iter = 10
    num_bo_iters = 20
    num_init_points = 10
    beta_const = 0
    ref_mean = 0.5
    ref_var = 0.05
    true_mean = 0.2
    true_var = 0.05
    seed = 0
    show_plots = False


@ex.automain
def main(obj_func_name, lowers, uppers, action_grid_density, rand_func_num_points,
         dims, ls, obs_variance, is_optimizing_gp, num_bo_iters, opt_max_iter, num_init_points, beta_const,
         ref_mean, ref_var, true_mean, true_var, seed, show_plots, figsize=(15, 6), dpi=None):
    Path("../runs/plots").mkdir(parents=True, exist_ok=True)
    context_grid_densities = np.arange(200, 1800, 200)
    divergences = ['MMD', 'TV', 'modified_chi_squared']
    acquisitions = ['GP-UCB', 'DRBOGeneral', 'DRBOWorstCaseSens', 'DRBOCubicApprox']
    color_dict = {'GP-UCB': '#d7263d',
                  'DRBOGeneral': '#fbb13c',
                  'DRBOWorstCaseSens': '#26c485',
                  'DRBOCubicApprox': '#00a6ed',
                  'WorstCaseSensTS': '#9f956c',
                  'CubicApproxTS': '#2f4858'}

    for divergence in divergences:
        timing_dict = {}
        for acq_name in acquisitions:
            average_acq_times = []
            for context_grid_density in context_grid_densities:
                print("Timing for {}-{}-{}".format(divergence, acq_name, context_grid_density))
                np.random.seed(seed)

                f_kernel = gpf.kernels.SquaredExponential(lengthscales=[ls] * dims)
                if divergence == 'MMD' or divergence == 'MMD_approx':
                    mmd_kernel = gpf.kernels.SquaredExponential(lengthscales=[ls])  # 1d for now
                else:
                    mmd_kernel = None

                # Get objective function
                obj_func = get_obj_func(obj_func_name, lowers, uppers, f_kernel, rand_func_num_points, seed)

                # Action space (1d for now)
                action_points = construct_grid_1d(lowers[0], uppers[0], action_grid_density)
                # Context space (1d for now)
                context_points = construct_grid_1d(lowers[1], uppers[1], context_grid_density)
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
                true_dist_func = lambda x: get_discrete_uniform_dist(context_points)
                margin = get_margin(ref_dist_func(0), true_dist_func(0), mmd_kernel, context_points, divergence)
                margin_func = lambda x: margin  # Constant margin for now
                print("Using margin = {}".format(margin))

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
                average_acq_times.append(average_acq_time)
            timing_dict[acq_name] = np.array(average_acq_times)
        # Plots
        fig = plt.figure(figsize=figsize, dpi=dpi)
        for acquisition in acquisitions:
            plt.plot(context_grid_densities, timing_dict[acquisition], label=acquisition, color=color_dict[acquisition])
        plt.title("{} average acquisition time in seconds".format(divergence))
        plt.xlabel("Size of context set")
        plt.ylabel("Seconds")
        plt.legend()
        fig.savefig("runs/plots/" + "{}-timing.png".format(divergence))

    pickle.dump(timing_dict, open("runs/timing_dict.p", "wb"))

    if show_plots:
        plt.show()
