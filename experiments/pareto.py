import gpflow as gpf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
from pathlib import Path
from sacred import Experiment
from sacred.observers import FileStorageObserver
from tqdm import trange
from time import process_time

sys.path.append(sys.path[0][:-len('experiments')])  # for imports to work
print(sys.path)

from core.objectives import get_obj_func
from core.utils import construct_grid_1d, cross_product, get_discrete_normal_dist, get_discrete_uniform_dist, \
    get_margin, adversarial_expectation, get_mid_approx_func, worst_case_sens, get_action_contexts, wass_cost_vector

matplotlib.use('Agg')

ex = Experiment("Pareto")
ex.observers.append(FileStorageObserver('../runs'))


@ex.named_config
def default():
    obj_func_name = 'rand_func'
    divergence = 'wass'
    action_dims = 1
    context_dims = 2
    action_lowers = [0] * action_dims
    action_uppers = [1] * action_dims
    context_lowers = [0] * context_dims
    context_uppers = [1] * context_dims
    action_density_per_dim = 20
    context_density_per_dim = 30
    rand_func_num_points = 100
    ls = 0.1
    ref_mean = 0.5
    ref_var = 0.1
    num_scs_max_iters = 20
    scs_max_iter_block = 20
    seed = 0
    show_plots = False


@ex.automain
def main(obj_func_name, divergence, action_lowers, action_uppers, context_lowers, context_uppers,
         action_density_per_dim, context_density_per_dim, rand_func_num_points, action_dims, context_dims, ls, ref_mean,
         ref_var, num_scs_max_iters, scs_max_iter_block, seed, show_plots):
    result_dir = "runs/pareto/"
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    all_dims = action_dims + context_dims
    all_lowers = action_lowers + context_lowers
    all_uppers = action_uppers + context_uppers
    lengthscales = np.array([ls] * all_dims)
    context_lengthscales = np.array([ls] * context_dims)

    f_kernel = gpf.kernels.SquaredExponential(lengthscales=lengthscales)
    if divergence == 'MMD' or divergence == 'MMD_approx':
        mmd_kernel = gpf.kernels.SquaredExponential(lengthscales=context_lengthscales)
    else:
        mmd_kernel = None

    # Get objective function
    obj_func = get_obj_func(obj_func_name, all_lowers, all_uppers, f_kernel, 1, rand_func_num_points, seed)

    # Action space
    action_points = construct_grid_1d(action_lowers[0], action_uppers[0], action_density_per_dim)
    for i in range(action_dims - 1):
        action_points = cross_product(action_points, construct_grid_1d(action_lowers[i + 1], action_uppers[i + 1],
                                                                       action_density_per_dim))

    # Context space
    context_points = construct_grid_1d(context_lowers[0], context_uppers[0], context_density_per_dim)
    for i in range(context_dims - 1):
        context_points = cross_product(context_points, construct_grid_1d(context_lowers[i + 1], context_uppers[i + 1],
                                                                         context_density_per_dim))

    if divergence == "wass":
        v = wass_cost_vector(context_points, 1)
    else:
        v = None

    # Distribution generating functions
    ref_mean_arr = ref_mean * np.ones(context_dims)
    ref_cov = np.eye(context_dims) * ref_var
    ref_dist_func = lambda x: get_discrete_normal_dist(context_points, ref_mean_arr, ref_cov)
    true_dist_func = lambda x: get_discrete_uniform_dist(context_points)
    margin = get_margin(ref_dist_func(0), true_dist_func(0), mmd_kernel, context_points, divergence)
    margin_func = lambda x: margin  # Constant margin for now
    print("Using margin = {}".format(margin))

    kernel = mmd_kernel
    ref_dist = ref_dist_func(0)
    epsilon = margin_func(0)

    num_actions = len(action_points)
    num_context_points = len(context_points)
    domain = cross_product(action_points, context_points)
    all_max_iters = [i * scs_max_iter_block for i in range(1, num_scs_max_iters)]

    true_adv_exps = []
    wcs_adv_approxs = []
    wcs_timings = []
    all_truncated_exps = []
    all_truncated_timings = []
    for i in trange(num_actions):
        action_contexts = get_action_contexts(i, domain, num_context_points)
        if divergence == 'MMD' or divergence == 'MMD_approx':
            M = kernel(context_points)
        else:
            M = None
        f = np.squeeze(obj_func(action_contexts), axis=-1)

        # Ground truth adversarial expectations
        expectation, _ = adversarial_expectation(f=f,
                                                 M=M,
                                                 w_t=ref_dist,
                                                 epsilon=epsilon,
                                                 divergence=divergence,
                                                 v=v)
        true_adv_exps.append(expectation)

        # Worst case sensitivity approximation
        start = process_time()
        worst_case_sensitivity = worst_case_sens(fvals=f,
                                                 p=ref_dist,
                                                 context_points=context_points,
                                                 kernel=kernel,
                                                 divergence=divergence,
                                                 v=v)
        V_approx_func = get_mid_approx_func(context_points=context_points,
                                            fvals=f,
                                            kernel=kernel,
                                            ref_dist=ref_dist,
                                            worst_case_sensitivity=worst_case_sensitivity,
                                            divergence=divergence)
        end = process_time()
        wcs_adv_approxs.append(V_approx_func(epsilon))
        wcs_timings.append(end - start)

        # Truncated convex optimization
        truncated_exps = []
        truncated_timings = []
        for max_iters in all_max_iters:
            start = process_time()
            truncated_expectation, _ = adversarial_expectation(f=f,
                                                               M=M,
                                                               w_t=ref_dist,
                                                               epsilon=epsilon,
                                                               divergence=divergence,
                                                               cvx_opt_max_iters=max_iters,
                                                               cvx_solver='SCS',
                                                               v=v)
            end = process_time()
            truncated_exps.append(truncated_expectation)
            truncated_timings.append(end - start)
        all_truncated_exps.append(truncated_exps)
        all_truncated_timings.append(truncated_timings)

    true_adv_exps = np.array(true_adv_exps)
    print("Adv. expectations = {}".format(true_adv_exps))

    # Get WCS approximation error, averaged across all actions
    wcs_adv_approxs = np.squeeze(np.array(wcs_adv_approxs))
    print("WCS adv. approximations = {}".format(wcs_adv_approxs))

    wcs_approx_error = abs(true_adv_exps - wcs_adv_approxs)
    print("WCS approximation errors = {}".format(wcs_approx_error))
    wcs_average_error = np.mean(wcs_approx_error)
    print("WCS average approximation error = {}".format(wcs_average_error))

    wcs_average_timing = np.mean(wcs_timings)
    print("WCS timing: {}".format(wcs_average_timing))

    # Get approximation error from early stopping convex optimization
    all_truncated_exps = np.array(all_truncated_exps)
    all_truncated_timings = np.array(all_truncated_timings)

    truncated_error = abs(true_adv_exps[:, None] - all_truncated_exps)  # (num_actions, num_scs_max_iters)
    truncated_average_error = np.mean(truncated_error, axis=0)
    print("Truncated average approximation error = {}".format(truncated_average_error))

    truncated_average_timings = np.mean(all_truncated_timings, axis=0)
    print("Average timings: {}".format(truncated_average_timings))

    file_name = "pareto-{}-{}-seed{}-refmean{}-maxiterblock{}-cdensity{}".format(obj_func_name,
                                                                                 divergence,
                                                                                 seed,
                                                                                 ref_mean,
                                                                                 scs_max_iter_block,
                                                                                 context_density_per_dim)

    pickle.dump((true_adv_exps, wcs_adv_approxs, wcs_timings, all_truncated_exps, all_truncated_timings),
                open(result_dir + file_name + ".p", "wb"))

    plt.scatter(truncated_average_timings, truncated_average_error, label='Truncated convex opt.')
    plt.scatter([wcs_average_timing], [wcs_average_error], label='Worst case sens.')

    plt.xlabel("Timing")
    plt.ylabel("Approximation error")
    plt.legend()
    if show_plots:
        plt.show()
    fig = plt.gcf()
    fig.savefig(result_dir + file_name + ".png")
