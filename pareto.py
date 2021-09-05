import gpflow as gpf
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path
from sacred import Experiment
from sacred.observers import FileStorageObserver

from core.objectives import get_obj_func

from core.utils import construct_grid_1d, cross_product, get_discrete_normal_dist, get_discrete_uniform_dist, \
    get_margin, adversarial_expectation, get_mid_approx_func, worst_case_sens, get_action_contexts
from tqdm import trange
from timeit import default_timer as timer

ex = Experiment("Pareto")
ex.observers.append(FileStorageObserver('runs'))


@ex.named_config
def rand_func():
    acq_name = 'GP-UCB'
    obj_func_name = 'rand_func'
    divergence = 'MMD_approx'  # 'MMD', 'TV' or 'modified_chi_squared'' or 'modified_chi_squared'
    action_dims = 1
    context_dims = 2
    action_lowers = [0] * action_dims
    action_uppers = [1] * action_dims
    context_lowers = [0] * context_dims
    context_uppers = [1] * context_dims
    grid_density_per_dim = 20
    rand_func_num_points = 100
    ls = 0.1
    obs_variance = 0.001
    is_optimizing_gp = False
    opt_max_iter = 10
    num_bo_iters = 10
    num_init_points = 10
    beta_const = 2
    beta_schedule = 'constant'  # 'constant' or 'linear'
    ref_mean = 0.5
    ref_var = 0.1
    true_mean = 0.2
    true_var = 0.05
    seed = 4
    show_plots = False
    num_scs_max_iters = 10
    scs_max_iter_block = 1


@ex.automain
def main(acq_name, obj_func_name, divergence, action_lowers, action_uppers, context_lowers, context_uppers,
         grid_density_per_dim, rand_func_num_points, action_dims, context_dims, ls, obs_variance, is_optimizing_gp,
         num_bo_iters, opt_max_iter, num_init_points, beta_const, beta_schedule, ref_mean, ref_var, true_mean, true_var,
         seed, show_plots, num_scs_max_iters, scs_max_iter_block):
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
    obj_func = get_obj_func(obj_func_name, all_lowers, all_uppers, f_kernel, rand_func_num_points, seed)

    # Action space
    action_points = construct_grid_1d(action_lowers[0], action_uppers[0], grid_density_per_dim)
    for i in range(action_dims - 1):
        action_points = cross_product(action_points, construct_grid_1d(action_lowers[i + 1], action_uppers[i + 1],
                                                                       grid_density_per_dim))

    # Context space
    context_points = construct_grid_1d(context_lowers[0], context_uppers[0], grid_density_per_dim)
    for i in range(context_dims - 1):
        context_points = cross_product(context_points, construct_grid_1d(context_lowers[i + 1], context_uppers[i + 1],
                                                                         grid_density_per_dim))

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
                                                 divergence=divergence)
        true_adv_exps.append(expectation)

        # Worst case sensitivity approximation
        start = timer()
        worst_case_sensitivity = worst_case_sens(fvals=f,
                                                 context_points=context_points,
                                                 kernel=kernel,
                                                 divergence=divergence)
        V_approx_func = get_mid_approx_func(context_points=context_points,
                                            fvals=f,
                                            kernel=kernel,
                                            ref_dist=ref_dist,
                                            worst_case_sensitivity=worst_case_sensitivity,
                                            divergence=divergence)
        wcs_adv_approxs.append(V_approx_func(epsilon))
        end = timer()
        wcs_timings.append(end - start)

        # Truncated convex optimization
        truncated_exps = []
        truncated_timings = []
        for max_iters in all_max_iters:
            start = timer()
            truncated_expectation, _ = adversarial_expectation(f=f,
                                                               M=M,
                                                               w_t=ref_dist,
                                                               epsilon=epsilon,
                                                               divergence=divergence,
                                                               cvx_opt_max_iters=max_iters,
                                                               cvx_solver='SCS')
            end = timer()
            truncated_exps.append(truncated_expectation)
            truncated_timings.append(end - start)
        all_truncated_exps.append(truncated_exps)
        all_truncated_timings.append(truncated_timings)

    true_adv_exps = np.array(true_adv_exps)
    best_action = np.argmax(true_adv_exps)
    best_expectation = np.max(true_adv_exps)

    print("Best action is {}, with adv. expectation = {}".format(best_action, best_expectation))

    wcs_action = np.argmax(wcs_adv_approxs)
    wcs_expectation = true_adv_exps[wcs_action]

    print("Worst case sensitivity chose {}, with adv. expectation = {}".format(wcs_action,
                                                                               wcs_expectation))
    mean_wcs_timing = np.mean(wcs_timings)
    print("Worst case sensitivity timing: {}".format(mean_wcs_timing))

    truncated_actions = np.argmax(np.array(all_truncated_exps), axis=0)
    truncated_expectation = true_adv_exps[truncated_actions]
    truncated_regret = best_expectation - truncated_expectation
    print("Truncated regret: {}".format(truncated_regret))
    mean_truncated_timings = np.mean(all_truncated_timings, axis=0)
    print("Average timings: {}".format(mean_truncated_timings))

    Path("runs/plots").mkdir(parents=True, exist_ok=True)
    file_name = "pareto-{}-{}-seed{}-refmean{}-{}maxiterblock".format(obj_func_name,
                                                                      divergence,
                                                                      seed,
                                                                      ref_mean,
                                                                      scs_max_iter_block)
    fig = plt.figure()
    plt.scatter(mean_truncated_timings, truncated_regret, label='Truncated convex opt.')
    plt.scatter([mean_wcs_timing], [best_expectation - wcs_expectation], label='Worst case sens.')

    plt.xlabel("Timing")
    plt.ylabel("Regret")
    plt.legend()
    if show_plots:
        plt.show()
    fig.savefig("runs/plots/" + file_name + ".png")
    pickle.dump((true_adv_exps, wcs_adv_approxs, wcs_timings, all_truncated_exps, all_truncated_timings),
                open("runs/" + file_name + ".p", "wb"))
