import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import gpflow as gpf

from core.utils import cumu_robust_regrets, construct_grid_1d, cross_product, normalize_dist, \
    get_discrete_uniform_dist, get_discrete_normal_dist_1d, get_margin, get_robust_expectation_and_action, \
    get_discrete_normal_dist, imm_robust_regret
from core.objectives import get_obj_func

regret_type = 'cumu'

obj_func_name = 'plant'
show_plots = False
beta = 2
pri_divergences = ['MMD_approx', 'TV', 'modified_chi_squared']
sec_divergences = ['MMD_approx', 'TV', 'modified_chi_squared']
acquisition = 'DRBOGeneral'

dir = "runs/" + obj_func_name + "/"
result_dir = dir + "indiv_results/"
sum_results_dir = "runs/cross/" + obj_func_name + "/"
Path(sum_results_dir).mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif"})
text_size = 28
tick_size = 20
color_dict = {'MMD_approx': '#d7263d',
              'TV': '#fbb13c',
              'modified_chi_squared': '#00a6ed'}
figsize = (10, 6)
dpi = 200

if obj_func_name == 'rand_func':
    num_bo_iters = 1000
    num_seeds = 10
    ref_means = [0, 0.5]
elif obj_func_name == 'plant':
    num_bo_iters = 1000
    num_seeds = 10
    ref_means = np.array([[0.], [1.]])


def get_params(obj_func_name, ref_mean):
    if obj_func_name == 'rand_func':
        num_init_points = 10

        dims = 2
        lowers = [0] * dims
        uppers = [1] * dims
        grid_density_per_dim = 20
        rand_func_num_points = 100
        ls = 0.05
        ref_var = 0.02

        f_kernel = gpf.kernels.SquaredExponential(lengthscales=[ls] * dims)
        mmd_kernel = gpf.kernels.SquaredExponential(lengthscales=[ls])
        obj_func = get_obj_func(obj_func_name, lowers, uppers, f_kernel, 1, rand_func_num_points, seed)
        action_points = construct_grid_1d(lowers[0], uppers[0], grid_density_per_dim)
        context_points = construct_grid_1d(lowers[1], uppers[1], grid_density_per_dim)
        # Distribution generating functions
        if divergence == 'modified_chi_squared':  # Add small uniform everywhere for numeric reasons
            ref_dist_func = lambda x: normalize_dist(
                get_discrete_normal_dist_1d(context_points, ref_mean, ref_var) +
                get_discrete_uniform_dist(context_points) / 100)
        else:
            ref_dist_func = lambda x: get_discrete_normal_dist_1d(context_points, ref_mean, ref_var)
        true_dist_func = lambda x: get_discrete_uniform_dist(context_points)
        margin = get_margin(ref_dist_func(0), true_dist_func(0), mmd_kernel, context_points, divergence)
        margin_func = lambda x: margin  # Constant margin for now
    elif obj_func_name == 'plant':
        action_dims = 1
        context_dims = 1
        action_lowers = [0] * action_dims
        action_uppers = [1] * action_dims
        context_lowers = [0] * context_dims
        context_uppers = [1] * context_dims
        action_density_per_dim = 50
        context_density_per_dim = 50
        ls = [0.1] * (action_dims + context_dims)
        num_init_points = 10
        ref_var = 0.02
        ref_cov = ref_var * np.eye(context_dims)
        all_lowers = action_lowers + context_lowers
        all_uppers = action_uppers + context_uppers
        lengthscales = np.array(ls)
        context_lengthscales = lengthscales[-context_dims:]
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

        # Warning: move kernels into inner loop if optimizing kernel
        f_kernel = gpf.kernels.SquaredExponential(lengthscales=lengthscales)
        if divergence == 'MMD' or divergence == 'MMD_approx':
            mmd_kernel = gpf.kernels.SquaredExponential(lengthscales=context_lengthscales)
        else:
            mmd_kernel = None

        # Get objective function
        obj_func = get_obj_func(obj_func_name, all_lowers, all_uppers, f_kernel, context_dims)

        # Distribution generating functions
        if divergence == 'modified_chi_squared':
            ref_dist_func = lambda x: normalize_dist(get_discrete_normal_dist(context_points, ref_mean, ref_cov) +
                                                     get_discrete_uniform_dist(context_points) / 100)
        else:
            ref_dist_func = lambda x: get_discrete_normal_dist(context_points, ref_mean, ref_cov)
        true_dist_func = lambda x: get_discrete_uniform_dist(context_points)
        margin = get_margin(ref_dist_func(0), true_dist_func(0), mmd_kernel, context_points, divergence)
        margin_func = lambda x: margin  # Constant margin for now
    else:
        raise Exception("wrong obj_func_name to get_params")

    x = np.arange(num_bo_iters)
    return x, num_init_points, action_points, context_points, mmd_kernel, \
           ref_dist_func, margin_func, obj_func


if regret_type == 'cumu':
    for divergence in pri_divergences:
        print(f"===========Primary divergence: {divergence}============")
        fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        for i, ref_mean in enumerate(ref_means):
            plot_name = f"Ref. mean = {ref_mean}"
            print(f"====={plot_name}=====")
            axs[i].set_title(plot_name, size=text_size)
            for div in sec_divergences:
                print(f"===Secondary divergence: {div}===")
                color = color_dict[div]
                all_regrets = np.zeros((num_seeds, num_bo_iters))
                all_times = []
                all_imm_regrets = []
                for seed in range(num_seeds):
                    print(f"seed: {seed}")
                    file_name = "{}-{}-{}-seed{}-beta{}-refmean{}.p".format(obj_func_name,
                                                                            div,
                                                                            acquisition,
                                                                            seed,
                                                                            beta,
                                                                            ref_mean)
                    _, __, ___, query_points = pickle.load(open(result_dir + file_name, "rb"))

                    # Goal: Get cumulative regrets, but measured by the first divergence's optimal robust point
                    x, num_init_points, action_points, context_points, \
                    mmd_kernel, ref_dist_func, margin_func, obj_func = get_params(obj_func_name, ref_mean)

                    print("Calculating robust expectation and action")
                    robust_expectation, robust_action = get_robust_expectation_and_action(action_points=action_points,
                                                                                          context_points=context_points,
                                                                                          kernel=mmd_kernel,
                                                                                          fvals_source='obj_func',
                                                                                          ref_dist=ref_dist_func(0),
                                                                                          divergence=divergence,
                                                                                          epsilon=margin_func(0),
                                                                                          obj_func=obj_func)
                    robust_expectation_action = (robust_expectation, robust_action)

                    print("Calculating cumulative regrets")
                    cumulative_regrets = cumu_robust_regrets(obj_func,
                                                             query_points,
                                                             action_points,
                                                             context_points,
                                                             mmd_kernel,
                                                             ref_dist_func,
                                                             margin_func,
                                                             divergence,
                                                             robust_expectation_action)

                    # cut out initial points
                    base_cumulative_regret = cumulative_regrets[num_init_points - 1]
                    cumulative_regrets = np.array(cumulative_regrets[num_init_points:]) - base_cumulative_regret
                    all_regrets[seed] = cumulative_regrets

                mean_regrets = np.mean(all_regrets, axis=0)
                std_err_regrets = np.std(all_regrets, axis=0) / np.sqrt(num_seeds)

                if div == 'MMD_approx':
                    div_name = 'MMD'
                elif div == 'TV':
                    div_name = 'TV'
                elif div == 'modified_chi_squared':
                    div_name = '$\\chi^2$'
                axs[i].plot(x, mean_regrets, label=div_name, color=color)
                axs[i].fill_between(x, mean_regrets - std_err_regrets,
                                    mean_regrets + std_err_regrets,
                                    alpha=0.2, color=color)
            axs[i].legend(fontsize=20)
            axs[i].set_xlabel("Iterations", size=text_size)
            axs[i].set_ylabel("Cumulative robust regret", size=text_size)
            axs[i].tick_params(labelsize=tick_size)

        fig.tight_layout()
        fig.savefig(sum_results_dir + f"{obj_func_name}-{acquisition}-{divergence}-crosscomparison.pdf", figsize=figsize,
                    dpi=dpi,
                    bbox_inches='tight', format='pdf')
elif regret_type == 'imm':
    for divergence in pri_divergences:
        print(f"===========Primary divergence: {divergence}============")
        fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        for i, ref_mean in enumerate(ref_means):
            plot_name = f"Ref. mean = {ref_mean}"
            print(f"====={plot_name}=====")
            axs[i].set_title(plot_name, size=text_size)
            for div in sec_divergences:
                print(f"===Secondary divergence: {div}===")
                color = color_dict[div]
                all_regrets = np.zeros((num_seeds, num_bo_iters))
                all_times = []
                all_imm_regrets = []
                for seed in range(num_seeds):
                    print(f"seed: {seed}")
                    file_name = "{}-{}-{}-seed{}-beta{}-refmean{}.p".format(obj_func_name,
                                                                            div,
                                                                            acquisition,
                                                                            seed,
                                                                            beta,
                                                                            ref_mean)
                    _, __, ___, query_points = pickle.load(open(result_dir + file_name, "rb"))
                    print(f"Last 5 query points: {query_points[-5:, 0]}")

                    # Goal: Get cumulative regrets, but measured by the first divergence's optimal robust point
                    x, num_init_points, action_points, context_points, \
                    mmd_kernel, ref_dist_func, margin_func, obj_func = get_params(obj_func_name, ref_mean)

                    #print("Calculating robust expectation and action")
                    robust_expectation, robust_action = get_robust_expectation_and_action(action_points=action_points,
                                                                                          context_points=context_points,
                                                                                          kernel=mmd_kernel,
                                                                                          fvals_source='obj_func',
                                                                                          ref_dist=ref_dist_func(0),
                                                                                          divergence=divergence,
                                                                                          epsilon=margin_func(0),
                                                                                          obj_func=obj_func)
                    robust_expectation_action = (robust_expectation, robust_action)

                    #print("Calculating immediate regrets at end")
                    imm_regret = imm_robust_regret(obj_func,
                                                   query_points,
                                                   action_points,
                                                   context_points,
                                                   mmd_kernel,
                                                   ref_dist_func,
                                                   margin_func,
                                                   divergence,
                                                   robust_expectation_action)
                    all_imm_regrets.append(imm_regret)

                mean_regrets = np.mean(all_imm_regrets, axis=0)
                std_err_regrets = np.std(all_imm_regrets, axis=0) / np.sqrt(num_seeds)
                print("Imm regret: {0:1.5f} +- {1:1.5f}".format(mean_regrets, std_err_regrets))
