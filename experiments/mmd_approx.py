import gpflow as gpf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from sacred import Experiment
from sacred.observers import FileStorageObserver

sys.path.append(sys.path[0][:-len('experiments')])  # for imports to work
print(sys.path)

from core.objectives import get_obj_func
from core.utils import construct_grid_1d, cross_product, get_discrete_normal_dist_1d, \
    get_action_contexts, worst_case_sens, get_mid_approx_func, get_ordering, kendall_tau_distance

ex = Experiment("DRBO")
ex.observers.append(FileStorageObserver('../runs'))


@ex.named_config
def default():
    dims = 2
    lowers = [0] * dims
    uppers = [1] * dims
    rand_func_num_points = 100
    ls = 0.05
    ref_var = 0.02
    figsize = (12, 6)
    dpi = 200


@ex.automain
def main(lowers, uppers, rand_func_num_points, dims, ls, ref_var, figsize, dpi):
    obj_func_name = 'rand_func'
    exp_name = "mmd_approx_quality"
    sum_results_dir = "runs/" + exp_name + "/summarized_results/"
    Path(sum_results_dir).mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams['text.usetex'] = True

    grid_densities = [50, 100, 500, 1000]

    divergence = 'MMD_approx'
    ref_means = [0, 0.5]
    seeds = np.arange(0, 5)
    seed_res = []
    all_eps = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    color = '#00a6ed'
    text_size = 20
    tick_size = 16

    for seed in seeds:
        density_res = []
        for grid_density_per_dim in grid_densities:
            print(f"Computing for grid density {grid_density_per_dim}")
            # Action space (1d for now)
            action_points = construct_grid_1d(lowers[0], uppers[0], 20)
            # Context space (1d for now)
            context_points = construct_grid_1d(lowers[1], uppers[1], grid_density_per_dim)

            # We can do this because not optimizing kernel
            f_kernel = gpf.kernels.SquaredExponential(lengthscales=[ls] * dims)
            if divergence == 'MMD' or divergence == 'MMD_approx':
                mmd_kernel = gpf.kernels.SquaredExponential(lengthscales=[ls])  # 1d for now
            else:
                mmd_kernel = None

            # Get objective function
            obj_func = get_obj_func(obj_func_name, lowers, uppers, f_kernel, rand_func_num_points, seed)
            ref_means_positions_per_eps = []
            for ref_mean in ref_means:
                ref_dist_func = lambda x: get_discrete_normal_dist_1d(context_points, ref_mean, ref_var)

                num_actions = len(action_points)
                num_context_points = len(context_points)
                domain = cross_product(action_points, context_points)

                all_true = []
                all_approx = []

                all_actions_true_mm_scores = []
                all_actions_approx_mm_scores = []

                for i in range(num_actions):
                    action_contexts = get_action_contexts(i, domain, num_context_points)
                    f = np.squeeze(obj_func(action_contexts), axis=-1)
                    wcs_true = worst_case_sens(fvals=f,
                                               p=ref_dist_func(0),
                                               context_points=context_points,
                                               kernel=mmd_kernel,
                                               divergence='MMD')
                    wcs_approx = worst_case_sens(fvals=f,
                                                 p=ref_dist_func(0),
                                                 context_points=context_points,
                                                 kernel=mmd_kernel,
                                                 divergence='MMD_approx')
                    true_minimax = get_mid_approx_func(context_points,
                                                       f,
                                                       mmd_kernel,
                                                       ref_dist_func(0),
                                                       wcs_true,
                                                       'MMD')
                    approx_minimax = get_mid_approx_func(context_points,
                                                         f,
                                                         mmd_kernel,
                                                         ref_dist_func(0),
                                                         wcs_approx,
                                                         'MMD_approx')
                    true_mm_scores = []
                    approx_mm_scores = []
                    for eps in all_eps:
                        true_mm_score = true_minimax(eps)
                        approx_mm_score = approx_minimax(eps)
                        true_mm_scores.append(true_mm_score)
                        approx_mm_scores.append(approx_mm_score)
                    all_actions_true_mm_scores.append(true_mm_scores)
                    all_actions_approx_mm_scores.append(approx_mm_scores)

                    all_true.append(wcs_true)
                    all_approx.append(wcs_approx)

                all_actions_true_mm_scores = np.array(all_actions_true_mm_scores)  # (num_actions, num_eps)
                all_actions_approx_mm_scores = np.array(all_actions_approx_mm_scores)

                positions_per_eps = []
                for i, eps in enumerate(all_eps):
                    true_ordering = get_ordering(all_actions_true_mm_scores[:, i])
                    approx_ordering = get_ordering(all_actions_approx_mm_scores[:, i])
                    approx_best = approx_ordering[-1]
                    positions_per_eps.append(20 - list(true_ordering).index(approx_best))
                positions_per_eps = np.array(positions_per_eps)  # (num_eps, )
                ref_means_positions_per_eps.append(positions_per_eps)
            ref_means_positions_per_eps = np.array(ref_means_positions_per_eps)  # (num_ref_means, num_eps)
            density_res.append(ref_means_positions_per_eps)
        density_res = np.array(density_res)  # (num_densities, num_ref_means, num_eps)
        seed_res.append(density_res)
    seed_res = np.array(seed_res)  # (num_seeds, num_densities, num_ref_means, num_eps)
    mean_res = np.mean(seed_res, axis=0)  # (num_densities, num_ref_means, num_eps)
    stderr_res = np.std(seed_res, axis=0)/np.sqrt(len(seeds))  # (num_densities, num_ref_means, num_eps)

    fig, axs = plt.subplots(len(ref_means), len(grid_densities), figsize=figsize, dpi=dpi)
    for i, ref_mean in enumerate(ref_means):
        for j, density in enumerate(grid_densities):
            mean = mean_res[j, i]
            stderr = stderr_res[j, i]
            axs[i][j].plot(all_eps, mean, color=color)
            axs[i][j].fill_between(all_eps, mean - stderr,
                                mean + stderr,
                                alpha=0.2, color=color)
            axs[i][j].set_xlabel("$\epsilon$", size=text_size)
            if j == 0:
                axs[i][j].set_ylabel("Rank", size=text_size)
            axs[i][j].set_ylim(0, 20)
            axs[i][j].set_yticks([1, 5, 10, 15, 20])
            axs[i][j].axhline(y=1, linestyle='--', color='black')
            axs[i][j].set_title(r"$|\mathcal C|$ = " + str(density), size=text_size)
            axs[i][j].tick_params(labelsize=tick_size)
    fig.tight_layout()
    fig.savefig(sum_results_dir + "mmd_approx_quality.pdf", bbox_inches='tight', format='pdf')
