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

matplotlib.use('Agg')

ex = Experiment("Pareto")
ex.observers.append(FileStorageObserver('../runs'))


@ex.named_config
def rand_func():
    obj_func_name = 'rand_func'
    divergence = 'MMD_approx'  # 'MMD', 'TV' or 'modified_chi_squared'' or 'modified_chi_squared'
    context_density_per_dim = 100
    ref_mean = 0.5
    scs_max_iter_block = 400
    seed = 0
    figsize = (8, 6)
    dpi = 200
    show_plots = False


@ex.automain
def main(obj_func_name, divergence, context_density_per_dim, ref_mean, scs_max_iter_block, seed, figsize, dpi,
         show_plots):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif"})
    text_size = 26
    tick_size = 20

    result_dir = "runs/pareto/"
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    file_name = "pareto-{}-{}-seed{}-refmean{}-maxiterblock{}-cdensity{}".format(obj_func_name,
                                                                                 divergence,
                                                                                 seed,
                                                                                 ref_mean,
                                                                                 scs_max_iter_block,
                                                                                 context_density_per_dim)
    true_adv_exps, wcs_adv_approxs, wcs_timings, all_truncated_exps, all_truncated_timings = pickle.load(
        open(result_dir + file_name + ".p", "rb"))

    wcs_approx_error = abs(true_adv_exps - wcs_adv_approxs)
    print("WCS approximation errors = {}".format(wcs_approx_error))
    wcs_average_error = np.mean(wcs_approx_error)
    print("WCS average approximation error = {}".format(wcs_average_error))

    wcs_average_timing = np.mean(wcs_timings)
    print("WCS timing: {}".format(wcs_average_timing))

    truncated_error = abs(true_adv_exps[:, None] - all_truncated_exps)  # (num_actions, num_scs_max_iters)
    truncated_average_error = np.mean(truncated_error, axis=0)
    print("Truncated average approximation error = {}".format(truncated_average_error))

    truncated_average_timings = np.mean(all_truncated_timings, axis=0)
    print("Average timings: {}".format(truncated_average_timings))

    plt.scatter(truncated_average_timings, truncated_average_error, label='Truncated convex opt.', color='#fbb13c')
    plt.scatter([wcs_average_timing], [wcs_average_error], label='\\textsc{MinimaxApprox}', color='#00a6ed')

    plt.xlabel("Mean time elapsed (seconds)", size=text_size)
    plt.ylabel("Mean approximation error", size=text_size)
    plt.legend(fontsize=text_size - 4)
    plt.xticks(size=tick_size)
    plt.yticks(size=tick_size)
    if show_plots:
        plt.show()
    fig = plt.gcf()
    fig.savefig(result_dir + 'pareto' + ".pdf", figsize=figsize, dpi=dpi, bbox_inches='tight', format='pdf')
