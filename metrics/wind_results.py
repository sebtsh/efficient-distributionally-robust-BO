import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment("DRBO_results")
ex.observers.append(FileStorageObserver('../runs'))


@ex.named_config
def default():
    obj_func_name = 'wind'
    num_init_points = 10
    num_months = 12
    num_bo_iters = 673  # Minimum number of days in all months considered
    beta = 2
    seed = 0
    show_plots = False


@ex.automain
def main(obj_func_name, num_init_points, num_months, num_bo_iters, beta, seed, show_plots, figsize=(15, 6), dpi=None):
    dir = "runs/" + obj_func_name + "/results/"
    indiv_results_dir = "runs/" + obj_func_name + "/indiv_results/"
    Path(dir).mkdir(parents=True, exist_ok=True)

    divergences = ['MMD', 'MMD_approx', 'TV', 'modified_chi_squared']
    acquisitions = ['GP-UCB', 'DRBOGeneral', 'DRBOWorstCaseSens', 'DRBOMidApprox']

    color_dict = {'GP-UCB': '#d7263d',
                  'DRBOGeneral': '#fbb13c',
                  'DRBOWorstCaseSens': '#26c485',
                  'DRBOMidApprox': '#cbbf7a'}

    for divergence in divergences:
        x = np.arange(num_bo_iters)

        fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        plot_name = "{}-{}-beta{}".format(obj_func_name,
                                          divergence,
                                          beta)
        print("Plotting " + plot_name)
        axs[0].set_title(plot_name + ": Immediate")
        axs[1].set_title(plot_name + ": Cumulative")
        for acquisition in acquisitions:
            color = color_dict[acquisition]
            all_regrets = np.zeros((num_months, num_bo_iters))
            all_cumu_regrets = np.zeros((num_months, num_bo_iters))
            all_times = []
            for month in range(num_months):
                file_name = "{}-month{}-{}-{}-seed{}-beta{}.p".format(obj_func_name,
                                                                      month,
                                                                      divergence,
                                                                      acquisition,
                                                                      seed,
                                                                      beta)
                regrets, cumulative_regrets, average_acq_time, query_points = pickle.load(
                    open(indiv_results_dir + file_name, "rb"))
                # cut out initial points
                regrets = np.array(regrets[num_init_points:])
                base_cumulative_regret = cumulative_regrets[num_init_points - 1]
                cumulative_regrets = np.array(cumulative_regrets[num_init_points:]) - base_cumulative_regret

                # cut out anything above minimum number of months for uniformity across months
                regrets = regrets[:num_bo_iters]
                cumulative_regrets = cumulative_regrets[:num_bo_iters]

                all_regrets[month] = regrets
                all_cumu_regrets[month] = cumulative_regrets
                all_times.append(average_acq_time)
            mean_regrets = np.mean(all_regrets, axis=0)
            std_err_regrets = np.std(all_regrets, axis=0) / np.sqrt(num_months)
            mean_cumu_regrets = np.mean(all_cumu_regrets, axis=0)
            std_err_cumu_regrets = np.std(all_cumu_regrets, axis=0) / np.sqrt(num_months)

            # Immediate regret
            axs[0].plot(x, mean_regrets, label=acquisition, color=color)
            axs[0].fill_between(x, mean_regrets - std_err_regrets, mean_regrets + std_err_regrets,
                                alpha=0.2, color=color)
            axs[0].legend()

            # Cumulative regret
            axs[1].plot(x, mean_cumu_regrets, label=acquisition, color=color)
            axs[1].fill_between(x, mean_cumu_regrets - std_err_cumu_regrets,
                                mean_cumu_regrets + std_err_cumu_regrets,
                                alpha=0.2, color=color)
            axs[1].legend()

            # Average acquisition time
            print("{}-{} average acquisition time in seconds: {}".format(divergence,
                                                                         acquisition,
                                                                         np.mean(all_times)))
        fig.savefig(dir + plot_name + "-regret.png")
    if show_plots:
        plt.show()
