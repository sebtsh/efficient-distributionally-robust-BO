import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment("DRBO_results")
ex.observers.append(FileStorageObserver('runs'))


@ex.named_config
def default():
    obj_func_name = 'rand_func'
    num_bo_iters = 200
    num_init_points = 10
    num_seeds = 10
    show_plots = True


@ex.automain
def main(obj_func_name, num_bo_iters, num_init_points, num_seeds, show_plots, figsize=(15, 6), dpi=None):
    Path("runs/results").mkdir(parents=True, exist_ok=True)
    divergences = ['MMD', 'TV', 'modified_chi_squared']
    acquisitions = ['GP-UCB', 'DRBOGeneral', 'DRBOWorstCaseSens', 'DRBOCubicApprox', 'WorstCaseSensTS', 'CubicApproxTS']
    x = np.arange(num_bo_iters)
    color_dict = {'GP-UCB': '#d7263d',
                  'DRBOGeneral': '#fbb13c',
                  'DRBOWorstCaseSens': '#26c485',
                  'DRBOCubicApprox': '#00a6ed',
                  'WorstCaseSensTS': '#9f956c',
                  'CubicApproxTS': '#2f4858'}
    for ref_mean in [0, 0.25, 0.5]: 
        for beta in [0, 0.5, 1, 2]:
            for divergence in divergences:
                fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
                plot_name = "{}-{}-beta{}-refmean{}".format(obj_func_name,
                                                            divergence,
                                                            beta,
                                                            ref_mean)
                print("Plotting " + plot_name)
                axs[0].set_title(plot_name + ": Immediate")
                axs[1].set_title(plot_name + ": Cumulative")
                for acquisition in acquisitions:
                    color = color_dict[acquisition]
                    all_regrets = np.zeros((num_seeds, num_bo_iters))
                    all_cumu_regrets = np.zeros((num_seeds, num_bo_iters))
                    all_times = []
                    for seed in range(num_seeds):
                        file_name = "{}-{}-{}-seed{}-beta{}-refmean{}.p".format(obj_func_name,
                                                                              divergence,
                                                                              acquisition,
                                                                              seed,
                                                                              beta,
                                                                              ref_mean)
                        regrets, cumulative_regrets, average_acq_time, query_points = pickle.load(
                            open("runs/" + file_name, "rb"))
                        # cut out initial points
                        regrets = np.array(regrets[num_init_points:])
                        base_cumulative_regret = cumulative_regrets[num_init_points-1]
                        cumulative_regrets = np.array(cumulative_regrets[num_init_points:]) - base_cumulative_regret
                        all_regrets[seed] = regrets
                        all_cumu_regrets[seed] = cumulative_regrets
                        all_times.append(average_acq_time)
                    mean_regrets = np.mean(all_regrets, axis=0)
                    std_err_regrets = np.std(all_regrets, axis=0) / np.sqrt(num_seeds)
                    mean_cumu_regrets = np.mean(all_cumu_regrets, axis=0)
                    std_err_cumu_regrets = np.std(all_cumu_regrets, axis=0) / np.sqrt(num_seeds)

                    # Immediate regret
                    axs[0].plot(x, mean_regrets, label=acquisition, color=color)
                    axs[0].fill_between(x, mean_regrets-std_err_regrets, mean_regrets+std_err_regrets,
                                           alpha=0.2, color=color)
                    axs[0].legend()

                    # Cumulative regret
                    axs[1].plot(x, mean_cumu_regrets, label=acquisition, color=color)
                    axs[1].fill_between(x, mean_cumu_regrets-std_err_cumu_regrets, mean_cumu_regrets+std_err_cumu_regrets,
                                           alpha=0.2, color=color)
                    axs[1].legend()

                    # Average acquisition time
                    print("{}-{} average acquisition time in seconds: {}".format(divergence,
                        acquisition,
                        np.mean(all_times)))
                fig.savefig("runs/results/" + plot_name +"-regret.png")
    if show_plots:
        plt.show()
