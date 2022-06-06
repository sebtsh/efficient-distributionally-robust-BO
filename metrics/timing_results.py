import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
from sacred import Experiment
from sacred.observers import FileStorageObserver
from pathlib import Path

sys.path.append(sys.path[0][:-len('experiments')])  # for imports to work
print(sys.path)

ex = Experiment("DRBO_timing")
ex.observers.append(FileStorageObserver('../runs'))


@ex.named_config
def default():
    show_plots = False
    figsize = (8, 6)
    dpi = 200


@ex.automain
def main(show_plots, figsize, dpi):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif"})
    text_size = 26
    tick_size = 20

    result_dir = "runs/timing/"
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    context_grid_densities = np.arange(100, 901, 100)
    divergences = ['MMD_approx', 'TV', 'modified_chi_squared', 'wass']
    acquisitions = ['GP-UCB', 'DRBOGeneral', 'WorstCaseSens', 'MinimaxApprox']
    color_dict = {'GP-UCB': '#d7263d',
                  'DRBOGeneral': '#fbb13c',
                  'WorstCaseSens': '#26c485',
                  'MinimaxApprox': '#00a6ed'}
    marker_dict = {'GP-UCB': 's',
                   'DRBOGeneral': '8',
                   'WorstCaseSens': 'x',
                   'MinimaxApprox': 'P'}

    acq_name_dict = {'GP-UCB': 'GP-UCB',
                     'DRBOGeneral': 'Exact',
                     'WorstCaseSens': 'WCS',
                     'MinimaxApprox': 'MinimaxApprox'}

    for divergence in divergences:
        timing_dict = pickle.load(open(result_dir + f"timing_dict-{divergence}.p", "rb"))

        # Plots
        plt.figure(figsize=figsize, dpi=dpi)
        for acquisition in acquisitions:
            plt.plot(context_grid_densities, timing_dict[acquisition], label="\\textsc{" + acq_name_dict[acquisition] + "}",
                     color=matplotlib.colors.to_rgba(color_dict[acquisition], 0.7), marker=marker_dict[acquisition], markersize=20)
        # plt.title("{} average acquisition time in seconds".format(divergence))
        plt.xlabel("Size of context set", size=text_size)
        plt.ylabel("Mean CPU time (seconds)", size=text_size)
        plt.legend(fontsize=text_size - 2)
        plt.xticks(size=tick_size)
        plt.yticks(size=tick_size)
        plt.savefig(result_dir + "{}-timing.pdf".format(divergence), figsize=figsize, dpi=dpi, bbox_inches='tight',
                    format='pdf')

    if show_plots:
        plt.show()
