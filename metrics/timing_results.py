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
def rand_func():
    show_plots = False
    figsize = (8, 6)
    dpi = 200


@ex.automain
def main(show_plots, figsize, dpi):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif"})
    text_size = 26
    tick_size = 20

    result_dir = "runs/timing/"
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    context_grid_densities = np.arange(200, 1800, 200)
    divergences = ['MMD_approx', 'TV', 'modified_chi_squared']
    acquisitions = ['GP-UCB', 'DRBOGeneral', 'DRBOWorstCaseSens', 'DRBOMidApprox']
    color_dict = {'GP-UCB': '#d7263d',
                  'DRBOGeneral': '#fbb13c',
                  'DRBOWorstCaseSens': '#26c485',
                  'DRBOMidApprox': '#00a6ed'}
    marker_dict = {'GP-UCB': '8',
                  'DRBOGeneral': 's',
                  'DRBOWorstCaseSens': 'p',
                  'DRBOMidApprox': 'P'}

    acq_name_dict = {'GP-UCB': 'GP-UCB',
                     'DRBOGeneral': 'DRBOGeneral',
                     'DRBOWorstCaseSens': 'WCS',
                     'DRBOMidApprox': 'MinimaxApprox'}

    for divergence in divergences:
        timing_dict = pickle.load(open(result_dir + f"timing_dict-{divergence}.p", "rb"))

        # Plots
        plt.figure(figsize=figsize, dpi=dpi)
        for acquisition in acquisitions:
            plt.plot(context_grid_densities, timing_dict[acquisition], label=acq_name_dict[acquisition],
                     color=color_dict[acquisition], marker=marker_dict[acquisition])
        #plt.title("{} average acquisition time in seconds".format(divergence))
        plt.xlabel("Size of context set", size=text_size)
        plt.ylabel("Seconds", size=text_size)
        plt.legend(fontsize=text_size-2)
        plt.xticks(size=tick_size)
        plt.yticks(size=tick_size)
        plt.savefig(result_dir + "{}-timing.png".format(divergence), figsize=figsize, dpi=dpi, bbox_inches='tight')

    if show_plots:
        plt.show()