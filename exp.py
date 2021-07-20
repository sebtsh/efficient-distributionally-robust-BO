import gpflow as gpf
import matplotlib.pyplot as plt
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from trieste.space import Box

from core.acquisitions import get_acquisition
from core.models import GPRModule
from core.objectives import standardize_objective, get_obj_func
from core.observers import mk_noisy_observer
from core.optimization import bayes_opt_loop
from core.utils import discretize_Box
from metrics.plotting import plot_function_2d, plot_bo_points_2d, plot_regret

ex = Experiment("SafeBO")
ex.observers.append(FileStorageObserver('runs'))


@ex.named_config
def gpucb():
    acq_name = 'GP-UCB'
    obj_func_name = 'rand_func'
    is_standardizing_obj = False
    dims = 2
    lowers = [0] * dims
    uppers = [1] * dims
    grid_density_per_dim = 100
    rand_func_num_points = 100
    ls = 0.1
    obs_variance = 0.001
    is_optimizing_gp = False
    opt_max_iter = 10
    num_bo_iters = 50
    num_init_points = 3
    beta_const = 3
    seed = 0


@ex.automain
def main(acq_name, obj_func_name, is_standardizing_obj, lowers, uppers, grid_density_per_dim, rand_func_num_points,
         dims, ls, obs_variance, is_optimizing_gp, num_bo_iters, opt_max_iter, num_init_points, beta_const, seed):
    np.random.seed(0)

    kernel = gpf.kernels.SquaredExponential(lengthscales=[ls] * dims)
    # Get objective function
    obj_func = get_obj_func(obj_func_name, lowers, uppers, kernel, rand_func_num_points, seed)
    if is_standardizing_obj:
        obj_func = standardize_objective(obj_func, lowers, uppers, grid_density_per_dim)

    # Search space
    search_space = discretize_Box(Box(lowers, uppers), grid_density_per_dim)
    search_points = search_space.points.numpy()

    observer = mk_noisy_observer(obj_func, obs_variance)
    init_dataset = observer(search_points[np.random.randint(0, len(search_points), num_init_points)])

    # Model
    model = GPRModule(dims=dims,
                      kernel=kernel,
                      noise_variance=obs_variance,
                      dataset=init_dataset,
                      opt_max_iter=opt_max_iter)

    # Acquisition
    acquisition = get_acquisition(acq_name=acq_name,
                                  beta=lambda x: beta_const)  # TODO: Implement beta function

    # Main BO loop
    final_dataset, maximizers, model_params = bayes_opt_loop(model=model,
                                                             init_dataset=init_dataset,
                                                             search_points=search_points,
                                                             observer=observer,
                                                             acq=acquisition,
                                                             num_iters=num_bo_iters,
                                                             optimize_gp=is_optimizing_gp)
    print("Final dataset: {}".format(final_dataset))
    print("Maximizers: {}".format(np.array(maximizers)))
    # Plots
    query_points = final_dataset.query_points.numpy()
    maximizer = search_points[[np.argmax(obj_func(search_points))]]
    if dims == 2:
        title = obj_func_name

        _, ax = plot_function_2d(obj_func, lowers, uppers, grid_density_per_dim, contour=True,
                                 title=title, colorbar=True)
        plot_bo_points_2d(query_points, ax, num_init=num_init_points, maximizer=maximizer)

    _, ax = plot_regret(obj_func, np.array(maximizers), search_points, query_points[num_init_points:],
                        regret_type='immediate',
                        fvals_source='best_seen')
    plt.show()
