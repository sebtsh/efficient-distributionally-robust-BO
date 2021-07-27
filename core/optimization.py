import numpy as np

from gpflow.kernels import Kernel
from timeit import default_timer as timer
from tqdm import trange
from trieste.data import Dataset
from trieste.observer import SingleObserver
from trieste.type import TensorType
from typing import Callable

from core.acquisitions import Acquisition
from core.models import ModelOptModule
from core.utils import get_maximizer


def bayes_opt_loop(model: ModelOptModule,
                   init_dataset: Dataset or None,
                   search_points: TensorType,
                   observer: SingleObserver,
                   acq: Acquisition,
                   num_iters: int,
                   optimize_gp: bool = True):
    """
    Main Bayesian optimization loop.
    :param model:
    :param init_dataset:
    :param search_points:
    :param observer:
    :param acq:
    :param num_iters:
    :param optimize_gp:
    :return:
    """
    dataset = init_dataset
    maximizers = []
    model_params = []
    for t in trange(num_iters):
        # Acquire next input locations to sample
        next_inputs = acq.acquire(model, search_points, t)  # TensorType of shape (b, d)
        if next_inputs is None:  # Early termination signal
            print("Early termination at timestep t={}".format(t))
            return dataset, maximizers, model_params

        # Obtain observations
        next_dataset = observer(next_inputs)
        if dataset is not None:
            dataset = dataset + next_dataset
        else:
            dataset = next_dataset

        # Update model's dataset and optimize
        model.update_dataset(dataset)

        if optimize_gp:
            model.optimize()

        # Get model's belief of maximizer at this timestep
        maximizers.append(get_maximizer(model.gp, search_points))

        # Save model parameters at this timestep
        model_params.append(model.get_params())

    return dataset, maximizers, model_params


def bayes_opt_loop_dist_robust(model: ModelOptModule,
                               init_dataset: Dataset or None,
                               action_points: TensorType,
                               context_points: TensorType,
                               observer: SingleObserver,
                               acq: Acquisition,
                               num_iters: int,
                               reference_dist_func: Callable,
                               true_dist_func: Callable,
                               margin_func: Callable,
                               divergence: str,
                               mmd_kernel: Kernel,
                               optimize_gp: bool = True):
    """
    Main distributionally robust Bayesian optimization loop.
    :param model:
    :param init_dataset:
    :param action_points:
    :param context_points:
    :param observer:
    :param acq:
    :param num_iters:
    :param reference_dist_func: A function that takes in a timestep and returns an array of shape |C|, the size
    of the context set that is a valid probability distribution (sums to 1).
    :param true_dist_func: A function that takes in a timestep and returns an array of shape |C|, the size
    of the context set that is a valid probability distribution (sums to 1).
    :param margin_func: A function that takes in a timestep and returns epsilon_t
    :param divergence: str, 'MMD', 'TV' or 'modified_chi_squared''
    :param kernel: GPflow kernel. For MMD
    :param optimize_gp:
    :return:
    """
    dataset = init_dataset
    maximizers = []
    model_params = []
    times = []
    for t in trange(num_iters):
        # Get reference distribution
        ref_dist = reference_dist_func(t)
        epsilon = margin_func(t)

        # Acquire next input locations to sample
        start = timer()
        action = acq.acquire(model=model,
                             action_points=action_points,
                             context_points=context_points,
                             t=t,
                             ref_dist=ref_dist,
                             divergence=divergence,
                             kernel=mmd_kernel,
                             epsilon=epsilon)  # TensorType of shape (1, d_x)
        end = timer()
        times.append(end - start)  # Time taken to acquire in seconds
        if action is None:  # Early termination signal
            print("Early termination at timestep t={}".format(t))
            return dataset, maximizers, model_params

        # Environment samples context from true distribution
        true_dist = true_dist_func(t)
        context = context_points[np.random.choice(len(context_points), p=true_dist)][None, :]  # (1, d_c)

        # Obtain observations
        next_input = np.concatenate([action, context], axis=-1)
        next_dataset = observer(next_input)
        if dataset is not None:
            dataset = dataset + next_dataset
        else:
            dataset = next_dataset

        # Update model's dataset and optimize
        model.update_dataset(dataset)

        if optimize_gp:
            model.optimize()

        # Save model parameters at this timestep
        model_params.append(model.get_params())

    return dataset, model_params, np.mean(times)
