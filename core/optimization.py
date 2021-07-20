from tqdm import trange
from trieste.data import Dataset
from trieste.observer import SingleObserver
from trieste.type import TensorType

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
