from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from trieste.type import TensorType

from core.models import ModelOptModule
from core.utils import get_upper_lower_bounds


def GP_UCB_point(model: ModelOptModule,
                 beta: float,
                 domain: TensorType):
    f_mean, f_var = model.predict_f(domain)
    scores = f_mean + beta * np.sqrt(f_var)
    return domain[np.argmax(scores)][None, :]


def get_acquisition(acq_name,
                    beta: Callable):
    """
    Acquisition function dispatcher.
    :param acq_name:
    :param beta: Function of timestep
    :return:
    """
    args = dict(sorted(locals().items()))
    if acq_name == 'GP-UCB':
        return GPUCBStochastic(**args)
    else:
        raise Exception('Acquisition name is wrong')


class Acquisition(ABC):
    """
    Abstract class that defines the necessary interface for acquisition functions.
    """

    def __init__(self):
        pass

    @abstractmethod
    def acquire(self,
                model: ModelOptModule,
                action_points: TensorType,
                context_points: TensorType,
                t: int,
                ref_dist: TensorType,
                epsilon: float) -> TensorType:
        """
        Takes in models and a search space and returns an array of shape (b, d) that represents a batch selection
        of next points to query.
        :param model: list of ModelOptModule
        :param action_points: Array of shape (num_points, dims)
        :param context_points: TensorType
        :param t: Timestep
        :param ref_dist: Array of shape |C| that is a probability distribution
        :param epsilon: margin
        :return: array of shape (b, d)
        """
        pass


class GPUCBStochastic(Acquisition):
    """
    GP-UCB acquisition function from Srinivas et. al. (2010). Chooses argmax of the expected ucb under the reference
    distribution.
    """
    def __init__(self,
                 beta: Callable,
                 **kwargs):
        super().__init__()
        self.beta = beta

    def acquire(self,
                model: ModelOptModule,
                action_points: TensorType,
                context_points: TensorType,
                t: int,
                ref_dist: TensorType,
                epsilon: float):
        num_search_points = len(action_points)
        num_context_points = len(context_points)
        max_val = -np.infty
        for i in range(num_search_points):
            tiled_action = np.tile(action_points[i:i + 1], (num_context_points, 1))  # (num_context_points, d_x)
            action_contexts = np.concatenate([tiled_action, context_points], axis=-1)  # (num_context_points, d_x + d_c)
            upper_bounds, _ = get_upper_lower_bounds(model, action_contexts, self.beta(t))  # (num_context_points, )
            expected_upper = np.sum(ref_dist * upper_bounds)
            if expected_upper > max_val:
                max_val = expected_upper
                max_idx = i
        return action_points[max_idx:max_idx + 1]
