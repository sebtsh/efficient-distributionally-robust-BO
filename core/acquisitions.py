from abc import ABC, abstractmethod

import numpy as np
from trieste.type import TensorType

from core.models import ModelOptModule


def GP_UCB_point(model: ModelOptModule,
                 beta: float,
                 domain: TensorType):
    f_mean, f_var = model.predict_f(domain)
    scores = f_mean + beta * np.sqrt(f_var)
    return domain[np.argmax(scores)][None, :]


def get_acquisition(acq_name):
    """
    Acquisition function dispatcher.
    :param acq_name:
    :return:
    """
    args = dict(sorted(locals().items()))
    pass  # unimplemented


class Acquisition(ABC):
    """
    Abstract class that defines the necessary interface for acquisition functions.
    """

    def __init__(self):
        pass

    @abstractmethod
    def acquire(self,
                model: ModelOptModule,
                search_points: TensorType,
                t: int) -> TensorType:
        """
        Takes in models and a search space and returns an array of shape (b, d) that represents a batch selection
        of next points to query.
        :param model: list of ModelOptModule
        :param search_points: Array of shape (num_points, dims)
        :param t: Timestep
        :return: array of shape (b, d)
        """
        pass


class GP_UCB(Acquisition):
    """
    GP-UCB acquisition function from Srinivas et. al. (2010).
    """
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def acquire(self,
                model: ModelOptModule,
                search_points: TensorType,
                t):
        return GP_UCB_point(model, self.beta(t), search_points)
