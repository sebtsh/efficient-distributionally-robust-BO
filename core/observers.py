import numpy as np
from collections.abc import Callable
from trieste.observer import SingleObserver
from trieste.data import Dataset


def mk_noisy_observer(objective: Callable,
                      std: float) -> SingleObserver:
    """
    Observer that adds Gaussian noise to the function values.
    :param objective: Objective function that takes in arrays of shape (n, d) and returns an array of shape (n, 1)
    :param std: float. Noise is drawn from a Gaussian distribution with mean 0 and the given standard deviation
    """
    return lambda qp: Dataset(qp, objective(qp) + np.random.normal(0, std, objective(qp).shape))
