from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from gpflow.kernels import Kernel
from trieste.type import TensorType

from core.models import ModelOptModule
from core.utils import get_upper_lower_bounds, get_robust_expectation_and_action, cholesky_inverse


def GP_UCB_point(model: ModelOptModule,
                 beta: float,
                 domain: TensorType):
    f_mean, f_var = model.predict_f(domain)
    scores = f_mean + beta * np.sqrt(f_var)
    return domain[np.argmax(scores)][None, :]


def get_acquisition(acq_name,
                    beta: Callable,
                    divergence: str):
    """
    Acquisition function dispatcher.
    :param acq_name:
    :param beta: Function of timestep
    :param divergence:
    :return:
    """
    args = dict(sorted(locals().items()))
    if acq_name == 'GP-UCB':
        return GPUCBStochastic(**args)
    elif acq_name == 'DRBOGeneral':
        return DRBOGeneral(**args)
    elif acq_name == 'DRBOWorstCaseSens':
        return DRBOWorstCaseSens(**args)
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
                divergence: str,
                kernel: Kernel,
                epsilon: float) -> TensorType:
        """
        Takes in models and a search space and returns an array of shape (b, d) that represents a batch selection
        of next points to query.
        :param model: list of ModelOptModule
        :param action_points: Array of shape (num_points, dims)
        :param context_points: TensorType
        :param t: Timestep
        :param ref_dist: Array of shape |C| that is a probability distribution
        :param divergence: str, 'MMD', 'TV' or 'modified_chi_squared''
        :param kernel: gpflow kernel
        :param epsilon: margin
        :return: array of shape (b, d)
        """
        pass


class DRBOGeneral(Acquisition):
    """
    Distributionally robust BO Algorithm 1 from Kirschner et. al. (2020).
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
                divergence: str,
                kernel: Kernel,
                epsilon: float):
        robust_expectation, robust_action = get_robust_expectation_and_action(action_points=action_points,
                                                                              context_points=context_points,
                                                                              kernel=kernel,
                                                                              fvals_source='ucb',
                                                                              ref_dist=ref_dist,
                                                                              divergence=divergence,
                                                                              epsilon=epsilon,
                                                                              model=model,
                                                                              beta=self.beta(t))
        return robust_action


class DRBOWorstCaseSens(Acquisition):
    """
    Uses worst case sensitivity (Gotoh et. al., 2020) as a fast upper bound for the distributionally robust
    maximin problem.
    """

    def __init__(self,
                 beta: Callable,
                 divergence: str,  # "TV", "modified_chi_squared"
                 **kwargs):
        super().__init__()
        self.beta = beta
        self.divergence = divergence

    def acquire(self,
                model: ModelOptModule,
                action_points: TensorType,
                context_points: TensorType,
                t: int,
                ref_dist: TensorType,
                divergence: str,
                kernel: Kernel,
                epsilon: float):
        num_action_points = len(action_points)
        num_context_points = len(context_points)
        adv_lower_bounds = []
        for i in range(num_action_points):
            tiled_action = np.tile(action_points[i:i + 1], (num_context_points, 1))  # (num_context_points, d_x)
            action_contexts = np.concatenate([tiled_action, context_points],
                                             axis=-1)  # (num_context_points, d_x + d_c)
            ucb_vals, _ = get_upper_lower_bounds(model, action_contexts, self.beta(t))  # (num_context_points, )
            expected_ucb = np.sum(ref_dist * ucb_vals)  # SAA
            if divergence == 'MMD':
                K_inv = cholesky_inverse(kernel(context_points))
                f_T_K_inv = ucb_vals[None, :] @ K_inv  # (1, num_context_points)
                worst_case_sensitivity = np.sqrt(f_T_K_inv @ ucb_vals[:, None] -
                                                 (np.squeeze(f_T_K_inv @ np.ones((num_context_points, 1))) ** 2) /
                                                 np.sum(K_inv))
                sens_factor = epsilon
            elif divergence == 'TV':
                worst_case_sensitivity = 0.5 * (np.max(ucb_vals) - np.min(ucb_vals))
                sens_factor = epsilon  # might be square root epsilon for others
            elif divergence == 'modified_chi_squared':
                worst_case_sensitivity = np.sqrt(2 * np.var(ucb_vals))
                sens_factor = np.sqrt(epsilon)
            else:
                raise Exception("Invalid divergence passed to DRBOWorstCaseSens")
            adv_lower_bound = expected_ucb - (sens_factor * worst_case_sensitivity)
            adv_lower_bounds.append(adv_lower_bound)
        max_idx = np.argmax(adv_lower_bounds)
        return action_points[max_idx:max_idx + 1]


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
                divergence: str,
                kernel: Kernel,
                epsilon: float):
        num_action_points = len(action_points)
        num_context_points = len(context_points)
        max_val = -np.infty
        for i in range(num_action_points):
            tiled_action = np.tile(action_points[i:i + 1], (num_context_points, 1))  # (num_context_points, d_x)
            action_contexts = np.concatenate([tiled_action, context_points], axis=-1)  # (num_context_points, d_x + d_c)
            upper_bounds, _ = get_upper_lower_bounds(model, action_contexts, self.beta(t))  # (num_context_points, )
            expected_upper = np.sum(ref_dist * upper_bounds)
            if expected_upper > max_val:
                max_val = expected_upper
                max_idx = i
        return action_points[max_idx:max_idx + 1]
