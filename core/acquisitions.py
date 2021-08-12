from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from numpy.random import default_rng
from gpflow.kernels import Kernel
from trieste.type import TensorType

from core.models import ModelOptModule
from core.utils import get_upper_lower_bounds, get_robust_expectation_and_action, cholesky_inverse, worst_case_sens, \
    get_cubic_approx_func, cross_product, get_action_contexts


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
    :param mode:
    :return:
    """
    args = dict(sorted(locals().items()))
    if acq_name == 'GP-UCB':
        return GPUCBStochastic(**args)
    elif acq_name == 'DRBOGeneral':
        return DRBOGeneral(**args)
    elif acq_name == 'DRBOWorstCaseSens':
        return DRBOWorstCaseSens(**args)
    elif acq_name == 'DRBOCubicApprox':
        return DRBOCubicApprox(**args)
    elif acq_name == 'WorstCaseSensTS':
        return WorstCaseSensTS(**args)
    elif acq_name == 'CubicApproxTS':
        return CubicApproxTS(**args)
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
                epsilon: float):
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
        domain = cross_product(action_points, context_points)

        for i in range(num_action_points):
            action_contexts = get_action_contexts(i, domain, num_context_points)
            ucb_vals, _ = get_upper_lower_bounds(model, action_contexts, self.beta(t))  # (num_context_points, )
            expected_ucb = np.sum(ref_dist * ucb_vals)  # SAA

            worst_case_sensitivity = worst_case_sens(fvals=ucb_vals,
                                                     context_points=context_points,
                                                     kernel=kernel,
                                                     divergence=divergence)

            if divergence == 'MMD':
                sens_factor = epsilon  # might be square root epsilon for others
            elif divergence == 'TV':
                sens_factor = epsilon
            elif divergence == 'modified_chi_squared':
                sens_factor = np.sqrt(epsilon)
            else:
                raise Exception("Invalid divergence passed to DRBOWorstCaseSens")

            adv_lower_bound = expected_ucb - (sens_factor * worst_case_sensitivity)
            adv_lower_bounds.append(adv_lower_bound)
        max_idx = np.argmax(adv_lower_bounds)
        return action_points[max_idx:max_idx + 1]


class WorstCaseSensTS(Acquisition):
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
        domain = cross_product(action_points, context_points)

        fmean, fvar = model.predict_f(domain, full_cov=True)
        rng = default_rng()
        all_fvals = rng.multivariate_normal(np.squeeze(fmean), np.squeeze(fvar))  # (num_action_points * num_context_points)
        for i in range(num_action_points):
            fvals = all_fvals[i * num_context_points:(i+1) * num_context_points]
            expected_fvals = np.sum(ref_dist * fvals)  # SAA

            worst_case_sensitivity = worst_case_sens(fvals=fvals,
                                                     context_points=context_points,
                                                     kernel=kernel,
                                                     divergence=divergence)

            if divergence == 'MMD':
                sens_factor = epsilon  # might be square root epsilon for others
            elif divergence == 'TV':
                sens_factor = epsilon
            elif divergence == 'modified_chi_squared':
                sens_factor = np.sqrt(epsilon)
            else:
                raise Exception("Invalid divergence passed to WorstCaseSensTS")

            adv_lower_bound = expected_fvals - (sens_factor * worst_case_sensitivity)
            adv_lower_bounds.append(adv_lower_bound)
        max_idx = np.argmax(adv_lower_bounds)
        return action_points[max_idx:max_idx + 1]


class DRBOCubicApprox(Acquisition):
    """
    Uses a cubic approximation to the adversarial expectation function V to improve the worst case sensitivity
    approximation for large epsilon.
    """
    def __init__(self,
                 beta: Callable,
                 divergence: str,
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
        adv_approxs = []
        domain = cross_product(action_points, context_points)

        for i in range(num_action_points):
            action_contexts = get_action_contexts(i, domain, num_context_points)
            fvals, _ = get_upper_lower_bounds(model, action_contexts, self.beta(t))  # (num_context_points, )

            worst_case_sensitivity = worst_case_sens(fvals=fvals,
                                                     context_points=context_points,
                                                     kernel=kernel,
                                                     divergence=divergence)

            V_approx_func = get_cubic_approx_func(context_points=context_points,
                                                  fvals=fvals,
                                                  kernel=kernel,
                                                  ref_dist=ref_dist,
                                                  worst_case_sensitivity=worst_case_sensitivity,
                                                  divergence=divergence)

            adv_approxs.append(V_approx_func(epsilon))
        max_idx = np.argmax(adv_approxs)
        return action_points[max_idx:max_idx + 1]


class CubicApproxTS(Acquisition):
    """
    Uses a cubic approximation to the adversarial expectation function V to improve the worst case sensitivity
    approximation for large epsilon.
    """
    def __init__(self,
                 beta: Callable,
                 divergence: str,
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
        adv_approxs = []
        domain = cross_product(action_points, context_points)

        fmean, fvar = model.predict_f(domain, full_cov=True)
        rng = default_rng()
        all_fvals = rng.multivariate_normal(np.squeeze(fmean), np.squeeze(fvar))  # (num_action_points * num_context_points)

        for i in range(num_action_points):
            fvals = all_fvals[i * num_context_points:(i+1) * num_context_points]

            worst_case_sensitivity = worst_case_sens(fvals=fvals,
                                                     context_points=context_points,
                                                     kernel=kernel,
                                                     divergence=divergence)

            V_approx_func = get_cubic_approx_func(context_points=context_points,
                                                  fvals=fvals,
                                                  kernel=kernel,
                                                  ref_dist=ref_dist,
                                                  worst_case_sensitivity=worst_case_sensitivity,
                                                  divergence=divergence)

            adv_approxs.append(V_approx_func(epsilon))
        max_idx = np.argmax(adv_approxs)
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
        domain = cross_product(action_points, context_points)

        for i in range(num_action_points):
            action_contexts = get_action_contexts(i, domain, num_context_points)
            upper_bounds, _ = get_upper_lower_bounds(model, action_contexts, self.beta(t))  # (num_context_points, )
            expected_upper = np.sum(ref_dist * upper_bounds)
            if expected_upper > max_val:
                max_val = expected_upper
                max_idx = i
        return action_points[max_idx:max_idx + 1]
