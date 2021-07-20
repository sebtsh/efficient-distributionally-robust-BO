from collections.abc import Callable

import numpy as np
import tensorflow as tf
from trieste.space import Box
from trieste.type import TensorType
from trieste.utils.objectives import branin

from core.utils import discretize_Box


def neg_forrester(x: TensorType) -> TensorType:
    """
    Negative of the 1-dimensional test function by Forrester et. al. (2008). Defined as f(x) = -(6x-2)^2 * sin(12x-4).
    Typically evaluated over [0, 1]. We take the negative so that we have a maximization problem
    :param x: tensor of shape (..., 1), x1 in [0, 1]
    :return: tensor of shape (..., 1)
    """
    tf.debugging.assert_shapes([(x, (..., 1))])
    return -(6 * x - 2) * (6 * x - 2) * tf.sin(12 * x - 4)


NEG_FORRESTER_MAXIMIZER = 0.7572488
NEG_FORRESTER_MAXIMUM = 6.020740


def neg_branin(x: TensorType) -> TensorType:
    return -branin(x)


def standardize_objective(obj_func: Callable,
                          lower: list,
                          upper: list,
                          grid_density_per_dim: int = 100) -> Callable:
    """
    Estimates the mean and standard deviation of an objective function using a discrete space, then returns a
    standardized version of the objective function. May help with GP optimization
    :param obj_func: function that takes a tensor of shape (..., d) and returns a tensor of shape (..., 1)
    :param lower: lower bounds of input domain, list of length d
    :param upper: upper bounds of input domain, list of length d
    :param grid_density_per_dim: int
    :return: standardized version of obj_func
    """
    search_space = discretize_Box(Box(lower, upper), grid_density_per_dim)
    mean = np.mean(obj_func(search_space.points))
    std = np.std(obj_func(search_space.points))
    return lambda x: (obj_func(x) - mean) / std


def get_obj_func(name, lowers, uppers, kernel, rand_func_num_points=100, seed=0):
    if name == 'neg_forrester':
        return neg_forrester
    elif name == 'neg_branin':
        return neg_branin
    elif name == 'rand_func':
        return sample_GP_prior(kernel, lowers, uppers, rand_func_num_points, seed)


def sample_GP_prior(kernel, lowers, uppers, num_points, seed=0, jitter=1e-03):
    """
    Sample a random function from a GP prior with mean 0 and covariance specified by a kernel.
    :param kernel: a GPflow kernel
    :param lowers:
    :param uppers:
    :param num_points:
    :param seed:
    :param jitter:
    :return:
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)

    points = Box(lowers, uppers).sample(num_points).numpy().astype(np.float32)
    cov = kernel(points) + jitter * np.eye(len(points), dtype=np.float32)
    f_vals = np.random.multivariate_normal(np.zeros(num_points), cov)[:, None]
    L_inv = np.linalg.inv(np.linalg.cholesky(cov))
    K_inv_f = L_inv.T @ L_inv @ f_vals
    return lambda x: kernel(x, points) @ K_inv_f
