import numpy as np
import tensorflow as tf
from gpflow.models.model import GPModel
from trieste.space import Box, DiscreteSearchSpace
from trieste.type import TensorType


def construct_grid_1d(min_range,
                      max_range,
                      grid_density):
    return np.expand_dims(np.linspace(min_range, max_range, grid_density), axis=1)


def discretize_Box(search_space: Box,
                   grid_density_per_dim: int):
    dims = len(search_space.lower)
    lower = search_space.lower
    upper = search_space.upper
    discrete_space = DiscreteSearchSpace(
        tf.convert_to_tensor(construct_grid_1d(lower[0], upper[0], grid_density_per_dim)))
    for i in range(1, dims):
        discrete_space *= DiscreteSearchSpace(
            tf.convert_to_tensor(construct_grid_1d(lower[i], upper[i], grid_density_per_dim)))
    return discrete_space


def get_maximizer(gp: GPModel,
                  search_points: TensorType):
    """
    Gets the maximizer of the posterior mean of the current GP model. If search_space is continuous,
    discretize it with grid_density_per_dim first
    """
    f_preds = gp.predict_f(search_points)[0][:, 0:]  # Indexing at the back is in case of heteroscedastic GPs
    # where predict_f returns both the preds for f and g
    return search_points[np.argmax(f_preds)]


def get_upper_lower_bounds(model,
                           search_points,
                           beta,
                           ):
    f_means, f_vars = model.predict_f(search_points)
    beta_vars = beta * np.squeeze(np.sqrt(f_vars.numpy()))
    upper = np.squeeze(f_means.numpy()) + beta_vars
    lower = np.squeeze(f_means.numpy()) - beta_vars
    return upper, lower


def cholesky_inverse(M):
    """
    Computes the inverse of M using the Cholesky decomposition. M must be a positive definite matrix.
    :param M: shape (b, m, m)
    :return:
    """
    inv_L = np.linalg.inv(np.linalg.cholesky(M))
    P = np.transpose(inv_L, [0, 2, 1])
    return P @ inv_L
