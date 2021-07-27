import numpy as np
import tensorflow as tf
import cvxpy as cp
from typing import Callable
import gpflow as gpf
from gpflow.models.model import GPModel
from gpflow.kernels import Kernel
from trieste.space import Box, DiscreteSearchSpace
from trieste.type import TensorType
from scipy.stats import norm

from core.models import ModelOptModule


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
    :param M: shape (b, m, m) or (m, m)
    :return:
    """
    if len(M.shape) == 3:
        b, m, _ = M.shape
        inv_L = np.linalg.inv(np.linalg.cholesky(M + gpf.config.default_jitter() * np.eye(m)[None, :, :]))
        P = np.transpose(inv_L, [0, 2, 1])
    elif len(M.shape) == 2:
        m, _ = M.shape
        inv_L = np.linalg.inv(np.linalg.cholesky(M + gpf.config.default_jitter() * np.eye(m)))
        P = np.transpose(inv_L)
    else:
        raise Exception("Wrong shape passed to cholesky_inverse")
    return P @ inv_L


def get_discrete_normal_dist_1d(context_points, mean, var):
    """
    Returns an array of shape |C| that is a probability distribution over the context set. Uses the normal distribution
    with the specified mean and variance.
    :param context_points: Array of shape (|C|, 1)
    :param mean:
    :param var:
    :return: array of shape (|C|, )
    """
    normal_rv = norm(loc=mean, scale=np.sqrt(var))
    pdfs = normal_rv.pdf(context_points)
    return np.squeeze((pdfs / np.sum(pdfs)))


def get_discrete_uniform_dist_1d(context_points):
    """
    Returns an array of shape |C| that is a uniform probability distribution over the context set.
    :param context_points: Array of shape (|C|, 1)
    :return: array of shape (|C|, )
    """
    return np.ones(len(context_points)) * (1 / len(context_points))


def cross_product(x, y):
    """

    :param x: array of shape (m, d_x)
    :param y: array of shape (n, d_y)
    :return:  array of shape (m * n, d_x + d_y)
    """
    m, d_x = x.shape
    n, d_y = y.shape
    x_temp = np.tile(x[:, :, None], (1, n, 1))
    x_temp = np.reshape(x_temp, [m * n, d_x])
    y_temp = np.tile(y, (m, 1))
    return np.concatenate([x_temp, y_temp], axis=-1)


def adversarial_expectation(f: TensorType,
                            M: TensorType,
                            w_t: TensorType,
                            epsilon: float,
                            divergence: str):
    """
    Calculates inf_Q E_{c~Q}[f(x, c)]
    :param f: Array of shape (|C|, ). We are adversarially minimizing the expectation of these values. Could be
    objective function or UCB scores
    :param M: Array of shape (|C|, |C|). Kernel matrix for MMD
    :param w_t: Array of shape (|C|, ). Reference distribution
    :param epsilon: margin. Radius of ball around which we can choose our adversarial distribution
    :param divergence: str, either 'MMD', 'TV' or 'modified_chi_squared'
    :return: value, w
    """
    num_context = len(f)
    w = cp.Variable(num_context)

    objective = cp.Minimize(w @ f)
    if divergence == "MMD":
        constraints = [cp.sum(w) == 1.0,
                       w >= 0.,
                       cp.quad_form(w - w_t, M) <= epsilon ** 2]
    elif divergence == "TV":
        constraints = [cp.sum(w) == 1.0,
                       w >= 0.,
                       cp.norm(w - w_t, 1) <= epsilon]
    elif divergence == "modified_chi_squared":
        phi = lambda x: 0.5 * ((x - 1) ** 2)
        constraints = [cp.sum(w) == 1.0,
                       w >= 0.,
                       w_t @ phi(w / w_t) <= epsilon]
    else:
        raise Exception("Incorrect divergence given")

    prob = cp.Problem(objective, constraints)
    expectation = prob.solve()
    return expectation, w.value


def get_robust_expectation_and_action(action_points: TensorType,
                                      context_points: TensorType,
                                      kernel: Kernel,
                                      fvals_source: str,
                                      ref_dist: TensorType,
                                      divergence: str,
                                      epsilon: float,
                                      obj_func: Callable = None,
                                      model: ModelOptModule = None,
                                      beta: float = None
                                      ):
    """
    Calculates max_x inf_Q E_{c~Q}[f(x, c)]
    :param action_points: Array of shape (num_actions, d_x)
    :param context_points: Array of shape (|C|, d_c)
    :param kernel: gpflow kernel
    :param fvals_source: Either 'obj_func' or 'ucb'
    :param ref_dist: Array of shape (|C|)
    :param divergence: str, 'MMD', 'TV' or 'modified_chi_squared''
    :param epsilon: margin. Radius of ball around which we can choose our adversarial distribution
    :param obj_func: objective function
    :param model: ModelOptModule
    :param beta: Beta for UCB scores. Only used if fvals_source is 'ucb'
    :return: tuple (float, array of shape (1, d_x)). Value of robust action, and robust action
    """
    num_actions = len(action_points)
    expectations = []
    for i in range(num_actions):
        domain = cross_product(action_points[i:i + 1], context_points)
        if divergence == 'MMD':
            M = kernel(domain)
        else:
            M = None
        if fvals_source == 'obj_func':
            f = np.squeeze(obj_func(domain), axis=-1)
        elif fvals_source == 'ucb':
            f, _ = get_upper_lower_bounds(model, domain, beta)
        else:
            raise Exception("Invalid fvals_source given")
        expectation, _ = adversarial_expectation(f=f,
                                                 M=M,
                                                 w_t=ref_dist,
                                                 epsilon=epsilon,
                                                 divergence=divergence)
        expectations.append(expectation)
    max_idx = np.argmax(expectations)
    return np.max(expectations), action_points[max_idx:max_idx + 1]


def get_margin(ref_dist: TensorType,
               true_dist: TensorType,
               mmd_kernel: Kernel,
               context_points: TensorType,
               divergence: str):
    """
    Computes the divergence between the reference distribution and the true distribution.
    :param ref_dist:
    :param true_dist:
    :param mmd_kernel:
    :param context_points:
    :param divergence: str, 'MMD', 'TV' or 'modified_chi_squared''
    :return:
    """
    if divergence == 'MMD':
        return MMD(ref_dist, true_dist, mmd_kernel, context_points)
    elif divergence == 'TV':
        return TV(ref_dist, true_dist)
    elif divergence == 'modified_chi_squared':
        return modified_chi_squared(ref_dist, true_dist)
    else:
        raise Exception("Wrong divergence passed to get_margin")


def MMD(w1: TensorType,
        w2: TensorType,
        kernel: Kernel,
        context_points: TensorType):
    """
    Calculates the MMD between distributions on a finite context set.
    :param w1: array of shape (|C|, )
    :param w2: array of shape (|C|, )
    :param kernel: gpflow kernel
    :param context_points: array of shape (|C|, d_c)
    :return: float
    """
    M = kernel(context_points)
    return np.sqrt((w1 - w2)[None, :] @ M @ (w1 - w2)[:, None])


def TV(w1: TensorType,
       w2: TensorType):
    """
    Calculates the total variation distance between 2 discrete distributions.
    :param w1: array of shape (|C|, )
    :param w2: array of shape (|C|, )
    :return: float
    """
    return np.linalg.norm(w1 - w2, ord=1)


def modified_chi_squared(w1: TensorType,
                         w2: TensorType):
    """
    Calculates the modified chi-squared phi-divergence between 2 discrete distributions. Calculates the divergence
    of w1 from w2, i.e. takes expectation of (w1/w2) under w2's distribution
    :param w1: array of shape (|C|, )
    :param w2: array of shape (|C|, )
    :return: float
    """
    phi = lambda x: 0.5 * ((x - 1) ** 2)
    return w2 @ phi(w1 / w2)
