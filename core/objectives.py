from collections.abc import Callable

import numpy as np
import pickle
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow as gpf
from trieste.space import Box


from core.utils import discretize_Box


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


def get_obj_func(name, lowers, uppers, kernel, context_dims, rand_func_num_points=100, seed=0):
    if name == 'rand_func':
        return sample_GP_prior(kernel, lowers, uppers, rand_func_num_points, seed)
    elif name == 'wind':
        return wind_cost
    elif name == 'portfolio':
        X = pickle.load(open("data/portfolio/normalized_samples.p", "rb"))
        y = pickle.load(open("data/portfolio/standardized_returns.p", "rb"))
        return get_obj_func_from_samples(kernel, X, y)
    elif name == 'covid':
        X, y = pickle.load(open("data/covid/covid_X_y.p", "rb"))
        func = get_obj_func_from_samples(kernel, X, y)
        if context_dims == 3:
            return func
        elif context_dims == 4:
            return lambda x: func(x[:, :3])

    elif name == 'plant':
        NH3pH_leaf_max_area_func, _, _ = create_synth_funcs(params='NH3pH')

        def NH3pH_wrapper(vals):
            X = np.zeros(vals.shape)
            X[:, 0] = vals[:, 1] * 30000
            X[:, 1] = 2.5 + vals[:, 0] * (6.5 - 2.5)

            mean, _ = NH3pH_leaf_max_area_func(X)
            leaf_mean = 67.2466342112483
            leaf_std = 59.347376136036964
            return (mean - leaf_mean) / leaf_std

        return NH3pH_wrapper
    else:
        raise Exception("Incorrect name passed to get_obj_func")


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


def get_obj_func_from_samples(kernel, X, y, jitter=1e-03):
    """
    Constructs an objective function from samples. Interpolates between the samples like a GP posterior would.
    :param kernel: a GPflow kernel
    :param X: array of shape (n, d)
    :param y: array of shape (n, 1)
    :param jitter:
    :return:
    """
    cov = kernel(X) + jitter * np.eye(len(X), dtype=np.float32)
    L_inv = np.linalg.inv(np.linalg.cholesky(cov))
    K_inv_f = L_inv.T @ L_inv @ y
    return lambda x: kernel(x, X) @ K_inv_f


def wind_cost(action_contexts):
    """
    :param action_contexts: array of shape (n, 2)
    :return: tensor of shape (n, 1)
    """
    n, _ = action_contexts.shape

    c_minus_x = action_contexts[:, 1:] - action_contexts[:, 0:1]
    max_c_minus_x = np.max(np.concatenate([c_minus_x, np.zeros((n, 1))], axis=1), axis=1)[:, None]

    min_x_c = np.min(action_contexts, axis=1)[:, None]

    x_minus_c = action_contexts[:, 0:1] - action_contexts[:, 1:]
    max_x_minus_c = np.max(np.concatenate([x_minus_c, np.zeros((n, 1))], axis=1), axis=1)[:, None]

    return 0.1 * max_c_minus_x + min_x_c - 5 * max_x_minus_c


def create_synth_funcs(params):
    """

    :param params:
    :return:
    """
    gp_leaf_dict = pickle.load(open(f"data/plant/{params}_gp_leaf_dict.p", "rb"))
    leaf_mean, leaf_std, tbm_mean, tbm_std, tbs_mean, tbs_std, num_inducing, d = pickle.load(
        open(f"data/plant/{params}_req_variables.p", "rb"))

    gp_leaf = init_heteroscedastic_gp(num_inducing=num_inducing, d=d)
    gpf.utilities.multiple_assign(gp_leaf, gp_leaf_dict)

    def leaf_max_area_func(X):
        """
        Returns the predictive mean and variance of the maximum leaf area.
        :param X: Array of shape (num_preds, d).
        :return: Tuple (array of shape (num_preds, 1), array of shape (num_preds, 1). First element is mean, second
        is variance.
        """
        mean, var = gp_leaf.predict_y(X)
        return mean.numpy() * leaf_std + leaf_mean, var.numpy() * (leaf_std**2)

    return leaf_max_area_func, None, None


def init_heteroscedastic_gp(num_inducing, d):
    """
    Initializes default heteroscedastic GP, so that we can load the parameters later.
    :param num_inducing:
    :param d:
    :return:
    """
    likelihood = gpf.likelihoods.HeteroskedasticTFPConditional(
        distribution_class=tfp.distributions.Normal,  # Gaussian Likelihood
        scale_transform=tfp.bijectors.Exp(),  # Exponential Transform
    )

    kernels = [
        gpf.kernels.SquaredExponential(lengthscales=np.ones(d)),
        gpf.kernels.SquaredExponential(lengthscales=np.ones(d))
    ]

    kernel = gpf.kernels.SeparateIndependent(kernels)

    Z = tf.zeros((num_inducing, d))
    inducing_variable = gpf.inducing_variables.SeparateIndependentInducingVariables(
        [
            gpf.inducing_variables.InducingPoints(Z),  # This is U1 = f1(Z1)
            gpf.inducing_variables.InducingPoints(Z),  # This is U2 = f2(Z2)
        ])

    model = gpf.models.SVGP(kernel=kernel,
                            likelihood=likelihood,
                            inducing_variable=inducing_variable,
                            num_latent_gps=likelihood.latent_dim)
    return model
