import numpy as np
import tensorflow as tf
import cvxpy as cp
from typing import Callable
import gpflow as gpf
from gpflow.models.model import GPModel
from gpflow.kernels import Kernel
from trieste.space import Box, DiscreteSearchSpace
from trieste.type import TensorType
from scipy.stats import norm, multivariate_normal

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


def cholesky_inverse(M, jitter=gpf.config.default_jitter()):
    """
    Computes the inverse of M using the Cholesky decomposition. M must be a positive definite matrix.
    :param M: shape (b, m, m) or (m, m)
    :param jitter:
    :return:
    """
    if len(M.shape) == 3:
        b, m, _ = M.shape
        inv_L = np.linalg.inv(np.linalg.cholesky(M + jitter * np.eye(m)[None, :, :]))
        P = np.transpose(inv_L, [0, 2, 1])
    elif len(M.shape) == 2:
        m, _ = M.shape
        inv_L = np.linalg.inv(np.linalg.cholesky(M + jitter * np.eye(m)))
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


def get_discrete_normal_dist(context_points, mean, cov):
    """
    Returns an array of shape |C| that is a probability distribution over the context set. Uses the normal distribution
    with the specified mean and variance.
    :param context_points: Array of shape (|C|, d)
    :param mean: array of shape (d, )
    :param cov: array of shape (d, d)
    :return: array of shape (|C|, )
    """
    rv = multivariate_normal(mean=mean, cov=cov, allow_singular=False)
    pdfs = rv.pdf(context_points)
    return np.squeeze((pdfs / np.sum(pdfs)))


def get_discrete_uniform_dist(context_points):
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
                            divergence: str,
                            cvx_opt_max_iters: int = None,
                            cvx_opt_verbose: bool = False,
                            cvx_solver: str = 'ECOS',
                            v: TensorType = None):
    """
    Calculates inf_Q E_{c~Q}[f(x, c)]
    :param f: Array of shape (|C|, ). We are adversarially minimizing the expectation of these values. Could be
    objective function or UCB scores
    :param M: Array of shape (|C|, |C|). Kernel matrix for MMD
    :param w_t: Array of shape (|C|, ). Reference distribution
    :param epsilon: margin. Radius of ball around which we can choose our adversarial distribution
    :param divergence: str, either 'MMD', 'TV' or 'modified_chi_squared'
    :param cvx_opt_max_iters:
    :param cvx_opt_verbose:
    :param cvx_solver:
    :param v: Array of shape (|C|**2, ). Cost vector using Euclidean norm. Only required for Wasserstein
    :return: value, w
    """
    num_context = len(f)

    if divergence != "wass":
        w = cp.Variable(num_context)

        if divergence == 'modified_chi_squared':
            f = np.around(f, 4)  # avoid numerical errors with ECOS convex solver

        objective = cp.Minimize(w @ f)
        if divergence == "MMD" or divergence == "MMD_approx":  # If we're calculating this, we want the true MMD
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

        try:
            if cvx_opt_max_iters is None:
                expectation = prob.solve(solver=cvx_solver, verbose=cvx_opt_verbose)
            else:
                expectation = prob.solve(solver=cvx_solver, max_iters=cvx_opt_max_iters, verbose=cvx_opt_verbose)
        except:
            print("ECOS failed, trying SCS")
            if cvx_opt_max_iters is None:
                expectation = prob.solve(solver='SCS', verbose=cvx_opt_verbose)
            else:
                expectation = prob.solve(solver='SCS', max_iters=cvx_opt_max_iters, verbose=cvx_opt_verbose)

        return expectation, w.value
    else:
        if v is None:
            raise Exception("v cannot be None for Wasserstein")
        p = 1
        eps_raised = epsilon ** p
        y = cp.Variable(num_context ** 2)

        M1 = np.zeros((num_context ** 2, num_context))
        for i in range(num_context):
            M1[i * num_context:(i + 1) * num_context, i] = np.ones(num_context)
        M2 = np.tile(np.eye(num_context), (num_context, 1))

        objective = cp.Minimize(y @ M2 @ f)

        constraints = [y @ v <= eps_raised,
                       y >= 0.,
                       M1.T @ y == w_t]

        prob = cp.Problem(objective, constraints)

        try:
            if cvx_opt_max_iters is None:
                expectation = prob.solve(solver=cvx_solver, verbose=cvx_opt_verbose)
            else:
                expectation = prob.solve(solver=cvx_solver, max_iters=cvx_opt_max_iters, verbose=cvx_opt_verbose)
        except:
            print("ECOS failed, trying SCS")
            if cvx_opt_max_iters is None:
                expectation = prob.solve(solver='SCS', verbose=cvx_opt_verbose)
            else:
                expectation = prob.solve(solver='SCS', max_iters=cvx_opt_max_iters, verbose=cvx_opt_verbose)

        return expectation, M2.T @ y.value


def get_robust_expectation_and_action(action_points: TensorType,
                                      context_points: TensorType,
                                      kernel: Kernel,
                                      fvals_source: str,
                                      ref_dist: TensorType,
                                      divergence: str,
                                      epsilon: float,
                                      obj_func: Callable = None,
                                      model: ModelOptModule = None,
                                      beta: float = None,
                                      cvx_opt_max_iters: int = 50000,
                                      cvx_opt_verbose: bool = False
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
    :param cvx_opt_max_iters:
    :param cvx_opt_verbose:
    :return: tuple (float, array of shape (1, d_x)). Value of robust action, and robust action
    """
    num_actions = len(action_points)
    num_context_points = len(context_points)
    expectations = []
    domain = cross_product(action_points, context_points)

    for i in range(num_actions):
        action_contexts = get_action_contexts(i, domain, num_context_points)
        if divergence == 'MMD' or divergence == 'MMD_approx':
            M = kernel(context_points)
        else:
            M = None
        if fvals_source == 'obj_func':
            f = np.squeeze(obj_func(action_contexts), axis=-1)
        elif fvals_source == 'ucb':
            f, _ = get_upper_lower_bounds(model, action_contexts, beta)
        else:
            raise Exception("Invalid fvals_source given")
        expectation, _ = adversarial_expectation(f=f,
                                                 M=M,
                                                 w_t=ref_dist,
                                                 epsilon=epsilon,
                                                 divergence=divergence,
                                                 cvx_opt_max_iters=cvx_opt_max_iters,
                                                 cvx_opt_verbose=cvx_opt_verbose)
        expectations.append(expectation)
    max_idx = np.argmax(expectations)
    return np.max(expectations), action_points[max_idx:max_idx + 1]


def get_robust_exp_action_with_cvxprob(action_points: TensorType,
                                       context_points: TensorType,
                                       fvals_source: str,
                                       cvx_prob: Callable,
                                       obj_func: Callable = None,
                                       model: ModelOptModule = None,
                                       beta: float = None
                                       ):
    """
    Calculates max_x inf_Q E_{c~Q}[f(x, c)]
    :param action_points: Array of shape (num_actions, d_x)
    :param context_points: Array of shape (|C|, d_c)
    :param fvals_source: Either 'obj_func' or 'ucb'
    :param cvx_prob: function wrapper created by create_cvx_problem
    :param obj_func: objective function
    :param model: ModelOptModule
    :param beta: Beta for UCB scores. Only used if fvals_source is 'ucb'
    :return: tuple (float, array of shape (1, d_x)). Value of robust action, and robust action
    """
    num_actions = len(action_points)
    num_context_points = len(context_points)
    expectations = []
    domain = cross_product(action_points, context_points)

    for i in range(num_actions):
        action_contexts = get_action_contexts(i, domain, num_context_points)
        if fvals_source == 'obj_func':
            f = np.squeeze(obj_func(action_contexts), axis=-1)
        elif fvals_source == 'ucb':
            f, _ = get_upper_lower_bounds(model, action_contexts, beta)
        else:
            raise Exception("Invalid fvals_source given")
        expectation, _ = cvx_prob(f)
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
    if divergence == 'MMD' or divergence == 'MMD_approx':
        return MMD(ref_dist, true_dist, mmd_kernel, context_points)
    elif divergence == 'TV':
        return TV(ref_dist, true_dist)
    elif divergence == 'modified_chi_squared':
        return modified_chi_squared(true_dist, ref_dist)
    elif divergence == 'wass':
        return wasserstein(ref_dist, true_dist, context_points, 1)
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


def variance_p(fvals,
               p):
    mean = p @ fvals
    return p @ (fvals - mean) ** 2


def wasserstein(w1: TensorType,
                w2: TensorType,
                context_points: TensorType,
                p: int):
    """
    Computes the p-th Wasserstein metric between discrete probability distributions w1 and w2 using the Euclidean norm.
    :param w1: array of shape (|C|, )
    :param w2: array of shape (|C|, )
    :param context_points: array of shape (|C|, n). Context points.
    :param p: int
    :return: float
    """
    num_context = len(w1)
    y = cp.Variable(num_context ** 2)  # joint probability distribution

    # Compute cost vector f with Euclidean distance
    v = wass_cost_vector(context_points, p)

    objective = cp.Minimize(y @ v)
    M1 = np.zeros((num_context ** 2, num_context))
    for i in range(num_context):
        M1[i * num_context:(i + 1) * num_context, i] = np.ones(num_context)
    M2 = np.tile(np.eye(num_context), (num_context, 1))

    constraints = [y >= 0.,
                   M1.T @ y == w1,
                   M2.T @ y == w2]

    prob = cp.Problem(objective, constraints)
    try:
        cost = prob.solve(solver='ECOS')
    except:
        print("ECOS failed, trying SCS")
        cost = prob.solve(solver='SCS')

    return cost ** (1/p)


def wass_cost_vector(context_points, p):
    """

    :param context_points: array of shape (|C|, n)
    :param p: int
    :return: array of shape (|C| ** 2, )
    """
    return np.ravel(np.linalg.norm(context_points[:, None, :] - context_points[None, :, :], axis=2)) ** p


def worst_case_sens(fvals,
                    p,
                    context_points,
                    kernel,
                    divergence,
                    v=None):
    num_context_points = len(context_points)
    if divergence == 'MMD':
        K_inv = cholesky_inverse(kernel(context_points), jitter=1e-03)
        f_T_K_inv = fvals[None, :] @ K_inv  # (1, num_context_points)
        worst_case_sensitivity = np.sqrt(f_T_K_inv @ fvals[:, None] -
                                         ((np.squeeze(f_T_K_inv @ np.ones((num_context_points, 1))) ** 2) /
                                          np.sum(K_inv)))
    elif divergence == 'MMD_approx':  # Approximate K in MMD worst-case sensitivity with identity matrix
        worst_case_sensitivity = np.sqrt(fvals @ fvals - (np.sum(fvals) ** 2) / num_context_points)
    elif divergence == 'TV':
        worst_case_sensitivity = 0.5 * (np.max(fvals) - np.min(fvals))
    elif divergence == 'modified_chi_squared':
        worst_case_sensitivity = np.sqrt(2 * variance_p(fvals, p))
    elif divergence == 'wass':
        if v is None:
            raise Exception("v cannot be None for Wasserstein")
        worst_case_sensitivity = np.max(np.ravel(fvals[:, None] - fvals[None, :]) / (v + 1e-08))
    else:
        raise Exception("Invalid divergence passed to worst_case_sens")
    return worst_case_sensitivity


def get_mid_approx_func(context_points,
                        fvals,
                        kernel,
                        ref_dist,
                        worst_case_sensitivity,
                        divergence):
    """
    Approximates the adversarial expectation V over epsilon using a piecewise linear function that is the convex
    function in the middle of V's lower and upper bounds, which can be cheaply computed.
    :param action: array of shape (1, d_x)
    :param context_points:
    :param obj_func:
    :param kernel:
    :param ref_dist:
    :param worst_case_sensitivity:
    :param divergence:
    :return:
    """
    min_i = np.argmin(fvals)
    worst_dist = np.zeros(len(context_points))
    worst_dist[min_i] = 1

    f_eps_max = np.min(fvals)
    f_prime_0 = -np.squeeze(worst_case_sensitivity)
    f_0 = ref_dist @ fvals
    eps_l = (f_eps_max - f_0) / f_prime_0

    if divergence == 'MMD' or divergence == 'MMD_approx':
        eps_max = np.squeeze(MMD(worst_dist, ref_dist, kernel, context_points))
    elif divergence == 'TV':
        eps_max = np.squeeze(TV(worst_dist, ref_dist))
    elif divergence == 'modified_chi_squared':
        sqrt_eps_max = np.sqrt(np.squeeze(modified_chi_squared(worst_dist, ref_dist)))  # Take the square root
        sqrt_eps_l = eps_l
    elif divergence == 'wass':
        costs = np.linalg.norm(context_points - context_points[min_i:min_i+1, :], axis=1)
        eps_max = ref_dist @ costs
    else:
        raise Exception("Invalid divergence passed to get_mid_approx_func")

    if divergence == 'MMD' or divergence == 'MMD_approx' or divergence == 'TV' or divergence == 'wass':
        def f(eps):
            if 0 <= eps <= eps_l:
                fval = f_0 + 0.5 * eps * (f_prime_0 + (f_eps_max - f_0) / eps_max)
                return fval
            elif eps_l < eps < eps_max:
                fval = 0.5 * (f_0 + eps * ((f_eps_max - f_0) / eps_max) + f_eps_max)
                return fval
            else:
                return f_eps_max

        return np.vectorize(f)
    elif divergence == 'modified_chi_squared':
        # Account for the fact that what we have is actually a function on square root epsilon
        def f(eps):
            sqrt_eps = np.sqrt(eps)
            if 0 <= sqrt_eps <= sqrt_eps_l:
                fval = f_0 + 0.5 * sqrt_eps * (f_prime_0 + (f_eps_max - f_0) / sqrt_eps_max)
                return fval
            elif sqrt_eps_l < sqrt_eps < sqrt_eps_max:
                fval = 0.5 * (f_0 + sqrt_eps * ((f_eps_max - f_0) / sqrt_eps_max) + f_eps_max)
                return fval
            else:
                return f_eps_max

        return np.vectorize(f)


def get_action_contexts(action, domain, num_context_points):
    """

    :param action: int in range [m-1], where m is num_actions
    :param domain: cross product of actions and contexts. array of shape (m * n, d_x + d_y).
    :return: array of shape (n, d_x + d_y)
    """
    return domain[action * num_context_points:(action + 1) * num_context_points, :]


def discretize_1d(cont_values, context_points):
    """
    :param cont_values: array of shape (n)
    :param context_points: array of shape (m, 1)
    :return: array of shape (n)
    """
    return np.squeeze(context_points[np.argmin(abs(context_points - cont_values), axis=0)])


def get_discrete_dist_1d(arr, context_points):
    """

    :param arr: array of shape (n). Discretized to context_points
    :param context_points: array of shape (m, 1)
    :return: array of shape (m). Discrete probability distribution
    """
    unique, counts = np.unique(arr, return_counts=True)
    dic = dict(zip(unique, counts))
    dist = np.ones(len(context_points))  # have non-zero probability for every possible value
    for i in range(len(context_points)):
        ctx = context_points[i][0]
        if ctx in dic.keys():
            dist[i] = dic[ctx]
    return dist / len(arr)


def normalize_dist(dist):
    return dist / np.sum(dist)


def get_ordering(arr):
    enum = list(enumerate(arr))
    enum.sort(key=lambda x: x[1])
    return np.array(enum)[:, 0].astype(np.int32)


def create_cvx_problem(num_context,
                       M,
                       w_t,
                       epsilon,
                       divergence,
                       v=None):
    w = cp.Variable(num_context)
    g = cp.Parameter(num_context)
    objective = cp.Minimize(w @ g)
    if divergence == "MMD" or divergence == "MMD_approx":  # If we're calculating this, we want the true MMD
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
    elif divergence == "wass":
        if v is None:
            raise Exception("v cannot be None for Wasserstein")
        p = 1
        eps_raised = epsilon ** p
        y = cp.Variable(num_context ** 2)

        M1 = np.zeros((num_context ** 2, num_context))
        for i in range(num_context):
            M1[i * num_context:(i + 1) * num_context, i] = np.ones(num_context)
        M2 = np.tile(np.eye(num_context), (num_context, 1))

        objective = cp.Minimize(y @ M2 @ g)

        constraints = [y @ v <= eps_raised,
                       y >= 0.,
                       M1.T @ y == w_t]

        prob = cp.Problem(objective, constraints)
    else:
        raise Exception("Incorrect divergence given")
    prob = cp.Problem(objective, constraints)

    def wrapper(f):
        g.value = f
        try:
            value = prob.solve(solver='ECOS', warm_start=True)
        except:
            print("ECOS failed, trying SCS")
            value = prob.solve(solver='SCS', warm_start=True)
        sol = w.value
        if divergence != "wass":
            return value, sol
        else:
            return value, M2.T @ y.value

    return wrapper


def cumu_robust_regrets(obj_func,
                        query_points,
                        action_points,
                        context_points,
                        kernel,
                        ref_dist_func,
                        margin_func,
                        divergence,
                        robust_expectation_action):
    action_dims = action_points.shape[-1]
    regrets = []
    cumulative_regrets = []
    if divergence == 'MMD' or divergence == 'MMD_approx':
        M = kernel(context_points)
    else:
        M = None
    for t in range(len(query_points)):
        domain = cross_product(query_points[t:t + 1, 0:action_dims], context_points)
        f = obj_func(domain)
        query_expectation, w = adversarial_expectation(f=f, M=M, w_t=ref_dist_func(t), epsilon=margin_func(t),
                                                       divergence=divergence)
        if robust_expectation_action is None:
            robust_expectation, robust_action = get_robust_expectation_and_action(action_points=action_points,
                                                                                  context_points=context_points,
                                                                                  kernel=kernel,
                                                                                  fvals_source='obj_func',
                                                                                  ref_dist=ref_dist_func(t),
                                                                                  divergence=divergence,
                                                                                  epsilon=margin_func(t),
                                                                                  obj_func=obj_func)
        else:
            robust_expectation, robust_action = robust_expectation_action  # Warning: assumes constant functions
        regrets.append(robust_expectation - query_expectation)
        cumulative_regrets.append(np.sum(regrets))
    return cumulative_regrets


def imm_robust_regret(obj_func,
                        query_points,
                        action_points,
                        context_points,
                        kernel,
                        ref_dist_func,
                        margin_func,
                        divergence,
                        robust_expectation_action):
    action_dims = action_points.shape[-1]
    regrets = []
    if divergence == 'MMD' or divergence == 'MMD_approx':
        M = kernel(context_points)
    else:
        M = None
    for t in range(len(query_points)-5, len(query_points)):
        domain = cross_product(query_points[t:t + 1, 0:action_dims], context_points)
        f = obj_func(domain)
        query_expectation, w = adversarial_expectation(f=f, M=M, w_t=ref_dist_func(t), epsilon=margin_func(t),
                                                       divergence=divergence)
        if robust_expectation_action is None:
            robust_expectation, robust_action = get_robust_expectation_and_action(action_points=action_points,
                                                                                  context_points=context_points,
                                                                                  kernel=kernel,
                                                                                  fvals_source='obj_func',
                                                                                  ref_dist=ref_dist_func(t),
                                                                                  divergence=divergence,
                                                                                  epsilon=margin_func(t),
                                                                                  obj_func=obj_func)
        else:
            robust_expectation, robust_action = robust_expectation_action  # Warning: assumes constant functions
        regrets.append(robust_expectation - query_expectation)
    return np.mean(regrets)
