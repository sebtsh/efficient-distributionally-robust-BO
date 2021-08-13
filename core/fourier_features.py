import numpy as np
from numpy.random import default_rng
from core.utils import cholesky_inverse


def sample_W_b_SqExp(lengthscales, m=100):
    """
    Samples the W matrix and b vector required to approximate sample paths from a GP posterior using random Fourier
    features, as described by (Hernandez-Lobato et. al., 2014). This method samples from the probability distribution
    that is the Fourier dual of the squared exponential kernel with ARD.
    :param lengthscales: array of shape (d).
    :param m: Number of Fourier features to use.
    :return: matrix of shape (m, d) and matrix of shape (m, 1).
    """
    d = len(lengthscales)
    rng = default_rng()
    W = rng.normal(loc=0.0,
                   scale=1.0 / lengthscales,
                   size=(m, d))
    b = rng.uniform(low=0,
                    high=2 * np.pi,
                    size=(m, 1))
    return W, b


def fourier_features(X, W, b):
    """
    Given sampled tensors W and b, construct Fourier features of X, i.e. Phi matrix.
    :param X: matrix of shape (n, d).
    :param W: matrix of shape (m, d).
    :param b: matrix of shape (m, 1).
    :return: matrix of shape (n, m).
    """
    m = W.shape[0]
    WX_b = W @ X.T + b  # (m, n)
    alpha = 1
    return np.sqrt(2.0 * alpha / m) * np.cos(WX_b.T)  # (n, m)


def get_theta_mean_cov(Phi, y, sigma):
    """
    :param Phi: matrix of shape (n, m). Matrix of n observations with m random Fourier features.
    :param y: Output values of shape (n).
    :param sigma: float. Observational variance.
    :return: vector of shape (m) and covariance matrix of shape (m, m).
    """
    n, m = Phi.shape
    A = Phi.T @ Phi + (sigma ** 2) * np.eye(m)
    A_inv = cholesky_inverse(A)
    mean = A_inv @ Phi.T @ y
    cov = (sigma ** 2) * A_inv
    return mean, cov


def sample_theta(W, b, X, y, sigma):
    """
    Samples theta from its posterior distribution. Handles non-degenerate case (m < n, number of Fourier features
    less than number of observations) and degenerate case (m > n, number of Fourier features more than
    number of observations) using method from (Seeger, 2008).
    :param W: matrix of shape (m, d). m is the number of random Fourier features, d is the dimensionality of the
    dataset.
    :param b: vector of shape (m, 1).
    :param X: Input points of shape (n, d).
    :param y: Output values of shape (n).
    :param sigma: float. Observational variance.
    :return: vector of shape (m).
    """
    m, d = W.shape
    n = X.shape[0]
    rng = default_rng()

    Phi = fourier_features(X, W, b)
    mean, cov = get_theta_mean_cov(Phi, y, sigma)

    if m <= n:  # non-degenerate case
        theta_sample = rng.multivariate_normal(mean, cov, method='cholesky')
    else:  # degenerate case
        Pi_inv = ((1 / sigma) * np.eye(m))
        U, d, _ = np.linalg.svd(Phi @ Pi_inv @ Phi.T)
        R = np.diag(1 / (np.sqrt(d) * (np.sqrt(d) + 1)))
        c_factor = (np.eye(m) - Pi_inv @ Phi.T @ U @ R @ U.T @ Phi) @ np.sqrt(Pi_inv)
        n = rng.standard_normal(m)
        c = c_factor @ n
        theta_sample = sigma * c + mean

    return theta_sample
