"""
Utilities for Approximate Bayesian Computation

"""
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy import stats
from numpy.testing import assert_almost_equal


def calc_weights(theta_prev, theta, tau_squared, weights, prior="None"):
    """
    Calculates importance weights
    """
    weights_new = np.zeros_like(weights)

    if len(theta.shape) == 1:
        norm = np.zeros_like(theta)
        for i, T in enumerate(theta):
            for j in xrange(theta_prev[0].size):
                # print T, theta_prev[0][j], tau_squared
                # print type(T), type(theta_prev), type(tau_squared)
                norm[j] = stats.norm.pdf(T, loc=theta_prev[0][j],
                                         scale=tau_squared)
            weights_new[i] = prior[0].pdf(T)/sum(weights * norm)

        return weights_new/weights_new.sum()

    else:
        norm = np.zeros(theta_prev.shape[1])
        for i in xrange(theta.shape[1]):
            p = prior.pdf(theta[:, i])

            for j in xrange(theta_prev.shape[1]):
                norm[j] = stats.multivariate_normal.pdf(theta[:, i],
                                                        mean=theta_prev[:, j],
                                                        cov=tau_squared)

            weights_new[i] = p/sum(weights * norm)

        return weights_new/weights_new.sum()


def weighted_covar(x, w):
    """
    Calculates weighted covariance matrix
    :param x: 1 or 2 dimensional array-like, values
    :param w: 1 dimensional array-like, weights
    :return C: Weighted covariance of x or weighted variance if x is 1d
    """
    sumw = w.sum()
    assert_almost_equal(sumw, 1.0)
    if len(x.shape) == 1:
        assert x.shape[0] == w.size
    else:
        assert x.shape[1] == w.size
    sum2 = np.sum(w**2)

    if len(x.shape) == 1:
        xbar = (w*x).sum()
        var = sum(w * (x - xbar)**2)
        return var * sumw/(sumw*sumw-sum2)
    else:
        xbar = [(w*x[i]).sum() for i in xrange(x.shape[0])]
        covar = np.zeros((x.shape[0], x.shape[0]))
        for k in xrange(x.shape[0]):
            for j in xrange(x.shape[0]):
                for i in xrange(x.shape[1]):
                    covar[j, k] += (x[j, i]-xbar[j])*(x[k, i]-xbar[k]) * w[i]

        return covar * sumw/(sumw*sumw-sum2)


def effective_sample_size(w):
    """
    Calculates effective sample size
    :param w: array-like importance sampleing weights
    :return: float, effective sample size
    """

    sumw = sum(w)
    sum2 = sum(w**2)
    return sumw*sumw/sum2
