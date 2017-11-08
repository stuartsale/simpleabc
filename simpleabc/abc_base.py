"""
Module for Approximate Bayesian Computation

"""
from abc import ABCMeta, abstractmethod
import multiprocessing as mp
import numpy as np
from scipy import stats
from numpy.lib.recfunctions import stack_arrays

from simpleabc.util import calc_weights, weighted_covar, effective_sample_size


class ABCProcess(mp.Process):
    '''

    '''

    def run(self):
        np.random.seed()
        if self._target:
            self._target(*self._args, **self._kwargs)

# ============================================================================
# ======================    ABC Algorithms   =================================
# ============================================================================


def basic_abc(model, data, epsilon=1, min_samples=10,
              parallel=False, n_procs='all', pmc_mode=False,
              weights='None', theta_prev='None', tau_squared='None'):
    """
    Perform Approximate Bayesian Computation (ABC) on a data set given a
    forward model.

    ABC is a likelihood-free method of Bayesian inference that uses simulation
    to approximate the true posterior distribution of a parameter. It is
    appropriate to use in situations where:

    The likelihood function is unknown or is too computationally
    expensive to compute.

    There exists a good forward model that can produce data sets
    like the one of interest.

    It is not a replacement for other methods when a likelihood
    function is available!

    Parameters
    ----------
    model : object
        A model that is a subclass of simpleabc.Model
    data  : object, array_like
        The "observed" data set for inference.
    epsilon : float, optional
        The tolerance to accept parameter draws, default is 1.
    min_samples : int, optional
        Minimum number of posterior samples.
    parallel : bool, optional
        Run in parallel mode. Default is a single thread.
    n_procs : int, str, optional
        Number of subprocesses in parallel mode. Default is 'all' one for each
        available core.
    pmc_mode : bool, optional
        Population Monte Carlo mode on or off. Default is False. This is not
        meant to be called by the user, but is set by simple_abc.pmc_abc.
    weights : object, array_like, str, optional
        Importance sampling weights from previous PMC step. Used  by
        simple_abc.pmc_abc only.
    theta_prev : object, array_like, str, optional
        Posterior draws from previous PMC step.  Used by simple_abc.pmc_abc
        only.
    tau_squared : object, array_like, str, optional
        Previous Gaussian kernel variances. for importance sampling. Used by
        simple_abc.pmc_abc only.

    Returns
    -------
    posterior : numpy array
        Array of posterior samples.
    distances : object
        Array of accepted distances.
    accepted_count : float
        Number of  posterior samples.
    trial_count : float
        Number of total samples attempted.
    epsilon : float
        Distance tolerance used.
    weights : numpy array
        Importance sampling weights. Returns an array of 1s where
        size = posterior.size when not in pmc mode.
    tau_squared : numpy array
        Gaussian kernel variances. Returns an array of 0s where
        size = posterior.size when not in pmc mode.
    eff_sample : numpy array
        Effective sample size. Returns an array of 1s where
        size = posterior.size when not in pmc mode.

    Examples
    --------
    Forth coming.

    """

    posterior, rejected, distances = [], [], []
    trial_count, accepted_count = 0, 0

    data_summary_stats = model.summary_stats(data)
    model.set_epsilon(epsilon)

    while accepted_count < min_samples:
        trial_count += 1

        if pmc_mode:
            theta_star = theta_prev[:, np.random.choice(
                                    xrange(0, theta_prev.shape[1]),
                                    replace=True, p=weights/weights.sum())]

            theta = stats.multivariate_normal.rvs(theta_star, tau_squared)
            if np.isscalar(theta) is True:
                theta = [theta]

        else:
            theta = model.draw_theta()

        synthetic_data = model.generate_data(theta)

        synthetic_summary_stats = model.summary_stats(synthetic_data)
        distance = model.distance_function(data_summary_stats,
                                           synthetic_summary_stats)

        if distance < epsilon:
            accepted_count += 1
            posterior.append(theta)
            distances.append(distance)

        else:
            pass
            # rejected.append(theta)

    posterior = np.asarray(posterior).T

    if len(posterior.shape) > 1:
        n = posterior.shape[1]
    else:
        n = posterior.shape[0]

    weights = np.ones(n)
    tau_squared = np.zeros((posterior.shape[0], posterior.shape[0]))
    eff_sample = n

    return (posterior, distances,
            accepted_count, trial_count,
            epsilon, weights, tau_squared, eff_sample)


def pmc_abc(model, data, epsilon_0=1, min_samples=10,
            steps=10, resume=None, parallel=False, n_procs='all',
            sample_only=False, target_epsilon=0.):
    """
    Perform a sequence of ABC posterior approximations using the sequential
    population Monte Carlo algorithm.


    Parameters
    ----------
    model : object
        A model that is a subclass of simpleabc.Model
    data  : object, array_like
        The "observed" data set for inference.
    epsilon_0 : float, optional
        The initial tolerance to accept parameter draws, default is 1.
    min_samples : int, optional
        Minimum number of posterior samples.
    steps : int
        The number of pmc steps to attempt
    resume : numpy record array, optional
        A record array of a previous pmc sequence to continue the sequence on.
    parallel : bool, optional
        Run in parallel mode. Default is a single thread.
    n_procs : int, str, optional
        Number of subprocesses in parallel mode. Default is 'all' one for each
        available core.
    target_epsilon : float
        If this value of epsilon is reached the sampling is stopped at that
        pmc step, even if the number of steps is less than steps.

    Returns
    -------

    output_record : numpy record array
        A record array containing all ABC output for each step indexed by step
        (0, 1, ..., n,). Each step sub arrays is made up of the following
        variables:
    posterior : numpy array
        Array of posterior samples.
    distances : object
        Array of accepted distances.
    accepted_count : float
        Number of  posterior samples.
    trial_count : float
        Number of total samples attempted.
    epsilon : float
        Distance tolerance used.
    weights : numpy array
        Importance sampling weights. Returns an array of 1s where
        size = posterior.size when not in pmc mode.
    tau_squared : numpy array
        Gaussian kernel variances. Returns an array of 0s where
        size = posterior.size when not in pmc mode.
    eff_sample : numpy array
        Effective sample size. Returns an array of 1s where
        size = posterior.size when not in pmc mode.

    Examples
    --------
    Forth coming.

    """

    output_record = np.empty(steps, dtype=[('theta accepted', object),
                                           # ('theta rejected', object),
                                           ('D accepted', object),
                                           ('n accepted', float),
                                           ('n total', float),
                                           ('epsilon', float),
                                           ('weights', object),
                                           ('tau_squared', object),
                                           ('eff sample size', object),
                                           ])

    if resume is not None:
        steps = xrange(resume.size, resume.size + steps)
        output_record = stack_arrays((resume, output_record), asrecarray=True,
                                     usemask=False)
        epsilon = stats.scoreatpercentile(resume[-1]['D accepted'],
                                          per=75)
        theta = resume['theta accepted'][-1]
        weights = resume['weights'][-1]
        tau_squared = resume['tau_squared'][-1]

    else:
        steps = xrange(steps)
        epsilon = epsilon_0

    for step in steps:
        print 'Starting step {0}, epsilon={1:.5f}'.format(step, epsilon)
        if step == 0:
            # First ABC calculation

            if parallel:
                if n_procs == 'all':
                    n_procs = mp.cpu_count()

                chunk = np.ceil(min_samples/float(n_procs))
                print "Running {} particles on {} processors".format(chunk,
                                                                     n_procs)

                output = mp.Queue()
                processes = [ABCProcess(target=parallel_basic_abc,
                                        args=(model, data, output),
                                        kwargs={'epsilon': epsilon,
                                                'min_samples': chunk,
                                                'pmc_mode': False})
                             for i in xrange(n_procs)]

                for p in processes:
                    p.start()
                for p in processes:
                    p.join()
                results = [output.get() for p in processes]

                output_record[step] = combine_parallel_output(results)

            else:

                output_record[step] = basic_abc(model, data, epsilon=epsilon,
                                                min_samples=min_samples,
                                                parallel=False, pmc_mode=False)

            theta = output_record[step]['theta accepted']
            # print theta.shape
            tau_squared = 2 * np.cov(theta)
            # print tau_squared
            weights = np.ones(theta.shape[1]) * 1.0/theta.shape[1]
            # print weights

            output_record[step]['weights'] = weights
            output_record[step]['tau_squared'] = tau_squared

        else:
            # print weights, tau_squared
            # print theta
            theta_prev = theta
            weights_prev = weights

            if parallel:
                if n_procs == 'all':
                    n_procs = mp.cpu_count()

                chunk = np.ceil(min_samples/float(n_procs))
                print "Running {} particles on {} processors".format(chunk,
                                                                     n_procs)

                output = mp.Queue()
                processes = [ABCProcess(target=parallel_basic_abc,
                                        args=(model, data, output),
                                        kwargs={'epsilon': epsilon,
                                                'min_samples': chunk,
                                                'pmc_mode': True,
                                                'weights': weights,
                                                'theta_prev': theta_prev,
                                                'tau_squared': tau_squared})
                             for i in xrange(n_procs)]

                for p in processes:
                    p.start()
                for p in processes:
                    p.join()
                results = [output.get() for p in processes]

                output_record[step] = combine_parallel_output(results)

            else:

                output_record[step] = basic_abc(model, data, epsilon=epsilon,
                                                min_samples=min_samples,
                                                parallel=False,
                                                n_procs=n_procs, pmc_mode=True,
                                                weights=weights,
                                                theta_prev=theta_prev,
                                                tau_squared=tau_squared)

            theta = output_record[step]['theta accepted']

            # print theta

            # print theta_prev
            effective_sample = effective_sample_size(weights_prev)

            if sample_only:
                weights = []
                tau_squared = []
            else:
                weights = calc_weights(theta_prev, theta, tau_squared,
                                       weights_prev, prior=model.prior)
                tau_squared = 2 * weighted_covar(theta, weights)

            output_record[step]['tau_squared'] = tau_squared

            output_record[step]['eff sample size'] = effective_sample

            output_record[step]['weights'] = weights

        # stop if target epsilon has been reached
        if epsilon < target_epsilon:
            output_record = output_record[: step+1]
            break

        epsilon = stats.scoreatpercentile(
                            output_record[step]['D accepted'], per=75)

    return output_record


def combine_parallel_output(x):
    """
    Combines multiple basic_abc output arrays into one (for parallel mode)
    :param x: array of basic_abc output arrays
    :return: Combined basic_abc output arrays
    """

    posterior = np.hstack([p[0] for p in x])
    distances = []
    for p in x:
        distances = distances + p[1]

    accepted_count = sum([p[2] for p in x])
    trial_count = sum([p[3] for p in x])
    epsilon = x[0][4]
    weights = np.hstack([p[5] for p in x])
    tau_squared = x[0][6]
    eff_sample = x[0][7]
    return (posterior, distances,
            accepted_count, trial_count,
            epsilon, weights, tau_squared, eff_sample)


def parallel_basic_abc(data, model, output, **kwds):
    """
    Wrapper for running basic_abc in parallel and putting the outputs
    in a queue
    :param data: same as basic_abc
    :param model: same as basic_abc
    :param output: multiprocess queue object
    :param kwds: sames basic_abc
    :return:
    """
    output.put(basic_abc(data, model, **kwds))
