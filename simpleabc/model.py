"""
Module that contains base class for building models for Approximate
Bayesian Computation

"""
from abc import ABCMeta, abstractmethod
from scipy import stats


class Model(object):
    """
    Base class for constructing models for approximate bayesian computing
    and various uses limited only by the user's imagination.

    WARNING!! Not meant for direct use! You must implement your own model as a
    subclass and override the all following methods:

    * Model.draw_theta
    * Model.generate_data
    * Model.summary_stats
    * Model.distance_function

    """

    __metaclass__ = ABCMeta

    def __call__(self, theta):
        return self.generate_data_and_reduce(theta)

    def set_data(self, data):
        self.data = data
        self.data_sum_stats = self.summary_stats(self.data)

    # TODO think about a beter way to handle prior functions
    def set_prior(self, prior):
        self.prior = prior

    def set_epsilon(self, epsilon):
        """
        A method to give the model object the value of epsilon if your model
        code needs to know it.
        """
        self.epsilon = epsilon

    def generate_data_and_reduce(self, theta):
        """
        A combined method for generating data, calculating summary statistics
        and evaluating the distance function all at once.
        """
        synth = self.generate_data(theta)
        sum_stats = self.summary_stats(synth)
        d = self.distance_function(sum_stats, self.data_sum_stats)

        return d

    # These methods handle the serialization of the frozen scipy.stats
    # distributions used for the priors when running in parallel mode
    def __getstate__(self):
        '''
        Copies the state dict of the model and pulls the keyword arguements
        out of self.prior to send to self.__setstate__
        '''
        result = self.__dict__.copy()
        result['prior'] = [p.kwds for p in self.prior]
        return result

    def __setstate__(self, state):
        '''
        Reconstructs the frozen prior distributions with the keywords
        used to make the oriniginal distribution objects acquired with
        self.__getstate__
        '''
        self.__dict__ = state
        # Replace this line with calls to create the frozen distributions you
        # are using for your prior, example:
        #   new_prior = [stats.norm(**state['prior'][0]),
        #               stats.uniform(**state['prior'][1])]
        new_prior = []
        self.__dict__['prior'] = new_prior

    @abstractmethod
    def draw_theta(self):
        """
        Sub-classable method for drawing from a prior distribution.

        This method should return an array-like iterable that is a vector of
        proposed model parameters from your prior distribution.
        """

    @abstractmethod
    def generate_data(self, theta):
        """
        Sub-classable method for generating synthetic data sets from forward
        model.

        This method should return an array/matrix/table of simulated data
        taking vector theta as an argument.
        """

    @abstractmethod
    def summary_stats(self, data):
        """
        Sub-classable method for computing summary statistics.

        This method should return an array-like iterable of summary statistics
        taking an array/matrix/table as an argument.
        """

    @abstractmethod
    def distance_function(self, summary_stats, summary_stats_synth):
        """
        Sub-classable method for computing a distance function.

        This method should return a distance D of for comparing to the
        acceptance tolerance (epsilon) taking two array-like iterables of
        summary statistics as an argument (nominally the observed summary
        statistics and .
        """
