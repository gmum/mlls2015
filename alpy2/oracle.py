from abc import ABCMeta, abstractmethod
import numpy as np

class BudgetExceededException(Exception):
    pass

class Oracle(object):
    """
    Oracle is a class managing also budget.

    It returns new labels, as well as
    raise error indicating no interesting samples or finished budget.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, X, y):
        """
        :param X: Set of examples to label
        :param y: Masked array
        :return: y': Array with labels
        """
        pass

    @property
    def cost_so_far(self):
        pass

class SimulatedOracle(Oracle):
    def __init__(self, sample_budget=np.inf):
        self._sample_budget = sample_budget
        self._cost_so_far = 0

    @property
    def cost_so_far(self):
        return self._cost_so_far

    def __call__(self, X, y):
        self._cost_so_far += 1
        if self._cost_so_far > self._sample_budget:
            raise BudgetExceededException()
        return y.data