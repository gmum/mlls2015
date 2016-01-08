"""
Strategies for picking samples from datasets.
"""
import numpy as np
from sklearn.utils import validation as val
from abc import ABCMeta, abstractmethod
from alpy.utils import _check_masked_labels, unmasked_indices, masked_indices


class BaseStrategy(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, X, y, model, batch_size):
        raise NotImplemented("Virtual method got called")

class UncertaintySampling(BaseStrategy):

    def __init__(self):
        super(UncertaintySampling, self).__init__()


    def __call__(self, X, y, model, batch_size, **kwargs):
        """
        Parameters
        ----------
        X: numpy.ndarray
            Numpy array with shape (n, m), where `n` is the number of samples
            and `m` the number of features.

        y: numpy.ma.masked_array
            Masked numpy array with shape (n, )
            The mask is masking with True the unknown labels in `y`.

        current_model: sklearn estimator

        batch_size: int
            How many examples to query in one active learning loop iteration.

        Returns
        -------
        indices: numpy.ndarray

        """
        # checks
        _check_masked_labels(y)

        unknown_ids = masked_indices(y)
        known_ids = unmasked_indices(y)

        X = val.as_float_array(X)

        val.check_consistent_length(X, y)

        # mask samples with obscure labels
        pairwise = getattr(model, "_pairwise", False) or \
                   getattr(getattr(model, "estimator", {}), "_pairwise", False)
        X = X[unknown_ids, :][:, known_ids] if pairwise else X[unknown_ids]


        if hasattr(model, "decision_function"):
            # Settles page 12
            fitness = np.abs(np.ravel(model.decision_function(X)))
            ids = np.argsort(fitness)[:batch_size]

        elif hasattr(model, "predict_proba"):
            p = model.predict_proba(X)
            # Settles page 13
            fitness = np.sum(p * np.log(p), axis=1).ravel()
            ids = np.argsort(fitness)[:batch_size]

        else:
            raise AttributeError("Model with either `decision_function` or `predict_proba` method")

        return unknown_ids[ids]


class PassiveStrategy(BaseStrategy):

    def __init__(self):
        super(PassiveStrategy, self).__init__()

    def __call__(self, X, batch_size, rng, **kwargs):
        """
        Parameters
        ----------
        X: numpy.ndarray
            Numpy array with shape (n, m), where `n` is the number of samples
            and `m` the number of features.

        batch_size: int
            How many examples to query in one active learning loop iteration.

        rng:
            seed for random number generator
        Returns
        -------
        indices: numpy.ndarray

        """

        X = val.as_float_array(X)
        rng = val.check_random_state(rng)

        return rng.choice(X.shape[0], size=batch_size, replace=False)