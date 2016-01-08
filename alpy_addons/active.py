"""
The active learning meta class
"""
import logging
from collections import defaultdict
import time
import numpy as np
from sklearn.utils import validation as val
from sklearn.utils import check_random_state
from sklearn.base import clone, BaseEstimator

from alpy.utils import _check_masked_labels, unmasked_indices, masked_indices
from alpy.oracle import BudgetExceededException


class ActiveLearner(object):
    """
    Object for Active Learning.
    This class is intended to do as little as possible.
    If user wants more complicated behavior it is recommended to look into one of
    experiment classes.

    Parameters
    ----------
    strategy: function
        strategy method for querying examples to label by a oracle, as function or callable class.

    batch_size:
        how many examples to query in one active learning loop iteration.

    oracle: function
        the oracle object to determine the label of unknown samples.
        object of this type is instantiated at the beginning of fit

    param_grid: dict of iterables
        The parameter grid to search for
        #NOTE(kudkudak): we could, this would involve passing different base_search

    estimator:
        sklearn estimator that is partial_fitted (if accessible) every iteration
        NOTE: estimator can be also GridSearchCV

    random_state:
        seed for random number generator

    verbose: int
        verbosity level. 0: ERROR, 1: INFO, 2: DEBUG
    """

    def __init__(self,
                 strategy,
                 batch_size=1,
                 oracle=None,
                 estimator=None,
                 random_state=None,
                 verbose=0):

        # input arguments check
        if batch_size < 1:
            raise ValueError('`batch_size` is expected to be a positive integer, got {}'.format(batch_size))

        if not callable(strategy):
            raise TypeError('`strategy` is expected to be a callable, got {}.'.format(strategy))

        if not hasattr(estimator, 'fit'):
            raise TypeError('`estimator` is expected to have `fit` method')

        # set the verbose
        self._log = self._setup_log(verbose)

        # set up the private members of self
        self._strategy = strategy
        self._batch_size = batch_size
        self._oracle = oracle
        self._estimator = estimator
        self._rng = random_state

        # declare the results members
        self.model = None

        # add default monitors

    @staticmethod
    def _setup_log(verbose):
        if verbose == 0:
            lvl = logging.ERROR
        elif verbose == 1:
            lvl = logging.INFO
        elif verbose == 2:
            lvl = logging.DEBUG
        else:
            raise ValueError('`verbose` value is expected to be in (0, 1, 2), got {}.'.format(verbose))

        log = logging.getLogger(__name__)
        log.setLevel(lvl)
        return log

    def fit(self, X, y, monitors=None):
        """
        Fit ActiveLearner object

        Parameters
        ----------
        X: numpy.ndarray
            A numpy array with shape (n, m), where `n` is the number of samples
            and `m` the number of features.

        y: numpy.masked_array
            A masked numpy array with shape (n, ).
            It contains the class labels of the samples in `X`.
            Its mask is `True` where the class label is unknown.
        """

        # check monitors
        if not monitors:
            monitors = []
        elif not isinstance(monitors, list):
            monitors = [monitors]

        self._monitor_outputs = defaultdict(list)

        # check if `y` is okay and save a copy
        labels = _check_masked_labels(y).copy()

        # check if not all samples are masked, instead raise exception for now.
        if np.all(labels.mask):
            raise ValueError("Expected at least one value in `y.mask` to be `False`.")

        # check `X` and consistency with `y`
        X = val.as_float_array(X)
        val.check_consistent_length(X, labels.data)

        # clone and prefit the model
        self.model = clone(self._estimator)
        self._rng = check_random_state(self._rng)
        partial_fittable = hasattr(self.model, "partial_fit")

        known_idx = unmasked_indices(labels)
        X_known = X[known_idx]
        labels_known = labels[known_idx]

        self.model.fit(X_known, labels_known)

        # while it has not unmasked all the elements or budget is exceeded
        n_iter = 0;
        while len(known_idx) < X.shape[0]:
            self._monitor_outputs['iter_time'].append(-time.time())

            self._monitor_outputs['strategy_time'].append(-time.time())
            selected_idx = self._strategy(X=X, y=labels, model=self.model, batch_size=self._batch_size, rng=self._rng)

            self._monitor_outputs['strategy_time'][-1] += time.time()

            try:
                self._monitor_outputs['oracle_time'].append(-time.time())
                labels[selected_idx] = self._oracle(X[selected_idx], labels[selected_idx])
                self._monitor_outputs['oracle_time'][-1] += time.time()
            except BudgetExceededException as ex:
                self._log.info("Exceeded oracle budget at iteration {0}. Stopping active learning loop".format(i))
                break

            # get the known data
            # TODO: This might be very slow, every iteration we are copying data
            # TODO: what we could do is to reserve another block of memory and rewrite
            # or do swaps in current block of memory
            known_idx = unmasked_indices(labels)
            X_known = X[known_idx]
            labels_known = labels[known_idx]

            # Fit the model with parameter grid search
            try:
                self._monitor_outputs['fit_time'].append(-time.time())
                if partial_fittable:
                    self.model = self.model.partial_fit(X[selected_idx], labels[selected_idx])
                else:
                    self.model = self.model.fit(X_known, labels_known)
                self._monitor_outputs['fit_time'][-1] += time.time()
            except Exception as e:
                msg = 'Failed fitting model. \n Known IDs: {}. \n ' \
                      'Known data shape: {}.'.format(known_idx, X[known_idx].shape)
                self._log.exception(msg)
                raise Exception(msg)

            # Call monitors
            for monitor in monitors:
                if n_iter % monitor.frequency == 0:
                    try:
                        self._monitor_outputs[monitor.short_name + '_time'].append(-time.time())
                        self._monitor_outputs[monitor.short_name].append(monitor(self.model, X, labels))
                        self._monitor_outputs[monitor.short_name + '_time'][-1] += time.time()
                    except Exception as e:
                        msg = 'Failed calling monitor {} with error {}'.format(monitor.name, e)
                        self._log.exception(msg)
                        raise Exception(msg)

            self._monitor_outputs['iter_time'][-1] += time.time()

            n_iter += 1

    def predict(self, X):
        """
        Predict using base_estimator

        X: numpy.ndarray
            numpy array with shape (p, m)

        Returns
        -------
        predictions: numpy.ndarray
            Array with shape (p, ) with the estimated predictions
        """
        if self.model is None:
            raise RuntimeError("Model needs to be fitted before using predict method.")

        return self.model.predict(X)