from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator
import numpy as np
from alpy2.utils import _check_masked_labels, unmasked_indices, masked_indices
import copy
import logging
from scipy.sparse import issparse

logger = logging.getLogger(__name__)


class BaseMonitor(object):
    """
    Base monitor meta class
    """

    __metaclass__ = ABCMeta

    def __init__(self, name, short_name, frequency=1):
        """
        Parameters
        ----------
        name: string
                Full name of the monitor
        short_name: string
                Shortened name of the monitor
        frequency: integer
                How often to call the monitor, default: 1

        -------

        """
        self.name = name
        self.short_name = short_name

        if not isinstance(frequency, int) or frequency <= 0:
            raise TypeError('`validation_frequency` is expected to be positive integer, \
            got{}'.format(frequency))

        self.frequency = frequency

    @abstractmethod
    def __call__(self, estimator, X, labels):
        raise NotImplementedError("Virtual method got called")


class EstimatorMonitor(BaseMonitor):
    def __init__(self, only_params):
        self.only_params = only_params
        super(EstimatorMonitor, self).__init__(name="EstimatorMonitor", short_name="est_mon")

    def __call__(self, estimator, X, labels):
        if not isinstance(estimator, BaseEstimator):
            raise TypeError("Got bad estimator: {}".format(type(estimator)))

        if self.only_params:
            return copy.deepcopy(estimator.get_params())
        else:
            return copy.copy(estimator)


class GridScoresMonitor(BaseMonitor):
    def __init__(self):
        super(GridScoresMonitor, self).__init__(name="GridScoresMonitor", short_name="grid_mon")

    def __call__(self, estimator, X, labels):
        if not isinstance(estimator, BaseEstimator):
            raise TypeError("Got bad estimator: {}".format(type(estimator)))

        # Type standardization
        return [list(score_tuple) for score_tuple in estimator.grid_scores_]


class SimpleLogger(BaseMonitor):
    def __init__(self, batch_size=10, frequency=1):
        self.n_iter = 0
        self.batch_size = batch_size
        super(SimpleLogger, self).__init__(name="SimpleLogger", short_name="simple_logger", frequency=frequency)

    def __call__(self, estimator, X, labels):
        logger.info("iter " + str(self.n_iter))
        self.n_iter += self.batch_size
        return 0


class ExtendedMetricMonitor(BaseMonitor):
    """
    Base class for validating classification on either training data, or holdout set
    """

    def __init__(self, name,
                 short_name,
                 function,
                 frequency=1,
                 ids="all",
                 X=None,
                 y=None,
                 duds_mask=None,
                 **kwargs):
        """

        Parameters
        ----------
        ids: string, default: "all"
                Which ids to use. Possible values are "all", "known", "unknown"
        name: string
                Full name of the monitor
        short_name: string
                Shortened name of the monitor
        function: callable
                Function for calculation given classification metric
        frequency: integer
                How often to call the monitor, default: 1
        X: np.array
                Holdout set to validate on, if omitted, the metric will be calcualted on training data.
                Requires `y` to be provided for validating
        y: np.array
                Labels for holdout set,
        duds_mask: np.array
                Indices of DUDs in data, if theya are there
        -------

        """

        assert ids in ['known', 'unknown', 'all']

        if not callable(function):
            raise TypeError("`function` is expected to be callable, got {}".format(type(function)))

        self.function = function
        self.ids = ids
        self.duds = False

        if duds_mask is not None:
            self.duds_mask = duds_mask
            self.duds_ids = np.where(self.duds_mask == 1)[0]
            self.duds = True

        if X is not None and y is not None:
            self.X = X
            self.y = y
            self.holdout = True

            if self.duds and self.X.shape[0] != self.duds_mask.shape[0]:
                # import pdb
                # pdb.set_trace()
                raise ValueError("`X` and `duds_ids` need to have the same length")
        else:
            self.holdout = False

        super(ExtendedMetricMonitor, self).__init__(name=name, short_name=short_name, frequency=frequency)

    def __call__(self, estimator, X=None, labels=None):

        if not isinstance(estimator, BaseEstimator):
            raise TypeError("Got bad estimator: {}".format(type(estimator)))

        pairwise = getattr(estimator, "_pairwise", False) or \
                   getattr(getattr(estimator, "estimator", {}), "_pairwise", False)

        if self.holdout:
            assert self.ids == "all", "Not suported ids type for holdout dataset"

            unmasked_ids = unmasked_indices(labels)
            all_ids = np.arange(self.X.shape[0])

            # TODO: copy! What can we do about this?
            X = self.X[:, unmasked_ids] if pairwise else self.X

            if self.duds:

                all_ids_without_duds = np.setdiff1d(all_ids,  self.duds_ids, assume_unique=True)
                unmasked_ids_without_duds = np.setdiff1d(unmasked_ids,  self.duds_ids, assume_unique=True)

                X_without_duds = self.X[all_ids_without_duds][:, unmasked_ids] if pairwise else self.X[all_ids_without_duds]
                labels_without_duds = self.y[all_ids_without_duds].data if isinstance(self.y, np.ma.masked_array) else self.y[all_ids_without_duds]
                assert isinstance(X_without_duds, np.ndarray) or issparse(X), "Not supported masked array here"

            assert isinstance(X, np.ndarray) or issparse(X), "Not supported masked array here"

            pred_y = estimator.predict(X)
            test_labels = self.y.data if isinstance(self.y, np.ma.masked_array) else self.y

            if self.duds:
                pred_without_duds = estimator.predict(X_without_duds)

        else:
            all_ids = np.arange(X.shape[0])
            masked_ids = masked_indices(labels)
            unmasked_ids = unmasked_indices(labels)

            if self.duds:
                if X.shape[0] != self.duds_mask.shape[0]:
                    raise ValueError("`X` and `duds_ids` need to have the same length")

                all_ids_without_duds = np.setdiff1d(all_ids,  self.duds_ids, assume_unique=True)
                masked_ids_without_duds = np.setdiff1d(masked_ids,  self.duds_ids, assume_unique=True)
                unmasked_ids_without_duds = np.setdiff1d(unmasked_ids,  self.duds_ids, assume_unique=True)

            if X is None or labels is None:
                raise ValueError("`X` and `y` can't be None for non-holdout validation")


            if self.ids == "known":
                test_X = X[unmasked_ids][:, unmasked_ids] if pairwise else X[unmasked_ids]
                test_labels = labels[unmasked_ids].data

                if self.duds:
                    X_without_duds = X[unmasked_ids_without_duds][:, unmasked_ids] if pairwise else X[unmasked_ids_without_duds]
                    labels_without_duds = labels[unmasked_ids_without_duds].data

            elif self.ids == "unknown":
                test_X = X[masked_ids][:, unmasked_ids] if pairwise else X[masked_ids]
                test_labels = labels[masked_ids].data

                if self.duds:
                    X_without_duds = X[masked_ids_without_duds][:, unmasked_ids] if pairwise else X[masked_ids_without_duds]
                    labels_without_duds = labels[masked_ids_without_duds].data

            else: # all ids
                test_X = X[:,unmasked_ids] if pairwise else X
                test_labels = labels.data

                if self.duds:
                    X_without_duds = X[all_ids_without_duds][:, unmasked_ids] if pairwise else X[all_ids_without_duds]
                    labels_without_duds = labels[all_ids_without_duds].data

            if isinstance(test_X, np.ma.masked_array):
                test_X = test_X.data


            if test_X.shape[0] > 0:
                pred_y = estimator.predict(test_X)
            else:
                pred_y = None

            if self.duds:
                if isinstance(X_without_duds, np.ma.masked_array):
                    X_without_duds = X_without_duds.data

                if X_without_duds.shape[0] > 0:
                    pred_without_duds = estimator.predict(X_without_duds)
                else:
                    pred_without_duds = None


        if pred_y is not None:
            results = {"score": self.function(test_labels, pred_y),
                    "predictions": pred_y.astype("int"),
                    "true": test_labels.astype("int")}
            if self.duds:
                if pred_without_duds is not None:
                    results.update({"score-duds": self.function(labels_without_duds, pred_without_duds),
                                    "predictions-duds": pred_without_duds.astype("int"),
                                    "true-duds": labels_without_duds.astype("int")})
                else:
                    results.update({"score-duds": 1,
                                    "predictions-duds": [],
                                    "true-duds": []})

            return results
        else:
            return {"score": 1,
                    "predictions": [],
                    "true": [],
                    "score-duds": 1,
                    "predictions-duds": [],
                    "true-duds": []}

