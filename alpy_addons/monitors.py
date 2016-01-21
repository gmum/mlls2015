from alpy.monitors import BaseMonitor
from sklearn.base import BaseEstimator
import numpy as np
from alpy.utils import _check_masked_labels, unmasked_indices, masked_indices
import copy
import logging

logger = logging.getLogger(__name__)

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

    def __init__(self, name, short_name, function, frequency=1, ids="all", X=None, y=None, **kwargs):
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
        -------

        """

        assert ids in ['known', 'unknown', 'all']

        if not callable(function):
            raise TypeError("`function` is expected to be callable, got {}".format(type(function)))

        self.function = function
        self.ids = ids

        if X is not None and y is not None:
            self.X = X
            self.y = y
            self.holdout = True
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
            # TODO: copy! What can we do about this?
            X = self.X[:, unmasked_indices(labels)] if pairwise else self.X

            assert isinstance(X, np.ndarray), "Not supported masked array here"

            pred_y = estimator.predict(X)
            labels = self.y.data if isinstance(self.y, np.ma.masked_array) else self.y
        else:
            if X is None or labels is None:
                raise ValueError("`X` and `y` can't be None for non-holdout validation")

            if self.ids == "known":
                X = X[unmasked_indices(labels)][:, unmasked_indices(labels)] if pairwise else X[unmasked_indices(labels)]
                labels = labels[unmasked_indices(labels)].data
            elif self.ids == "unknown":
                X = X[masked_indices(labels)][:, unmasked_indices(labels)] if pairwise else X[masked_indices(labels)]
                labels = labels[masked_indices(labels)].data
            else:
                X = X[:, unmasked_indices(labels)] if pairwise else X
                labels = labels.data

            if isinstance(X, np.ma.masked_array):
                X = X.data

            if X.shape[0] > 0:
                pred_y = estimator.predict(X)
            else:
                pred_y = None

        if pred_y is not None:
            return {"score": self.function(labels, pred_y), "predictions": pred_y.astype("int"),
                "true": labels.astype("int")}
        else:
            return {"score": 1, "predictions": [],
                "true": []}

