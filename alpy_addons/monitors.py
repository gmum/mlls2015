from alpy.monitors import BaseMonitor
from sklearn.base import BaseEstimator
import numpy as np
from alpy.utils import _check_masked_labels, unmasked_indices, masked_indices

class MetricMonitor(BaseMonitor):
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

        super(MetricMonitor, self).__init__(name=name, short_name=short_name, frequency=frequency)

    def __call__(self, estimator, X=None, labels=None):

        if not isinstance(estimator, BaseEstimator):
            raise TypeError("Got bad estimator: {}".format(type(estimator)))

        if self.holdout:
            assert self.ids == "all", "Not suported ids type for holdout dataset"
            pred_y = estimator.predict(self.X)
            labels = self.y.data if isinstance(self.y, np.ma.masked_array) else self.y
        else:
            if X is None or labels is None:
                raise ValueError("`X` and `y` can't be None for non-holdout validation")

            if self.ids == "known":
                X = X[unmasked_indices(labels)]
                labels = labels[unmasked_indices(labels)]
            elif self.ids == "unknown":
                X = X[masked_indices(labels)]
                labels = labels[masked_indices(labels)]
            else:
                labels = labels.data

            if X.shape[0] > 0:
                pred_y = estimator.predict(X)
            else:
                pred_y = None

        if pred_y is not None:
            return self.function(labels, pred_y)
        else:
            return 0

