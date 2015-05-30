from sklearn.base import BaseEstimator
from models.utils import ObstructedY
from models.strategy import random_query
from sklearn.metrics import matthews_corrcoef as mcc
import numpy as np

from misc.config import main_logger, c


class ActiveLearningExperiment(BaseEstimator):

    def __init__(self,
                 strategy,
                 base_model_cls,
                 batch_size,
                 metrics=[mcc],
                 seed=666,
                 n_iter=None,
                 n_label=None):

        assert isinstance(metrics, list), "please pass metrics as a list"

        self.strategy = strategy
        self.base_model = base_model_cls()
        self.has_partial = hasattr(self.base_model, 'partial_fit')

        self.batch_size = batch_size
        self.seed = seed
        self.metrics = metrics

        # fit args - for active learning loop
        self.n_iter = n_iter
        self.n_label = n_label

        self.monitors = {}
        for metric in self.metrics:
            self.monitors.update({metric.__name__ + "_concept": [],
                                  metric.__name__ + '_train': []})


    # TODO: Refactor to only 2 arguments and we want to base on GridSearchCV from sk, passing split strategy
    def fit(self, X, y, X_test=None, y_test=None):

        if not isinstance(y, ObstructedY):
            y = ObstructedY(y)

        self.monitors['n_already_labeled'] = [0]
        self.monitors['iter'] = 0

        if self.n_label is None and self.n_iter is None:
            self.n_label = X.shape[0]

        while True:

            # check for warm start
            if self.monitors['iter'] == 0 and not any(y.known):
                ind_to_label = random_query(X, y,
                                            None,
                                            self.batch_size,
                                            self.seed)
            else:
                ind_to_label = self.strategy(X=X, y=y, current_model=self.base_model, \
                                             batch_size=self.batch_size, seed=self.seed)


            y.query(ind_to_label)

            if self.has_partial:
                self.base_model.partial_fit(X[ind_to_label], y[ind_to_label], classes=y.classes)
            else:
                self.base_model.fit(X[y.known], y[y.known])

            self.monitors['n_already_labeled'].append(self.monitors['n_already_labeled'][-1] + len(ind_to_label))
            self.monitors['iter'] += 1

            main_logger.debug("Iter: %i, labeled %i/%i"
                             % (self.monitors['iter'], self.monitors['n_already_labeled'][-1], self.n_label))

            # test concept error
            if X_test is not None and y_test is not None:
                pred = self.base_model.predict(X_test)
                for metric in self.metrics:
                    self.monitors[metric.__name__ + "_concept"].append(metric(y_test, pred))

            # test on remaining training data
            if self.n_label - self.monitors['n_already_labeled'][-1] > 0:
                pred = self.base_model.predict(X[np.invert(y.known)])
                for metric in self.metrics:
                    self.monitors[metric.__name__ + "_train"].append(metric(y.peek(), pred))


            # check stopping criterions
            if self.n_iter is not None:
                if self.monitors['iter'] == self.n_iter:
                    break
            elif self.n_label - self.monitors['n_already_labeled'][-1] == 0:
                break
            elif self.n_label - self.monitors['n_already_labeled'][-1] < self.batch_size:
                self.batch_size = self.n_label - self.monitors['n_already_labeled'][-1]
                main_logger.debug("Decreasing batch size to: %i" % self.batch_size)

            assert self.batch_size >= 0


    def predict(self, X):

        return self.base_model.predict(X)
