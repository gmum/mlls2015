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
                 concept_error_log_freq=0.05,
                 seed=777,
                 n_iter=None,
                 n_label=None):
        """
        :param strategy:
        :param base_model_cls:
        :param batch_size:
        :param metrics:
        :param concept_error_log_freq: Every concept_error_log_freq*100% of iterations it will calculate error
        :param seed:
        :param n_iter:
        :param n_label:
        :return:
        """

        assert isinstance(metrics, list), "please pass metrics as a list"

        self.strategy = strategy
        self.base_model = base_model_cls()
        self.has_partial = hasattr(self.base_model, 'partial_fit')

        self.concept_error_log_freq = concept_error_log_freq

        self.batch_size = batch_size
        self.seed = seed
        self.metrics = metrics

        # fit args - for active learning loop
        self.n_iter = n_iter
        self.n_label = n_label

        self.monitors = {}
        for metric in self.metrics:
            self.monitors.update({metric.__name__ + "_concept": [],
                                  metric.__name__ + '_not_seen': []})


    # TODO: Refactor to only 2 arguments and we want to base on GridSearchCV from sk, passing split strategy
    def fit(self, X, y, X_test=None, y_test=None):

        if not isinstance(y, ObstructedY):
            y = ObstructedY(y)


        self.monitors['n_already_labeled'] = [0]
        self.monitors['iter'] = 0

        max_iteration = (y.shape[0] - y.known.sum())/self.batch_size + 1

        concept_error_log_step= int(self.concept_error_log_freq * max_iteration)

        main_logger.info("Running Active Learninig Experiment for approximately "+str(max_iteration) + " iterations")
        main_logger.info("Logging concept error every "+str(concept_error_log_step)+" iterations")

        if self.n_label is None and self.n_iter is None:
            self.n_label = X.shape[0]

        while True:

            # check for warm start
            if self.monitors['iter'] == 0 and not any(y.known):
                ind_to_label, _ = random_query(X, y,
                                            None,
                                            self.batch_size,
                                            self.seed)
            else:
                ind_to_label, _ = self.strategy(X=X, y=y, current_model=self.base_model, \
                                             batch_size=self.batch_size, seed=self.seed)


            y.query(ind_to_label)

            if self.has_partial:
                self.base_model.partial_fit(X[ind_to_label], y[ind_to_label], classes=y.classes)
            else:
                self.base_model.fit(X[y.known], y[y.known])

            self.monitors['n_already_labeled'].append(self.monitors['n_already_labeled'][-1] + len(ind_to_label))
            self.monitors['iter'] += 1

            if self.monitors['iter'] % 100 == 0:
                main_logger.debug("Iter: %i, labeled %i/%i"
                                 % (self.monitors['iter'], self.monitors['n_already_labeled'][-1], self.n_label))

            # test concept error
            if self.monitors['iter'] % concept_error_log_step == 0:
                if X_test is not None and y_test is not None:
                    pred = self.base_model.predict(X_test)
                    for metric in self.metrics:
                        self.monitors[metric.__name__ + "_concept"].append(metric(y_test, pred))

                # test on remaining training data
                if self.n_label - self.monitors['n_already_labeled'][-1] > 0:
                    pred = self.base_model.predict(X[np.invert(y.known)])
                    for metric in self.metrics:
                        self.monitors[metric.__name__ + "_not_seen"].append(metric(y.peek(), pred))


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
