from sklearn.base import BaseEstimator
from models.utils import ObstructedY
from models.strategy import random_query
from sklearn.metrics import matthews_corrcoef as mcc, recall_score, precision_score
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from experiments.utils import wac_score
import numpy as np
from sklearn.metrics import make_scorer
from time import time

from misc.config import main_logger, c
from collections import defaultdict

class ActiveLearningExperiment(BaseEstimator):

    def __init__(self,
                 strategy,
                 base_model_cls,
                 batch_size,
                 param_grid,
                 metrics=[wac_score, mcc, recall_score, precision_score],
                 concept_error_log_freq=0.05,
                 seed=666,
                 n_iter=None,
                 n_label=None,
                 n_folds=3):
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
        self.base_model_cls = base_model_cls

        self.concept_error_log_freq = concept_error_log_freq

        self.batch_size = batch_size
        self.seed = seed
        self.metrics = metrics

        # fit args - for active learning loop
        self.n_iter = n_iter
        self.n_label = n_label
        self.param_grid = param_grid
        self.n_folds = n_folds




    # TODO: Refactor to only 2 arguments and we want to base on GridSearchCV from sk, passing split strategy
    def fit(self, X, y, test_error_datasets=[]):
        """
        :param test_error_datasets. Will calculate error on those datasets:
            list of tuple ["name", (X,y)] or list of indexes of train X, y
        >>>model.fit(X, y, [("concept", (X_test, y_test)), ("main_cluster", ids))])
        """
        self.monitors = defaultdict(list)
        self.base_model = self.base_model_cls()

        if not isinstance(y, ObstructedY):
            y = ObstructedY(y)


        self.monitors['n_already_labeled'] = [0]
        self.monitors['iter'] = 0

        # times
        self.monitors['strat_times'] = []
        self.monitors['grid_times'] = []
        self.monitors['concept_test_times'] = []
        self.monitors['unlabeled_test_times'] = []

        max_iteration = (y.shape[0] - y.known.sum())/self.batch_size + 1

        concept_error_log_step= max(1, int(self.concept_error_log_freq * max_iteration))

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
                start = time()
                ind_to_label, _ = self.strategy(X=X, y=y, current_model=self.grid, \
                                             batch_size=self.batch_size, seed=self.seed)
                self.monitors['start_times'].append(time() - start)

            y.query(ind_to_label)

            scorer = make_scorer(self.metrics[0])

            self.grid = GridSearchCV(self.base_model,
                                     self.param_grid,
                                     scoring=scorer,
                                     n_jobs=1,
                                     cv=StratifiedKFold(n_folds=self.n_folds, y=y[y.known], random_state=self.seed))
            start = time()
            self.grid.fit(X[y.known], y[y.known])
            self.monitors['grid_times'].append(time() - start)

            #self.base_model.fit(X[y.known], y[y.known])

            self.monitors['n_already_labeled'].append(self.monitors['n_already_labeled'][-1] + len(ind_to_label))
            self.monitors['iter'] += 1

            main_logger.info("Iter: %i, labeled %i/%i"
                                 % (self.monitors['iter'], self.monitors['n_already_labeled'][-1], self.n_label))

            # test concept error
            if self.monitors['iter'] % concept_error_log_step == 0:
                for reported_name, D in test_error_datasets:
                    if len(D) > 2 and isinstance(D, list):
                        X_test = X[D]
                        y_test = y[D]
                    elif len(D) == 2:
                        X_test = D[0]
                        y_test = D[1]
                    else:
                        raise ValueError("Incorrect format of test_error_datasets")

                    start = time()
                    pred = self.grid.predict(X_test)
                    self.monitors['concept_test_times'].append(time() - start)

                    for metric in self.metrics:
                        self.monitors[metric.__name__ + "_" + reported_name].append(metric(y_test, pred))

                # test on remaining training data
                if self.n_label - self.monitors['n_already_labeled'][-1] > 0:
                    start = time()
                    pred = self.grid.predict(X[np.invert(y.known)])
                    self.monitors['unlabeled_test_times'].append(time() - start)
                    for metric in self.metrics:
                        self.monitors[metric.__name__ + "_unlabeled"].append(metric(y.peek(), pred))


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

        return self.grid.predict(X)
