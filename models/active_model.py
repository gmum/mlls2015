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
from functools import partial
from misc.config import main_logger, c
from collections import defaultdict
from sklearn.metrics import pairwise_distances
from models.strategy import jaccard_dist
from sklearn.utils import check_random_state
from get_data import *

class ActiveLearningExperiment(BaseEstimator):

    def __init__(self,
                 strategy,
                 base_model_cls,
                 batch_size,
                 param_grid,
                 metrics=[wac_score, mcc, recall_score, precision_score],
                 concept_error_log_freq=0.05,
                 random_state=777,
                 logger=main_logger,
                 n_iter=None,
                 n_label=None,
                 n_folds=3,
                 strategy_projection_h=None,
                 strategy_kwargs={}):
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

        self.strategy_projection_h = strategy_projection_h

        self.logger = logger

        self.strategy_requires_D = strategy.__name__ in ["quasi_greedy_batch"]
        self.D = None
        self.strategy = partial(strategy, **strategy_kwargs)
        self.base_model_cls = base_model_cls

        self.concept_error_log_freq = concept_error_log_freq

        self.batch_size = batch_size
        self.rng = random_state
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

        self.rng = check_random_state(self.rng)

        self.D = get_tanimoto_pairwise_distances(loader=X["i"]["loader"], preprocess_fncs=X["i"]["preprocess_fncs"],
                                                     name=X["i"]["name"])

        if self.strategy_projection_h is not None:
            # Seeding is tricky, enforced by kaggle_ninja - but still important that it is reproducible
            X_strategy = get_tanimoto_projection(loader=X["i"]["loader"], preprocess_fncs=X["i"]["preprocess_fncs"],
                                                         name=X["i"]["name"], seed=self.rng.randint(0,100),
                                                         h=self.strategy_projection_h)
        else:
            X_strategy = X["data"]

        X = X["data"]

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

        self.logger.info("Running Active Learninig Experiment for approximately "+str(max_iteration) + " iterations")
        self.logger.info("Logging concept error every "+str(concept_error_log_step)+" iterations")

        if self.n_label is None and self.n_iter is None:
            self.n_label = X.shape[0]

        self.logger.info("Warm start size: " + str(len(y.known_ids)))

        # 0 warm start
        labeled = len(y.known_ids)
        if len(y.known_ids) == 0:
            labeled = self._query_labels(X, y, X_strategy)
            self.logger.warning("Model performing random query, because all labels are unknown")

        while True:

            # We assume that in first iteration first query is performed for us
            if self.monitors['iter'] != 0:
                labeled = self._query_labels(X, y, X_strategy)
            # Fit model parameters
            start = time.time()
            scorer = make_scorer(self.metrics[0])

            try:
                # Some if-ology to make sure we don't crash too often here.
                if len(y.known_ids) < 10:
                    n_folds = 2
                else:
                    n_folds = self.n_folds
                self.grid = GridSearchCV(self.base_model,
                                         self.param_grid,
                                         scoring=scorer,
                                         n_jobs=1,
                                         cv=StratifiedKFold(n_folds=n_folds, y=y[y.known_ids], \
                                         random_state=self.rng))
                self.grid.fit(X[y.known_ids], y[y.known_ids])
            except Exception, e:
                self.logger.warning("Failed to fit grid!. Fitting random parameters!")
                self.logger.warning(str(e))
                self.grid = self.base_model_cls().fit(X[y.known_ids], y[y.known_ids])


            self.monitors['grid_times'].append(time.time() - start)


            self.monitors['n_already_labeled'].append(self.monitors['n_already_labeled'][-1] + labeled)
            self.monitors['iter'] += 1

            self.logger.info("Iter: %i, labeled %i/%i"
                                 % (self.monitors['iter'], self.monitors['n_already_labeled'][-1], self.n_label))

            # Test on supplied datasets
            if self.monitors['iter'] % concept_error_log_step == 0:

                for reported_name, D in test_error_datasets:
                    self.logger.info(reported_name)

                    if len(D) > 2 and isinstance(D, list):
                        X_test = X[D]
                        y_test = y[D]
                    elif len(D) == 2:
                        X_test = D[0]
                        y_test = D[1]
                    else:
                        raise ValueError("Incorrect format of test_error_datasets")

                    start = time.time()
                    pred = self.grid.predict(X_test)
                    self.monitors['concept_test_times'].append(time.time() - start)

                    for metric in self.metrics:
                        self.monitors[metric.__name__ + "_" + reported_name].append(metric(y_test, pred))

                # test on remaining training data
                if self.n_label - self.monitors['n_already_labeled'][-1] > 0:
                    start = time.time()
                    pred = self.grid.predict(X[np.invert(y.known)])
                    self.monitors['unlabeled_test_times'].append(time.time() - start)
                    for metric in self.metrics:
                        self.monitors[metric.__name__ + "_unlabeled"].append(metric(y.peek(), pred))


            # Check stopping criterions
            if self.n_iter is not None:
                if self.monitors['iter'] == self.n_iter:
                    break
            elif self.n_label - self.monitors['n_already_labeled'][-1] == 0:
                break
            elif self.n_label - self.monitors['n_already_labeled'][-1] < self.batch_size:
                self.batch_size = self.n_label - self.monitors['n_already_labeled'][-1]
                self.logger.debug("Decreasing batch size to: %i" % self.batch_size)

            assert self.batch_size >= 0




    def _query_labels(self,X ,y, X_strategy):
        # We have to acquire at least one example of negative and postivie class
        # We need to sample at least 10 examples for grid to work
        # We need to label at least one example :)
        labeled = 0
        while labeled==0 or len(np.unique(y[y.known_ids])) <= 1 or len(y.known_ids) < 10:
            # Check for warm start
            if self.monitors['iter'] == 0 and len(y.known_ids) == 0:
                ind_to_label, _ = random_query(X, y,
                                            None,
                                            self.batch_size,
                                            self.rng, D=self.D)
            else:
                start = time.time()
                ind_to_label, _ = self.strategy(X=X_strategy, y=y, current_model=self.grid, \
                                                batch_size=self.batch_size, rng=self.rng, D=self.D)
                self.monitors['strat_times'].append(time.time() - start)
            labeled += len(ind_to_label)
            y.query(ind_to_label)
        return labeled

    def predict(self, X):

        return self.grid.predict(X)
