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
import traceback
from models.utils import GridSearch

class ActiveLearningExperiment(BaseEstimator):

    def __init__(self,
                 strategy,
                 base_model_cls,
                 batch_size,
                 param_grid,
                 metrics=[wac_score, mcc, recall_score, precision_score],
                 random_state=777,
                 logger=main_logger,
                 n_iter=None,
                 n_label=None,
                 strategy_projection_h=None,
                 n_folds=3,
                 strategy_kwargs={},
                 adaptive_grid=False):
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

        self.logger = logger

        self.strategy_name = strategy.__name__
        assert self.strategy_name in ["quasi_greedy_batch", "czarnecki_two_clusters",\
                                      "chen_krause", "random_query", "czarnecki", "uncertainty_sampling", "multiple_pick_best"]
        self.strategy_projection_h = strategy_projection_h
        self.strategy_requires_D = strategy.__name__ in ["quasi_greedy_batch"] or \
            strategy.__name__ in ["multiple_pick_best"] or \
            strategy.__name__ in ["czarnecki_two_clusters"]

        self.D = None
        self.strategy = partial(strategy, **strategy_kwargs)
        self.base_model_cls = base_model_cls
        self.model = None

        self.batch_size = batch_size
        self.rng = random_state
        self.metrics = metrics

        # fit args - for active learning loop
        self.n_iter = n_iter
        self.n_label = n_label
        self.param_grid = param_grid
        self.n_folds = n_folds
        self.adaptive_grid = adaptive_grid





    # TODO: Refactor to only 2 arguments and we want to base on GridSearchCV from sk, passing split strategy
    def fit(self, X, y, test_error_datasets=[]):
        """
        :param test_error_datasets. Will calculate error on those datasets:
            list of tuple ["name", (X,y)] or list of indexes of train X, y
        >>>model.fit(X, y, [("concept", (X_test, y_test)), ("main_cluster", ids))])
        """

        rng = check_random_state(self.rng)
        self.strategy_projection_seed = rng.randint(0,100)

        self.D = get_tanimoto_pairwise_distances(loader=X["i"]["loader"], preprocess_fncs=X["i"]["preprocess_fncs"],
                                                     name=X["i"]["name"])

        X_info = X["i"]
        X = X["data"]

        self.monitors = defaultdict(list)
        # self.base_model = self.base_model_cls()

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

        self.logger.info("Running Active Learninig Experiment for approximately "+str(max_iteration) + " iterations")
        self.logger.info("Logging concept error every iteration")

        if self.n_label is None and self.n_iter is None:
            self.n_label = X.shape[0]

        self.logger.info("Warm start size: " + str(len(y.known_ids)))

        # 0 warm start
        labeled = len(y.known_ids)
        if len(y.known_ids) == 0:
            labeled = self._query_labels(X, X_info, y, model=None, rng=rng)
            self.logger.info("WARNING: Model performing random query, because all labels are unknown")

        self.grid_seed = rng.randint(100)

        while True:

            # We assume that in first iteration first query is performed for us
            if self.monitors['iter'] != 0:
                labeled = self._query_labels(X, X_info, y, model=self.model, rng=rng)

            # Fit model parameters
            start = time.time()
            scorer = make_scorer(self.metrics[0])

            try:
                # Some if-ology to make sure we don't crash too often here.
                if len(y.known_ids) < 10:
                    n_folds = 2
                else:
                    n_folds = self.n_folds

                seed = self.grid_seed + self.monitors['iter']
                grid = GridSearch(base_model_cls=self.base_model_cls,
                                       param_grid=self.param_grid,
                                       seed=seed,
                                       n_folds=n_folds,
                                       adaptive=self.adaptive_grid)

                self.model = grid.fit(X[y.known_ids], y[y.known_ids])
            except Exception, e:
                self.logger.error(y.known_ids)
                self.logger.error(X[y.known_ids].shape)
                self.logger.error("Failed to fit grid!. Fitting random parameters!")
                self.logger.error(str(e))
                self.logger.error(traceback.format_exc())
                self.model = self.base_model_cls().fit(X[y.known_ids], y[y.known_ids])


            self.monitors['grid_times'].append(time.time() - start)


            self.monitors['n_already_labeled'].append(self.monitors['n_already_labeled'][-1] + labeled)
            self.monitors['iter'] += 1

            self.logger.info("Iter: %i, labeled %i/%i"
                             % (self.monitors['iter'], self.monitors['n_already_labeled'][-1], self.n_label))

            # Test on supplied datasets

            start = time.time()
            for reported_name, D in test_error_datasets:
                if len(D) > 2 and isinstance(D, list):
                    X_test = X[D]
                    y_test = y[D]
                elif len(D) == 2:
                    X_test = D[0]
                    y_test = D[1]
                else:
                    raise ValueError("Incorrect format of test_error_datasets")

                pred = self.model.predict(X_test)

                for metric in self.metrics:
                    self.monitors[metric.__name__ + "_" + reported_name].append(metric(y_test, pred))

            self.monitors['concept_test_times'].append(time.time() - start)

            # test on remaining training data
            if self.n_label - self.monitors['n_already_labeled'][-1] > 0:
                start = time.time()
                pred = self.model.predict(X[np.invert(y.known)])
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




    def _query_labels(self, X, X_info, y, model, rng):

        if len(y.unknown_ids) <= self.batch_size:
            labeled = len(y.unknown_ids)
            y.query(y.unknown_ids)
            self.monitors['strat_times'].append(0)
            return labeled

        # We have to acquire at least one example of negative and postivie class
        # We need to sample at least 10 examples for grid to work
        # We need to label at least one example :)
        labeled = 0
        while labeled==0 or len(np.unique(y[y.known_ids])) <= 1 or len(y.known_ids) < 10:
            # Check for warm start

            if self.monitors['iter'] == 0 and len(y.known_ids) <= 10:
                ind_to_label, _ = random_query(X, y,
                                            None,
                                            self.batch_size,
                                            rng, D=self.D)
            else:
                start = time.time()

                # TODO: fix that - we shouldn't be doing this magic here.
                D = self.D
                if hasattr(self.base_model_cls, "project") and self.strategy_name == "chen_krause":
                    self.logger.info("Projecting dataset for strategy")
                    X = model.transform(X)
                    D = None # This is a hack. We cannot/shouldnt calculate it each iteration, but if we have to
                             # we should rethink how to unify this with caching for other strategies.
                    assert self.strategy_projection_h is None
                elif self.strategy_projection_h and self.strategy_name == "chen_krause":
                    X = get_tanimoto_projection(loader=X_info["loader"], preprocess_fncs=X_info["preprocess_fncs"],
                                                         name=X_info["name"], seed=self.strategy_projection_seed,
                                                         h=self.strategy_projection_h, normalize=True)
                elif "czarnecki" in  self.strategy_name:

                    if hasattr(self.base_model_cls, "project"):
                        raise ValueError("Should have projected data in model already - conflict.")

                    # For czarnecki strategy use always full projection
                    X = [X, get_tanimoto_projection(loader=X_info["loader"], preprocess_fncs=X_info["preprocess_fncs"],
                                                             name=X_info["name"], seed=self.strategy_projection_seed,
                                                             h=X.shape[0]), \
                         get_sorensen_projection(loader=X_info["loader"], preprocess_fncs=X_info["preprocess_fncs"],
                                                             name=X_info["name"], seed=self.strategy_projection_seed,
                                                             h=X.shape[0])
                         ]
                    assert X[0].shape[0] == X[1].shape[0]
                elif self.strategy_name == "chen_krause":
                    raise ValueError("Failed because didn't project data")




                ind_to_label, _ = self.strategy(X=X, y=y, current_model=model, \
                                                batch_size=self.batch_size, rng=rng, D=D)

                assert len(ind_to_label) == self.batch_size, "Received required number of examples to query"

                self.monitors['strat_times'].append(time.time() - start)
            labeled += len(ind_to_label)
            y.query(ind_to_label)
        return labeled

    def predict(self, X):

        return self.model.predict(X)
