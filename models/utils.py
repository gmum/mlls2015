import numpy as np
from get_data import _generate_fold_indices
from sklearn.grid_search import ParameterGrid
from sklearn.base import BaseEstimator
from experiments.utils import wac_score

class ObstructedY(object):
    
    def __init__(self, y):
        self._y = np.array(y)
        self.size = len(y)
        self.shape = y.shape
        self.known = np.array([False for _ in xrange(self.size)])
        self.unknown_ids = np.where(self.known == False)[0]
        self.known_ids = np.where(self.known == True)[0]
        self.classes = np.unique(self._y)

    def __getitem__(self, key):
        if isinstance(key, int):
            if key > self.size :
                raise IndexError, "Index %i out of bound for array of size %i" % (key, self.size)

            if not self.known[key]:
                raise IndexError, "Element at index %i needs to be queried before accessing it!" % key

            return self._y[key]
        elif isinstance(key, slice):
            if max([key.start, key.stop, key.step]) >= self.size:
                raise IndexError, "Index out of bound"
            if not all(self.known[key]):
                raise IndexError, "All elements need to be queried before accessing them!"
            return self._y[key]
        elif isinstance(key, list):
            if not all (self.known[key]):
                raise IndexError, "All elements need to be queried before accessing them!"
            return self._y[key]
        elif isinstance(key, np.ndarray):
            if not all (self.known[key]):
                raise IndexError, "All elements need to be queried before accessing them!"
            return self._y[key]

        else:
            raise TypeError, "Invalid argument type"

    def query(self, ind):
        self.known[ind] = True
        self.unknown_ids = np.where(self.known == False)[0]
        self.known_ids = np.where(self.known == True)[0]
        return self._y[ind]

    def peek(self):
        return self._y[np.invert(self.known)]


class GridSearch(BaseEstimator):

    def __init__(self,
                 base_model_cls,
                 param_grid,
                 seed,
                 score=wac_score,
                 n_folds=5,
                 test_size=0.1,
                 refit=True,
                 adaptive=False):

        self.base_model_cls = base_model_cls
        self.param_grid = param_grid
        self.seed = seed
        self.n_folds = n_folds
        self.test_size = test_size
        self.refit = refit
        self.score = score
        self.adaptive = adaptive
        self.best_model = None

        self.param_list = None
        self.folds = None
        self.best_model = None
        self.results = None
        self.best_params = None


    def fit(self, X, y):

        self.folds = _generate_fold_indices(y, self.test_size, self.seed, self.n_folds)
        assert len(self.folds) == self.n_folds

        # adaptive
        if self.best_params is not None and self.adaptive:
            param_grid = {}
            for key, best_param in self.best_params.iteritems():
                i = self.param_grid[key].index(best_param)
                if i != 0 and i != len(self.param_grid[key]) - 1:
                    param_grid[key] = [self.param_grid[key][j] for j in [i-1, i, i+1]]
                elif i == 0:
                    param_grid[key] = [self.param_grid[key][j] for j in [i, i+1, i+2]]
                elif i == len(self.param_grid[key]) - 1:
                    param_grid[key] = [self.param_grid[key][j] for j in [i-2, i-1, i]]

            self.param_list = list(ParameterGrid(param_grid))
        else:
            self.param_list = list(ParameterGrid(self.param_grid))

        self.results = [0 for _ in xrange(len(self.param_list))]

        for i, params in enumerate(self.param_list):
            scores = []

            for train_id, test_id in self.folds:
                model = self.base_model_cls(**params)
                model.fit(X[train_id], y[train_id])
                pred = model.predict(X[test_id])

                scores.append(self.score(y[test_id], pred))

            self.results[i] = np.mean(scores)

        self.best_params = self.param_list[np.argmax(self.results)]

        if self.refit:
            self.best_model = self.base_model_cls(**self.best_params)
            self.best_model.fit(X, y)

        return self

    def predict(self, X):
        if self.best_model is None or not self.refit:
            raise AttributeError("You need to fit the grid first and pass refit=True")
        else:
            return self.best_model.predict(X)

    def transform(self, X):
        if not hasattr(self.best_model, 'transform'):
            raise AttributeError("base model has no attribute transform")
        else:
            return self.best_model.transform(X)

    def decision_function(self, X):
        return self.best_model.decision_function(X)
