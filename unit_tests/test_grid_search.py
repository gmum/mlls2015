import unittest
import numpy as np
from get_data import get_data
import kaggle_ninja
from models.balanced_models import *
from experiments.utils import wac_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from functools import partial
from models.utils import GridSearch
import time

class TestGridSearch(unittest.TestCase):

    def setUp(self):

        comps = [['5ht7', 'ExtFP']]

        loader = ["get_splitted_data", {
                "seed": 666,
                "valid_size": 0.25,
                "n_folds": 1,
                "percent": 0.5}]

        preprocess_fncs = []

        data = get_data(comps, loader, preprocess_fncs).values()[0][0][0]
        self.X = data['X_train']['data']
        self.y = data['Y_train']['data']

        self.X_test = data['X_valid']['data']
        self.y_test = data['Y_valid']['data']

        self.elm_param_grid = {'C': list(np.logspace(0, 5, 6)),
                               'h': [100, 200, 500, 1000]}

        self.svm_param_grid = {'C': list(np.logspace(-3, 4, 8))}
        self.nb_param_grid = {'h': [100, 200, 500, 1000]}

    def test_data(self):
        print "train size", self.y.shape
        print "test size", self.y_test.shape
        assert self.X.shape[0] == self.y.shape[0]
        assert self.X_test.shape[0] == self.y_test.shape[0]

    def test_twelm(self):

        projector = RandomProjector()
        model = partial(TWELM, projector=projector, random_state=666)

        grid = GridSearch(base_model_cls=model,
                          param_grid=self.elm_param_grid,
                          seed=666,
                          score=wac_score)

        start_time = time.time()
        grid.fit(self.X, self.y)
        grid_time = time.time() - start_time

        best_params = grid.best_params

        print "our params", grid.best_params
        print "results", max(grid.results)

        folds = grid.folds
        sk_model = TWELM(projector, random_state=666)
        sk_scorer = make_scorer(wac_score)

        sk_grid = GridSearchCV(estimator=sk_model,
                               param_grid=self.elm_param_grid,
                               scoring=sk_scorer,
                               cv=folds)

        start_time = time.time()
        sk_grid.fit(self.X, self.y)
        sk_time = time.time() - start_time

        sk_best_params = sk_grid.best_params_

        print "sk params", sk_grid.best_params_
        print "sk score", sk_grid.best_score_

        print "sk_time", sk_time
        print "our_time", grid_time

        for key in self.elm_param_grid.keys():
            print "our", best_params[key]
            print "sk", sk_best_params[key]


            assert best_params[key] == sk_best_params[key]


if __name__ == "__main__":
    unittest.main()


