import sys
sys.path.append("..")
from misc.config import *
log = main_logger
from get_data import *
from models.strategy import *
import sys
sys.path.append("..")
import unittest
import numpy as np
from models.balanced_models import *
from sklearn.linear_model import SGDClassifier

class TestRefittingModels(unittest.TestCase):

    def setUp(self):
        comps = [['5ht7', 'ExtFP']]

        loader = ["get_splitted_data", {
                "seed": 666,
                "valid_size": 0.1,
                "n_folds": 1}]

        preprocess_fncs = []

        data = get_data(comps, loader, preprocess_fncs).values()[0][0][0]
        self.X = data['X_train']['data']
        self.y = data['Y_train']['data']

    def test_twelm(self):

        projector = RandomProjector()
        model = TWELM(projector, random_state=666)

        model.fit(self.X, self.y)
        beta = model.beta

        model.fit(self.X, self.y)
        assert np.array_equal(beta, model.beta)

    def test_eem(self):

        projector = RandomProjector()
        model = EEM(projector, random_state=666)

        model.fit(self.X, self.y)
        beta = model.beta

        model.fit(self.X, self.y)
        assert np.array_equal(beta, model.beta)

    def test_svmtan(self):

        model = SVMTAN(random_state=666)

        model.fit(self.X, self.y)
        sv = model.clf.support_vectors_

        model.fit(self.X, self.y)
        assert np.array_equal(sv, model.clf.support_vectors_)

    def randomNB(self):

        projector = RandomProjector()
        model = RandomNB(projector, random_state=666)

        model.fit(self.X, self.y)
        mu = model.clf.theta_
        sigma = model.clf.sigma_

        model.fit(self.X, self.y)
        assert np.array_equal(mu, model.clf.theta_)
        assert np.array_equal(sigma, model.clf.sigma_)