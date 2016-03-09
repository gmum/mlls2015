# -*- coding: utf-8 -*-

"""
 Test class for all strategy methods
"""

import pytest

import numpy as np

from alpy2.strategy import UncertaintySampling, QueryByBagging, QuasiGreedyBatch
from alpy2.utils import mask_unknowns, unmasked_indices, masked_indices

from sklearn.svm import SVC
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

from itertools import product


class DummyStrategy(object):

    def __init__(self, seed):

        assert isinstance(seed, int)
        self.rng = np.random.RandomState(seed)

    def __call__(self, X, **kwargs):
        ref_ids = list(np.arange(X.shape[0]))
        del ref_ids[62]
        return ref_ids, self.rng.uniform(size=X.shape[0])


class DummyGaussEnviroment:
    def __init__(self):
        self.linear_model = SVC(C=1, kernel='linear')
        self.seed = 1337
        self.rng = np.random.RandomState(self.seed)
        self.prob_model = SVC(C=1, kernel='linear', probability=True, random_state=self.rng)
        self.strategy = DummyStrategy(self.seed)

        mean_1 = np.array([-2, 0])
        mean_2 = np.array([2, 0])
        cov = np.array([[1, 0], [0, 1]])
        X_1 = self.rng.multivariate_normal(mean_1, cov, 50)
        X_2 = self.rng.multivariate_normal(mean_2, cov, 50)

        X = np.vstack([X_1, X_2])
        y = np.ones(X.shape[0])
        y[51:] = -1

        p = self.rng.permutation(X.shape[0])
        y = y[p]

        self.X = X[p]
        self.y = mask_unknowns(y, np.arange(y.shape[0]))

        self.distance = self.normalized_euclidean_pairwise_distances(self.X)

    @staticmethod
    def cosine_dist_norm(a, b):
        return cosine(a, b) / 2.0

    @staticmethod
    def normalized_euclidean_pairwise_distances(X):
        """ d(i, j) = ||xi - xj||/max(||xk - xl||)"""
        D = pairwise_distances(X, metric="euclidean")
        return D / D.max()

    @staticmethod
    def qgb_score(ids, c, base_scores, distance_cache):
        all_pairs_x, all_pairs_y = zip(*product(ids, ids))
        # Product has n^2 while correct number is n * (n - 1) / 2.0
        all_pairs = (len(ids) * (len(ids) - 1))
        value = (1.-c)*base_scores[ids].mean() + \
                (c/all_pairs)*distance_cache[all_pairs_x, all_pairs_y].sum()
        return value


@pytest.fixture(scope='module')
def gauss_env():
    return DummyGaussEnviroment()


def test_numeric_qgb(gauss_env):
    dummy = gauss_env

    strategy = QuasiGreedyBatch(distance_cache=dummy.distance, c=0.3, base_strategy=DummyStrategy(dummy.seed))

    dummy.strategy.rng = np.random.RandomState(dummy.seed)

    ref_ids, score =  dummy.strategy(dummy.X)
    ref_score = dummy.qgb_score(ids=ref_ids, c=0.3, base_scores=score, distance_cache=dummy.distance)

    ids, qgb_score = strategy(X=dummy.X,
                              y=dummy.y,
                              model=None,
                              batch_size=dummy.X.shape[0] - 1,
                              rng=dummy.rng,
                              return_score=True)

    assert(set(ref_ids) == set(ids))
    assert(abs(ref_score - qgb_score) < 1e-10)















