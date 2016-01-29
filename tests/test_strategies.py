# -*- coding: utf-8 -*-

"""
 Test class for all strategy methods
"""

import pytest

import numpy as np

from alpy2.strategy import UncertaintySampling, QueryByBagging, QuasiGreedyBatch, CSJSampling
from alpy2.utils import mask_unknowns, unmasked_indices, masked_indices
from models.balanced_models import RandomProjector

from sklearn.svm import SVC
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from scipy.sparse import csc_matrix

from itertools import product


class DecisionDummy(object):
    """ Decision result faker """

    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    @staticmethod
    def predict(X):
        return np.zeros(X.shape[0])

    @staticmethod
    def decision_function(X):
        return (X[:, 0] / np.max(X, axis=0)[0]).reshape(-1, )


class ProbDummy(object):
    """ Probability decision result faker """

    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    @staticmethod
    def predict(X):
        return np.zeros(X.shape[0])

    @staticmethod
    def predict_proba(X):
        return (X[:, 0] / np.max(X, axis=0)[0]).reshape(-1, 1)


class DummyEnvironment:
    def __init__(self):
        self.decision_model = DecisionDummy()
        self.prob_model = ProbDummy()
        self.X = np.linspace(0.6, 1, 20).reshape(-1, 1)

        self.batch_size = 3
        self.rng = np.random.RandomState(666)

        self.y = np.ones(self.X.shape[0])
        self.y = mask_unknowns(self.y, [0, 1, 2, 17, 18, 19])


class DummyGaussEnviroment:
    def __init__(self):
        self.linear_model = SVC(C=1, kernel='linear')
        self.rng = np.random.RandomState(666)
        self.prob_model = SVC(C=1, kernel='linear', probability=True, random_state=self.rng)


        mean_1 = np.array([-2, 0])
        mean_2 = np.array([2, 0])
        cov = np.array([[1, 0], [0, 1]])
        X_1 = self.rng.multivariate_normal(mean_1, cov, 100)
        X_2 = self.rng.multivariate_normal(mean_2, cov, 200)

        X = np.vstack([X_1, X_2])
        y = np.ones(X.shape[0])
        y[101:] = -1

        p = self.rng.permutation(X.shape[0])
        y = y[p]

        self.X = X[p]
        self.y = mask_unknowns(y, self.rng.choice(X.shape[0], size=self.X.shape[0] - 50, replace=False))

        self.distance = pairwise_distances(self.X, metric=self.cosine_dist_norm)

    def cosine_dist_norm(self, a, b):
            return cosine(a, b) / 2.0


class ClustersEnviroment:
    def __init__(self):
        self.linear_model = SVC(C=1, kernel='linear')
        self.seed = 666
        self.rng = np.random.RandomState(666)
        self.prob_model = SVC(C=1, kernel='linear', probability=True, random_state=self.rng)

        mean_1 = np.array([4, 4])
        mean_2 = np.array([4, -4])
        mean_3 = np.array([-4, 4])
        mean_4 = np.array([-4, -4])

        rng = np.random.RandomState(666)

        cov_1 = np.array([[1, 0], [0, 1]])
        cov_2 = np.array([[0.2, 0], [0, 0.2]])

        X_1 = rng.multivariate_normal(mean_1, cov_1, 200)
        X_2 = rng.multivariate_normal(mean_2, cov_2, 20)
        X_3 = rng.multivariate_normal(mean_3, cov_2, 20)
        X_4 = rng.multivariate_normal(mean_4, cov_1, 200)

        self.cluster_ids = np.zeros(440)
        self.cluster_ids[200:220] = 1
        self.cluster_ids[220:240] = 2
        self.cluster_ids[240:] = 3

        y = np.zeros(440)
        y[220:] = 1

        self.X = csc_matrix(np.vstack([X_1, X_2, X_3, X_4]))
        self.y = mask_unknowns(y, self.rng.choice(self.X.shape[0], size=self.X.shape[0] - 20, replace=False))

        random_projector = RandomProjector(rng=self.rng)
        random_projector.fit(self.X)
        self.projection = random_projector.project(self.X)

        self.distance = pairwise_distances(self.X, metric='euclidean')


@pytest.fixture(scope='module')
def dummy_env():
    return DummyEnvironment()


@pytest.fixture(scope='module')
def gauss_env():
    return DummyGaussEnviroment()


@pytest.fixture(scope='module')
def clusters_env():
    return ClustersEnviroment()


def test_csj_clusers_pick(clusters_env):
    dummy = clusters_env

    csj = CSJSampling(projection=dummy.projection, c=1., k=4, distance_cache=dummy.distance)
    qgb = QuasiGreedyBatch(distance_cache=dummy.distance, c=1.)

    known_ids = unmasked_indices(dummy.y)
    model = dummy.linear_model.fit(dummy.X[known_ids], dummy.y[known_ids])

    batch_size = 8
    qgb_pick = qgb(dummy.X, dummy.y, model, batch_size=batch_size, rng=np.random.RandomState(dummy.seed))
    csj_pick = csj(dummy.X, dummy.y, model, batch_size=batch_size, rng=np.random.RandomState(dummy.seed))

    csj_cluster_picks = {id: 0 for id in range(4)}
    qgb_cluster_picks = {id: 0 for id in range(4)}

    for qgb_pick, csj_pick in zip(qgb_pick, csj_pick):
        csj_cluster_picks[dummy.cluster_ids[csj_pick]] += 1
        qgb_cluster_picks[dummy.cluster_ids[qgb_pick]] += 1


    assert(csj_cluster_picks.values() == [2,2,2,2])
    assert(qgb_cluster_picks.values() != csj_cluster_picks.values())

def test_csj_too_small_clusters(clusters_env):
    dummy = clusters_env

    csj = CSJSampling(projection=dummy.projection, c=1., k=4, distance_cache=dummy.distance)
    known_ids = unmasked_indices(dummy.y)
    model = dummy.linear_model.fit(dummy.X[known_ids], dummy.y[known_ids])

    batch_size = 40
    csj_pick = csj(dummy.X, dummy.y, model, batch_size=batch_size, rng=np.random.RandomState(dummy.seed))

    assert(len(csj_pick) == batch_size)

def test_uncertainty_sampling(dummy_env):
    dummy = dummy_env
    strategy = UncertaintySampling()

    decision_pick = strategy(dummy.X, dummy.y, dummy.decision_model, dummy.batch_size)
    prob_pick = strategy(dummy.X, dummy.y, dummy.prob_model, dummy.batch_size)

    assert (all(decision_pick == prob_pick))
    assert (np.array_equal(decision_pick, list(range(dummy.batch_size))))
    assert (np.array_equal(prob_pick, list(range(dummy.batch_size))))

    decision_pick = strategy(dummy.X[::-1], dummy.y, dummy.decision_model, dummy.batch_size)
    prob_pick = strategy(dummy.X[::-1], dummy.y, dummy.prob_model, dummy.batch_size)

    assert (all(decision_pick == prob_pick))
    assert (all(decision_pick == [dummy.X.shape[0] - i for i in range(1, dummy.batch_size + 1)]))
    assert (all(prob_pick == [dummy.X.shape[0] - i for i in range(1, dummy.batch_size + 1)]))


def test_qbb(gauss_env):
    dummy = gauss_env

    known_ids = unmasked_indices(dummy.y)

    dummy.linear_model.fit(dummy.X[known_ids], dummy.y[known_ids])
    dummy.prob_model.fit(dummy.X[known_ids], dummy.y[known_ids])

    ent_qbb = QueryByBagging(method='entropy')
    kl_qbb = QueryByBagging(method='KL')

    ent_pick = ent_qbb(dummy.X, dummy.y, model=dummy.linear_model, batch_size=10, rng=np.random.RandomState(42))
    mean_picked_dist_ent = np.abs(dummy.linear_model.decision_function(dummy.X[ent_pick])).mean()

    not_picked_ent = [i for i in range(dummy.X.shape[0]) if i not in set(ent_pick)]
    mean_unpicked_dist_ent = np.abs(dummy.linear_model.decision_function(dummy.X[not_picked_ent])).mean()

    assert(mean_picked_dist_ent < mean_unpicked_dist_ent)

    kl_pick = kl_qbb(dummy.X, dummy.y, model=dummy.prob_model, batch_size=10, rng=np.random.RandomState(42))
    mean_picked_dist_kl = np.abs(dummy.prob_model.decision_function(dummy.X[kl_pick])).mean()

    not_picked_kl = [i for i in range(dummy.X.shape[0]) if i not in set(kl_pick)]
    mean_unpicked_dist_kl = np.abs(dummy.linear_model.decision_function(dummy.X[not_picked_kl])).mean()

    assert(mean_picked_dist_kl < mean_unpicked_dist_kl)

def test_quasi_greedy_distance(gauss_env):

    dummy = gauss_env
    known_ids = unmasked_indices(dummy.y)

    dummy.linear_model.fit(dummy.X[known_ids], dummy.y[known_ids])

    qgb = QuasiGreedyBatch(distance_cache=dummy.distance, c=1.)
    unc = UncertaintySampling()

    picked = qgb(dummy.X, dummy.y, model=dummy.linear_model, rng=dummy.rng, batch_size=50)
    mean_picked_dist = np.mean([dummy.cosine_dist_norm(dummy.X[x1], dummy.X[x2]) for x1, x2 in product(picked, picked)])

    unc_pick = unc(dummy.X, dummy.y, dummy.linear_model, rng=dummy.rng, batch_size=50)
    mean_unc_picked_dist = np.mean([dummy.cosine_dist_norm(dummy.X[x1], dummy.X[x2]) for x1, x2 in product(unc_pick, unc_pick)])

    assert(mean_picked_dist > mean_unc_picked_dist)


def test_quasi_greedy_is_analogous_to_unc(gauss_env):

    dummy = gauss_env
    known_ids = unmasked_indices(dummy.y)

    dummy.linear_model.fit(dummy.X[known_ids], dummy.y[known_ids])

    qgb = QuasiGreedyBatch(distance_cache=dummy.distance, c=0.)
    unc = UncertaintySampling()

    picked = qgb(dummy.X, dummy.y, model=dummy.linear_model, rng=dummy.rng, batch_size=50)
    unc_pick = unc(dummy.X, dummy.y, dummy.linear_model, rng=dummy.rng, batch_size=50)

    assert(set(picked) == set(unc_pick))


def test_n_tries_quasi_greedy(gauss_env):

    dummy = gauss_env
    known_ids = unmasked_indices(dummy.y)

    dummy.linear_model.fit(dummy.X[known_ids], dummy.y[known_ids])

    n_qgb = QuasiGreedyBatch(distance_cache=dummy.distance, n_tries=10)
    qgb = QuasiGreedyBatch(distance_cache=dummy.distance,)

    n_picked, n_score = n_qgb(dummy.X, dummy.y, dummy.linear_model, rng=dummy.rng, batch_size=50, return_score=True)
    picked, score = qgb(dummy.X, dummy.y, dummy.linear_model, rng=dummy.rng, batch_size=50,  return_score=True)

    assert(len(picked) == len(n_picked))

    if set(n_picked) == set(n_picked):
        assert(np.abs(n_score - score) < 1e-6)
    else:
        assert(n_score >= score)




