import unittest
import sys
import numpy as np

sys.path.append("..")
import kaggle_ninja
kaggle_ninja.turn_off_cache()

from models.utils import ObstructedY
from models.strategy import *
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron

class DecisionDummy(object):

    def __init__(self):
        pass

    def fit(self, X,y):
        pass

    def predict(self, X):
        return np.zeros(X.shape[0])

    def decision_function(self, X):
        return (X[:,0] / np.max(X,axis=0)[0]).reshape(-1,)


class ProbDummy(object):

    def __init__(self):
        pass

    def fit(self, X,y):
        pass

    def predict(self, X):
        return np.zeros(X.shape[0])

    def predict_proba(self, X):
        return (X[:,0] / np.max(X,axis=0)[0]).reshape(-1,1)


class TestStrategies(unittest.TestCase):

    def setUp(self):
        self.decision_model = DecisionDummy()
        self.prob_model = ProbDummy()
        self.X = np.linspace(0.6, 1, 20).reshape(-1, 1)

        self.batch_size = 3
        self.rng = np.random.RandomState(666)

        self.y = np.ones(self.X.shape[0])
        self.y[np.random.randint(0, 20, 15)] = -1
        self.y = ObstructedY(self.y)
        self.y.query(np.random.randint(0, self.X.shape[0] / 2, self.batch_size))

    def test_random_sampling(self):
        pick, _ = random_query(self.X, self.y, self.decision_model, self.batch_size, self.rng)

        self.assertTrue(len(pick) == self.batch_size)
        self.assertTrue(len(pick) == len(np.unique(pick)))
        self.assertTrue(all(i in self.y.unknown_ids for i in pick))

    def test_speedup_greedy(self):
        X = np.random.uniform(-1, 1, size=(1000, 2))
        Y = np.ones(X.shape[0])
        negative_examples = np.where(X[:, 0] < 0)
        Y[negative_examples] = -1
        Y_obstructed = ObstructedY(Y)
        Y_obstructed.query(range(100))
        m = Perceptron(alpha=0, n_iter=100).fit(X, Y)

        dist = construct_normalized_euc(X)

        D = pairwise_distances(X, metric=dist)

        r1 = quasi_greedy_batch(X, Y_obstructed, m, rng=None, batch_size=20, D=D)
        r2 = quasi_greedy_batch_slow(X, Y_obstructed, m, rng=None, batch_size=20, dist=dist, D=D)

        self.assertTrue(np.array_equal(r1[0], r2[0]))
        self.assertAlmostEqual(r1[1], r2[1])

    def test_uncertainty_sampling(self):
        decision_pick, _ = uncertainty_sampling(self.X, self.y, self.decision_model, self.batch_size, self.rng)
        prob_pick, _ = uncertainty_sampling(self.X, self.y, self.prob_model, self.batch_size, self.rng)

        self.assertTrue(all(decision_pick == prob_pick))

        self.assertTrue(np.array_equal(decision_pick, [i for i in xrange(self.batch_size)]))
        self.assertTrue(np.array_equal(prob_pick, [i for i in xrange(self.batch_size)]))

        decision_pick, _ = uncertainty_sampling(self.X[::-1], self.y, self.decision_model, self.batch_size, self.rng)
        prob_pick, _ = uncertainty_sampling(self.X[::-1], self.y, self.prob_model, self.batch_size, self.rng)

        # print decision_pick

        self.assertTrue(all(decision_pick == prob_pick))
        self.assertTrue(all(decision_pick == [self.X.shape[0] - i for i in xrange(1, self.batch_size + 1)]))
        self.assertTrue(all(prob_pick == [self.X.shape[0] - i for i in xrange(1, self.batch_size + 1)]))

    def test_qbc(self):
        mean_1 = np.array([-2, 0])
        mean_2 = np.array([2, 0])
        cov = np.array([[1, 0], [0, 1]])
        X_1 = np.random.multivariate_normal(mean_1, cov, 100)
        X_2 = np.random.multivariate_normal(mean_2, cov, 200)
        X = np.vstack([X_1, X_2])
        y = np.ones(X.shape[0])
        y[101:] = -1

        # shuffle data
        p = np.random.permutation(X.shape[0])
        X = X[p]
        y = y[p]

        y = ObstructedY(y)
        y.query(np.random.randint(0, X.shape[0], 50))

        model = SVC(C=1, kernel='linear')
        model.fit(X[y.known], y[y.known])

        pick, _ = query_by_bagging(X, y, current_model=None, base_model=model,
                                   batch_size=50, rng=self.rng, n_bags=5, method='entropy')
        mean_picked_dist = np.abs(model.decision_function(X[pick])).mean()

        not_picked = [i for i in xrange(X.shape[0]) if i not in set(pick)]
        mean_unpicked_dist = np.abs(model.decision_function(X[not_picked])).mean()

        self.assertTrue(mean_picked_dist < mean_unpicked_dist)

    def test_greedy_distance(self):
            mean_1 = np.array([-2, 0])
            mean_2 = np.array([2, 0])
            cov = np.array([[1, 0], [0, 1]])
            X_1 = np.random.multivariate_normal(mean_1, cov, 100)
            X_2 = np.random.multivariate_normal(mean_2, cov, 200)
            X = np.vstack([X_1, X_2])
            y = np.ones(X.shape[0])
            y[101:] = -1

            # shuffle data
            p = np.random.permutation(X.shape[0])
            X = X[p]
            y = y[p]

            y = ObstructedY(y)
            y.query(np.random.randint(0, X.shape[0], 50))

            model = SVC(C=1, kernel='linear')
            model.fit(X[y.known], y[y.known])

            picked, _ = quasi_greedy_batch_slow(X, y, current_model=model, c=1.0, rng=self.rng, batch_size=50, dist='cosine_distance_normalized')

            mean_picked_dist = np.mean([cosine_distance_normalized(X[x1], X[x2]) for x1, x2 in product(picked, picked)])

            unc_pick, _ = uncertainty_sampling(X, y, model, rng=self.rng, batch_size=50)
            mean_unc_picked_dist = np.mean([cosine_distance_normalized(X[x1], X[x2]) for x1, x2 in product(unc_pick, unc_pick)])

            self.assertTrue(mean_picked_dist > mean_unc_picked_dist)

    def test_greedy_unc(self):
            mean_1 = np.array([-2, 0])
            mean_2 = np.array([2, 0])
            cov = np.array([[1, 0], [0, 1]])
            X_1 = np.random.multivariate_normal(mean_1, cov, 100)
            X_2 = np.random.multivariate_normal(mean_2, cov, 200)
            X = np.vstack([X_1, X_2])
            y = np.ones(X.shape[0])
            y[101:] = -1

            # shuffle data
            p = np.random.permutation(X.shape[0])
            X = X[p]
            y = y[p]

            y = ObstructedY(y)
            y.query(np.random.randint(0, X.shape[0], 50))

            model = SVC(C=1, kernel='linear')
            model.fit(X[y.known], y[y.known])

            picked, _ = quasi_greedy_batch_slow(X, y, current_model=model, c=0.0, rng=self.rng, batch_size=10, dist='cosine_distance_normalized', \
                                      base_strategy='uncertainty_sampling')
            unc_pick, _ = uncertainty_sampling(X, y, model, batch_size=10, rng=self.rng)

            self.assertTrue(set(picked) == set(unc_pick))

suite = unittest.TestLoader().loadTestsFromTestCase(TestStrategies)
print unittest.TextTestRunner(verbosity=3).run(suite)