import unittest
import sys
import numpy as np

sys.path.append("..")
import kaggle_ninja
kaggle_ninja.turn_off_cache()

from models.utils import ObstructedY
from models.strategy import *
from sklearn.svm import SVC


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
        self.seed = 666

        np.random.seed(self.seed)

        self.y = np.ones(self.X.shape[0])
        self.y[np.random.randint(0, 20, 15)] = -1
        self.y = ObstructedY(self.y)
        self.y.query(np.random.randint(0, self.X.shape[0] / 2, self.batch_size))

    def test_random_sampling(self):
        pick, _ = random_query(self.X, self.y, self.decision_model, self.batch_size, self.seed)

        self.assertTrue(len(pick) == self.batch_size)
        self.assertTrue(len(pick) == len(np.unique(pick)))
        self.assertTrue(all(i in self.y.unknown_ids for i in pick))

    def test_uncertainty_sampling(self):
        decision_pick, _ = uncertanity_sampling(self.X, self.y, self.decision_model, self.batch_size, self.seed)
        prob_pick, _ = uncertanity_sampling(self.X, self.y, self.prob_model, self.batch_size, self.seed)

        self.assertTrue(all(decision_pick == prob_pick))
        self.assertTrue(all(decision_pick == [i for i in xrange(self.batch_size)]))
        self.assertTrue(all(prob_pick == [i for i in xrange(self.batch_size)]))

        decision_pick, _ = uncertanity_sampling(self.X[::-1], self.y, self.decision_model, self.batch_size, self.seed)
        prob_pick, _ = uncertanity_sampling(self.X[::-1], self.y, self.prob_model, self.batch_size, self.seed)

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

        pick, _ = query_by_bagging(X, y, current_model=None, base_model=model, batch_size=50, seed=self.seed, n_bags=5, method='entropy')
        mean_picked_dist = np.abs(model.decision_function(X[pick])).mean()

        not_picked = [i for i in xrange(X.shape[0]) if i not in set(pick)]
        mean_unpicked_dist = np.abs(model.decision_function(X[not_picked])).mean()

        self.assertTrue(mean_picked_dist < mean_unpicked_dist)

    def test_greedy(self):
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

            picked, _ = quasi_greedy_batch(X, y, current_model=model, seed=666, batch_size=50, dist='cosine_distance_normalized', \
                                      base_strategy='uncertanity_sampling')

            mean_picked_dist = np.sum([cosine_distance_normalized(X[x1], X[x2]) for x1, x2 in product(picked, picked)])

            unc_pick, _ = uncertanity_sampling(self.X, self.y, self.prob_model, self.batch_size, self.seed)
            mean_unc_picked_dist = np.sum([cosine_distance_normalized(X[x1], X[x2]) for x1, x2 in product(unc_pick, unc_pick)])

            print mean_picked_dist
            print mean_unc_picked_dist

            self.assertTrue(mean_picked_dist > mean_unc_picked_dist)

suite = unittest.TestLoader().loadTestsFromTestCase(TestStrategies)
print unittest.TextTestRunner(verbosity=3).run(suite)