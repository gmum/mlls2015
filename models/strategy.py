import numpy as np
from sklearn.ensemble import BaggingClassifier
import kaggle_ninja
from experiments.utils import jaccard_similarity_score_fast
from itertools import product
from misc.config import main_logger
from sklearn.svm import SVC


def strategy(X, y, current_model, batch_size, seed):
    """
    :param X:
    :param y:
    :param current_model: Currently used model for making predictions
    :param batch_size:
    :param seed:
    :return: Indexes of picked examples and normalized fitness (in unknown index space)
    """
    pass

def random_query(X, y, current_model, batch_size, seed):
    X = X[np.invert(y.known)]
    np.random.seed(seed)
    ids = np.random.randint(0, X.shape[0], size=batch_size)
    return y.unknown_ids[ids], np.random.uniform(0, 1, size=(X.shape[0],))


def uncertanity_sampling(X, y, current_model, batch_size, seed):
    X = X[np.invert(y.known)]
    if hasattr(current_model, "decision_function"):
        # Settles page 12
        fitness = np.abs(current_model.decision_function(X))
        ids =  np.argsort(fitness)[:batch_size]
    elif hasattr(current_model, "predict_proba"):
        p = current_model.predict_proba(X)
        # Settles page 13
        fitness = np.sum(p * np.log(p), axis=1)
        ids =  np.argsort(fitness)[:batch_size]
    return y.unknown_ids[ids], fitness/np.max(fitness)


def query_by_bagging(X, y, current_model, batch_size, seed, base_model=SVC(C=1, kernel='linear'), n_bags=5, method="KL"):
    """
    :param base_model: Model that will be  **fitted every iteration**
    :param n_bags: Number of bags on which train n_bags models
    :param method: 'entropy' or 'KL'
    :return:
    """
    assert method == 'entropy' or method == 'KL'
    eps = 0.0000001
    if method == 'KL':
        assert hasattr(base_model, 'predict_proba'), "Model with probability prediction needs to be passed to this strategy!"
    clfs = BaggingClassifier(base_model, n_estimators=n_bags, random_state=seed)
    clfs.fit(X[y.known], y[y.known])
    pc = clfs.predict_proba(X[np.invert(y.known)])
    # Settles page 17
    if method == 'entropy':
        pc += eps
        fitness = np.sum(pc * np.log(pc), axis=1)
        ids =  np.argsort(fitness)[:batch_size]
    elif method == 'KL':
        p = np.array([clf.predict_proba(X[np.invert(y.known)]) for clf in clfs.estimators_])
        # l = p / pc
        # print l.shape
        # print l.mean(axis=(0))
        # s = p * np.log(l)
        # print s.shape
        # m =  np.sum(s, axis=2)
        # print m.shape
        # id = np.mean(m, axis=0)
        # print id.shape
        # ids = np.argsort(id)[-batch_size:]
        fitness = np.mean(np.sum(p * np.log(p / pc), axis=2), axis=0)
        ids = np.argsort(fitness)[-batch_size:]

    return y.unknown_ids[ids], fitness/np.max(fitness)


def jaccard_dist(x1, x2):
    return 1 - jaccard_similarity_score_fast(x1, x2)

import scipy
def cosine_distance_normalized(a, b):
    # 1-cos(a,b) e [0,2] so /2
    return scipy.spatial.distance.cosine(a,b)/2.0

def quasi_greedy_batch(X, y, current_model, batch_size, seed,
                       c=0.3,
                       base_strategy=uncertanity_sampling,
                       dist=jaccard_dist):
    """
    :param c: Used for averaging (1-C)*example_fitness + C*normalized_distance_to_current_set
    :param base_strategy:
    :param dist:
    :return:
    """
    """ Example using it ;)
    cosine_distance_normalized(X[0], X[1])
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

    pick = quasi_greedy_batch(X, y, current_model=model, seed=777, batch_size=50, dist=cosine_distance_normalized, \
                              base_strategy=uncertanity_sampling)
    print pick
    # mean_picked_dist = np.abs(np.array([model.decision_function(X[i]) for i in pick])).mean()

    # not_picked = np.array([i for i in xrange(X.shape[0]) if i not in set(pick)])
    # mean_nota
    y_picked_dist = np.abs(model.decision_function(X[not_picked])).mean()

    # print mean_picked_dist
    # print mean_not_picked_dist

    # self.assertTrue(mean_picked_dist < mean_not_picked_dist)
    """
    X_unknown = X[y.unknown_ids]

    def score(idx):
        assert 0 <= base_scores[idx] <= 1
        dists = [dist(X_unknown[idx], X_unknown[j]) for j in picked]
        assert all( 0 <= d <= 1 for d in dists)
        # Counting number of pairs, ill-defined for 1 (this is the max operator in front)
        all_pairs = max(1,len(picked)*(len(picked) + 1)/2.0)
        d_score = 1.0/(all_pairs) * (sum(dists) + picked_dissimilarity)
        # TODO: it failed on me once. Not sure why
        if d_score >= 1:
            print d_score, all_pairs, sum(dists)
        assert 0 <= d_score <= 1
        return (1 - c) * base_scores[idx] + c * d_score

    # We start with an empty set
    picked = set([])
    picked_dissimilarity = 0 # Keep track of distances within the picked set
    known_labeles = y.known.sum()

    # Retrieve base scores that will be used throughout calculation
    _, base_scores = base_strategy(X=X, y=y, current_model=current_model, batch_size=batch_size, seed = seed)

    for i in range(batch_size):
        # Have we exhausted all of our options?
        if known_labeles + len(picked) == y.shape[0]:
            break

        candidates_scores = [score(i) for i in xrange(X_unknown.shape[0]) if i not in picked]
        picked.add(np.argmax(candidates_scores))
        picked_dissimilarity += sum(dist(X_unknown[np.argmax(candidates_scores)], X_unknown[j]) for j in picked)
        main_logger.debug("quasi greedy batch is picking %i th example from %i" % (len(picked), len(y.known) + batch_size))

    main_logger.debug("quasi greedy batch picked %i examples from %i set" % (len(picked), len(y.unknown_ids)))

    return [y.unknown_ids[i] for i in picked]



kaggle_ninja.register("query_by_bagging", query_by_bagging)
kaggle_ninja.register("uncertanity_sampling", uncertanity_sampling)
kaggle_ninja.register("random_query", random_query)
kaggle_ninja.register("quasi_greedy_batch", quasi_greedy_batch)

