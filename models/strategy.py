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


def uncertainty_sampling(X, y, current_model, batch_size, seed):
    X = X[np.invert(y.known)]
    if hasattr(current_model, "decision_function"):
        # Settles page 12
        fitness = np.abs(np.ravel(current_model.decision_function(X)))
        ids = np.argsort(fitness)[:batch_size]
    elif hasattr(current_model, "predict_proba"):
        p = current_model.predict_proba(X)
        # Settles page 13
        fitness = np.sum(p * np.log(p), axis=1).ravel()
        ids = np.argsort(fitness)[:batch_size]

    fitness = np.abs(fitness)
    max_fit = np.max(fitness)
    return y.unknown_ids[ids], (max_fit - fitness)/max_fit


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

from scipy.spatial.distance import euclidean
from sklearn.metrics import pairwise_distances
def construct_normalized_euc(X):

    D = pairwise_distances(X, metric="euclidean")
    max_dist = D.max() + 1e-2

    def normalized_euc(x1, x2):
        return euclidean(x1, x2)/max_dist

    return normalized_euc

def exp_euc(x1, x2):
    return 1 - np.exp(-euclidean(x1, x2))

import scipy
def cosine_distance_normalized(a, b):
    # 1-cos(a,b) e [0,2] so /2
    return scipy.spatial.distance.cosine(a,b)/2.0

def quasi_greedy_batch(X, y, current_model, batch_size, seed=777,
                       c=0.3,
                       base_strategy='uncertanity_sampling',
                       dist='jaccard_dist', D=None):
    """
    :param c: Used for averaging (1-C)*example_fitness + C*normalized_distance_to_current_set
    :param base_strategy:
    :param dist:
    :return:
    """
    X_unknown = X[y.unknown_ids]



    if isinstance(dist, str):
        dist = globals()[dist]
    elif hasattr(dist, '__call__'):
        pass
    else:
        raise TypeError("dist must be a function or string, got %s" % type(dist))


    if isinstance(base_strategy, str):
        base_strategy = globals()[base_strategy]

    if D is None:
        D = pairwise_distances(X_unknown, metric=dist)
    else:
        D = D[y.unknown_ids, :][:, y.unknown_ids]

    if isinstance(base_strategy, str):
        base_strategy = globals()[base_strategy]
    elif hasattr(dist, '__call__'):
        pass
    else:
        raise TypeError("base_strategy must be a function or string, got %s" % type(base_strategy))

    def score(idx):
        assert 0 <= base_scores[idx] <= 1, "got score: %f" % base_scores[idx]
        #dists = [dist(X_unknown[idx], X_unknown[j]) for j in picked]
        dists = [D[idx, j] for j in picked]
        assert all( 0 <= d <= 1 for d in dists)
        # Counting number of pairs, ill-defined for 1 (this is the max operator in front)
        all_pairs = max(1,len(picked)*(len(picked) + 1)/2.0)
        d_score = 1.0/(all_pairs) * (sum(dists) + picked_dissimilarity)

        assert 0 <= d_score <= 1, "score calculated d_score: %f" % d_score
        # TODO: improve numerical stability
        return (1 - c) * base_scores[idx]/float(len(picked)) + c * d_score


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

        candidates_scores = [(score(i),i) for i in xrange(X_unknown.shape[0]) if i not in picked]
        new_index = max(candidates_scores)[1]
        picked_dissimilarity += sum(dist(X_unknown[new_index], X_unknown[j]) for j in picked)
        picked.add(new_index)
        main_logger.debug("quasi greedy batch is picking %i th example from %i" % (len(picked), len(y.known) + batch_size))

    main_logger.debug("quasi greedy batch picked %i examples from %i set" % (len(picked), len(y.unknown_ids)))
    return [y.unknown_ids[i] for i in picked], \
           (1 - c)*base_scores[np.array(list(picked))].mean() + c*(1.0/max(1,len(picked)*(len(picked) - 1)/2.0))*picked_dissimilarity


kaggle_ninja.register("query_by_bagging", query_by_bagging)
kaggle_ninja.register("uncertainty_sampling", uncertainty_sampling)
kaggle_ninja.register("random_query", random_query)
kaggle_ninja.register("quasi_greedy_batch", quasi_greedy_batch)

