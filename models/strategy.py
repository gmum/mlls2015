import numpy as np
from sklearn.ensemble import BaggingClassifier
import kaggle_ninja
from experiments.utils import jaccard_similarity_score_fast
from itertools import product
from misc.config import main_logger
from sklearn.svm import SVC
from kaggle_ninja import *
from scipy.spatial.distance import euclidean
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import Perceptron
from experiments.utils import wac_score
import hashlib
from collections import defaultdict
import math
from sklearn.cluster import KMeans




def strategy(X, y, current_model, batch_size, rng):
    """
    :param X:
    :param y:
    :param current_model: Currently used model for making predictions
    :param batch_size:
    :param rng:
    :param D: matrix of pairwise distances
    :return: Indexes of picked examples and normalized fitness (in unknown index space)
    """
    pass

def czarnecki(X, y, current_model, batch_size, rng, D=None):
    # Assumes X is an array [X, X_proj]

    # Cluster and get uncertanity
    cluster_ids = KMeans(n_clusters=batch_size, random_state=rng).fit_predict(X[1][y.unknown_ids])
    _, base_scores = uncertainty_sampling(X=X[0], y=y, rng=rng, current_model=current_model, batch_size=batch_size)

    assert len(cluster_ids) == len(base_scores), len(np.unique(cluster_ids)) == batch_size

    examples_by_cluster = {cluster_id_key:
                           zip(base_scores[cluster_ids == cluster_id_key], np.where(cluster_ids == cluster_id_key)[0])
                      for cluster_id_key in np.unique(cluster_ids)
                      }

    for k in examples_by_cluster:
        for value, ex_id in examples_by_cluster[k]:
            assert cluster_ids[ex_id] == k

    picked = []
    for cl_id in range(batch_size):
        picked.append(max(examples_by_cluster[cl_id])[1])

    return picked, np.inf

def random_query(X, y, current_model, batch_size, rng, D=None):
    X = X[np.invert(y.known)]
    ids = rng.randint(0, X.shape[0], size=batch_size)
    return y.unknown_ids[ids], np.random.uniform(0, 1, size=(X.shape[0],))


def uncertainty_sampling(X, y, current_model, batch_size, rng, D=None):
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
    else:
        raise AttributeError("Model with either decision_function or predict_proba method")

    fitness = np.abs(fitness)
    max_fit = np.max(fitness)
    return y.unknown_ids[ids], (max_fit - fitness)/max_fit


def query_by_bagging(X, y, current_model, batch_size, rng, base_model=SVC(C=1, kernel='linear'), n_bags=5, method="KL", D=None):
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
    clfs = BaggingClassifier(base_model, n_estimators=n_bags, random_state=rng)
    clfs.fit(X[y.known], y[y.known])
    pc = clfs.predict_proba(X[np.invert(y.known)])
    # Settles page 17
    if method == 'entropy':
        pc += eps
        fitness = np.sum(pc * np.log(pc), axis=1)
        ids =  np.argsort(fitness)[:batch_size]
    elif method == 'KL':
        p = np.array([clf.predict_proba(X[np.invert(y.known)]) for clf in clfs.estimators_])
        fitness = np.mean(np.sum(p * np.log(p / pc), axis=2), axis=0)
        ids = np.argsort(fitness)[-batch_size:]

    return y.unknown_ids[ids], fitness/np.max(fitness)


def jaccard_dist(x1, x2):
    return 1 - jaccard_similarity_score_fast(x1, x2)


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


def quasi_greedy_batch(X, y, current_model, batch_size, rng,
                       c=0.3, sample_first=False,
                       base_strategy='uncertainty_sampling',
                       D=None, warmstart=None):
    """
    :param c: Used for averaging (1-C)*example_fitness + C*normalized_distance_to_current_set
    :param base_strategy:
    :param D: Matrix of all distances (pairwise_distances(X, metric=dist))
    :return:
    """
    X_unknown = X[y.unknown_ids]


    if isinstance(base_strategy, str):
        base_strategy = find_obj(base_strategy)

    if D is None:
        raise ValueError("Please pass D matrix of pairwise distances")
        # D = pairwise_distances(X_unknown, metric=dist)
    else:
        D = D[y.unknown_ids, :][:, y.unknown_ids]

    if isinstance(base_strategy, str):
        base_strategy = globals()[base_strategy]
    elif hasattr(base_strategy, '__call__'):
        pass
    else:
        raise TypeError("base_strategy must be a function or string, got %s" % type(base_strategy))

    # D_prim keeps distance from all examples to picked set
    D_prim = np.zeros(shape=(X_unknown.shape[0], ))

    # Retrieve base scores that will be used throughout calculation
    _, base_scores = base_strategy(X=X, y=y, current_model=current_model, batch_size=batch_size, rng=rng)

    # We start with an empty set
    if sample_first:
        p = base_scores / np.sum(base_scores)
        start_point = rng.choice(X_unknown.shape[0], p=p)
        picked_sequence = [start_point]
        picked = set(picked_sequence)
        D_prim[:] = D[:, picked_sequence].sum(axis=1)
    elif warmstart:
        to_unknown_id = {v:k for k,v in enumerate(y.unknown_ids)}

        picked_sequence = [to_unknown_id[i] for i in warmstart]
        picked = set(picked_sequence)
        D_prim[:] = D[:, picked_sequence].sum(axis=1)
    else:
        picked = set([])
        picked_sequence = []

    known_labeles = y.known.sum()

    candidates = [i for i in range(X_unknown.shape[0]) if i not in picked]
    while len(picked) < batch_size:
        # Have we exhausted all of our options?
        if known_labeles + len(picked) == y.shape[0]:
            break

        all_pairs = max(1,len(picked)*(len(picked) + 1)/2.0)
        candidates_scores = c*D_prim[candidates]/all_pairs + (1-c)*base_scores[candidates]/max(1, len(picked))
        candidates_index = np.argmax(candidates_scores.reshape(-1))
        new_index = candidates[candidates_index]
        picked.add(new_index)
        picked_sequence.append(new_index)
        del candidates[candidates_index]
        D_prim += D[:, new_index]

    picked_dissimilarity = D_prim[picked_sequence].sum()/2.0
    return [y.unknown_ids[i] for i in picked_sequence], \
           (1 - c)*base_scores[picked_sequence].mean() + c*(1.0/max(1,len(picked)*(len(picked) - 1)/2.0))*picked_dissimilarity




def hit_and_run(X, Y, w0, rng, N=100, T=10, sub_sample_size=100, eps=0.5):
    """
    @param w0 - starting hypothesis point. Should almost separate !
    @param X - known samples
    @param Y - known samples labels
    @param N hypothesis of points wanted
    @param T number of mixing iterations
    @param sub_sample_size how many samples of hypothesis take on the ray.
    @param eps noise level. Note that i should be pretty big
    """

    def _solve_quadratic_safe(a,b,c):
        # calculate the discriminant
        d = (b**2) - (4*a*c)

        if d <= 0:
            return -1, 1 # Should be relatively rare and is ok to return this

        # find two solutions
        sol1 = (-b-math.sqrt(d))/(2*a)
        sol2 = (-b+math.sqrt(d))/(2*a)

        return sol1, sol2

    # In regular hit and run we do bound calculation in every turn
    # Here it is approximated becuase we are always inside a sphere
    MIN_DELTA = -2
    MAX_DELTA = 2

    d = X.shape[1]
    w = w0



    missed_w0 = float((np.abs(np.sign(w0.dot(X.T)) - Y).sum(axis=1)/2.0)[0])

    alpha = (eps / (1.0 - eps))
    out = []

    for i in range(N*T):
        theta = rng.uniform(-1,1,size=(1,d))
        theta = theta/np.linalg.norm(theta)

        # Find max and min ro (quadratic equation, trust me)

        a = 1
        b = (2*w[0,0]*theta[0,0] + 2*w[0,1]*theta[0,1])/(theta[0,0]**2 + theta[0,1]**2)
        c =(w[0,0]**2 + w[0,1]**2 - 1)/(theta[0,0]**2+theta[0,1]**2)
        ro_min, ro_max = sorted(_solve_quadratic_safe(a,b,c))

        # Simple way to make sure that w is always inside circle after step (S^d \intersection L)
        L = np.vstack([w + delta * theta for delta in np.linspace(ro_min, ro_max, sub_sample_size)])

        missed_for_sample = (np.abs(np.sign(L.dot(X.T)) - Y)).sum(axis=1)/2.0

        # WARNING: important change that assumes hypothesis picked as separating has 0 missed examples
        weights = np.array([alpha**(max(0, float(m) - missed_w0)) for m in missed_for_sample])
        #TODO: filter very weak probabilities or do some min?
        w = L[rng.choice(range(len(L)), 1, p=weights/sum(weights))]

        if i>0 and i%T == 0:
            out.append(w)

    return np.vstack(out)




def chen_krause(X, y, current_model, rng, batch_size, D=None, N=600, T=5, eps=0.3):
    """
    @param current_model Not used, but kept for consistency with interface of strategy
    @param N hypothesis of points wanted
    @param T number of mixing iterations
    @param sub_sample_size how many samples of hypothesis take on the ray.
    @param eps noise level. Note that i should be pretty big
    """
    Y = y


    X_known = X[Y.known_ids]
    Y_known = Y[Y.known_ids]
    X_unknown = X[Y.unknown_ids]

    # Start with point close to hypothesis ! Very important
    #TODO: logistic regression
    m = Perceptron(alpha=0, n_iter=100).fit(X_known, Y_known)
    w0 = m.coef_


    # # Also check 0 mean features
    # if X_known.shape[0] > 100:
    #     means = X_known.mean(axis=0).reshape(-1)
    #     assert all(abs(m) < 0.1 for m in means), "hit_and_run assumes mean 0 features"
    #
    # if wac_score(m.predict(X_known), Y_known) < 0.9:
    #     main_logger.warning("hit and run in this formulation works only \
    #             for almost separable case "+str(wac_score(m.predict(X_known), Y_known)))



    # Now it is possible it will work - proceed

    # Sample hypotheses
    H = hit_and_run(X_known, Y_known, rng=rng, w0=w0, N=N, T=T, eps=eps)
    k=0
    picked = []

    preds_known = np.sign(np.dot(H, X_known.T))
    preds_unknown = np.sign(np.dot(H, X_unknown.T))

    # For hypothesis dict
    def key(a):
        return hashlib.sha1(a.view(np.uint8)).hexdigest()

    # Construct k-batch
    for k in range(batch_size):
        # 1. Construct hypothesis codes
        preds_picked = (preds_unknown[:, picked]).copy()
        if len(picked):
            h_codes = [key(h_code) for h_code in np.hstack([preds_known, preds_picked])]
        else:
            h_codes = [key(h_code) for h_code in preds_known]
        counted_same = []

        for i in range(X_unknown.shape[0]):
            if i not in picked:
                counts = defaultdict(int)
                # 2. Count hypothesis by adding to dict
                for j in range(H.shape[0]):
                    hypothesis_key = h_codes[j] + str(preds_unknown[j,i])
                    counts[hypothesis_key] += 1

                # 3. Count removing 1 to obtain needed value
                counted_same.append(sum(counts.values()) - len(counts))
            else:
                counted_same.append(np.inf)


        picked.append(np.argmin(counted_same))
    return [Y.unknown_ids[p] for p in picked], H

def quasi_greedy_batch_slow(X, y, current_model, batch_size, rng,
                       c=0.3,
                       base_strategy='uncertainty_sampling',
                       dist='jaccard_dist', D=None, warmstart=None):
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
        base_strategy = find_obj(base_strategy)


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
        return (1 - c) * base_scores[idx]/max(1,float(len(picked))) + c * d_score


    # We start with an empty set
    if warmstart:
        to_unknown_id = {v:k for k,v in enumerate(y.unknown_ids)}

        picked_sequence = [to_unknown_id[i] for i in warmstart]
        picked = set(picked_sequence)
        picked_dissimilarity = (D[list(picked),list(picked)]).sum()/2.0
        for i in picked:
            for j in picked:
                if i > j:
                    picked_dissimilarity += D[i,j]
    else:
        picked = set([])
        picked_sequence = []
        picked_dissimilarity = 0 # Keep track of distances within the picked set
    known_labeles = y.known.sum()

    # Retrieve base scores that will be used throughout calculation
    _, base_scores = base_strategy(X=X, y=y, current_model=current_model, batch_size=batch_size, rng=rng)

    while len(picked) < batch_size:
        # Have we exhausted all of our options?
        if known_labeles + len(picked) == y.shape[0]:
            break
        candidates_scores = [(score(i),i) for i in xrange(X_unknown.shape[0]) if i not in picked]
        new_index = max(candidates_scores)[1]
        picked_dissimilarity += sum(D[new_index, j] for j in picked)
        picked.add(new_index)
        picked_sequence.append(new_index)

    return [y.unknown_ids[i] for i in picked_sequence], \
           (1 - c)*base_scores[np.array(list(picked))].mean() + c*(1.0/max(1,len(picked)*(len(picked) - 1)/2.0))*picked_dissimilarity


def multiple_pick_best(X, y,
                       current_model,
                       batch_size,
                       rng,
                       k=20,
                       D=None,
                       c=1.0):

    results = []
    for i in range(k):
        results.append(quasi_greedy_batch(X=X, y=y, current_model=current_model,
                                     batch_size=batch_size, rng=rng, D=D, c=c, sample_first=True))

    return results[np.argmax([r[1] for r in results])]

kaggle_ninja.register("multiple_pick_best", multiple_pick_best)
kaggle_ninja.register("czarnecki", czarnecki)
kaggle_ninja.register("query_by_bagging", query_by_bagging)
kaggle_ninja.register("uncertainty_sampling", uncertainty_sampling)
kaggle_ninja.register("random_query", random_query)
kaggle_ninja.register("quasi_greedy_batch", quasi_greedy_batch)
kaggle_ninja.register("chen_krause", chen_krause)

