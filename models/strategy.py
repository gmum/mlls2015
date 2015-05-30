import numpy as np
from sklearn.ensemble import BaggingClassifier
import kaggle_ninja
from experiments.utils import jaccard_similarity_score_fast
from itertools import product
from misc.config import main_logger


def strategy(X, y, current_model, batch_size, seed):
    """
    :param X:
    :param y:
    :param current_model: Currently used model for making predictions
    :param batch_size:
    :param seed:
    :return: Indexes of picked examples
    """
    pass

def random_query(X, y, current_model, batch_size, seed):
    X = X[np.invert(y.known)]
    np.random.seed(seed)
    ids = np.random.randint(0, X.shape[0], size=batch_size)
    return y.unknown_ids[ids]


def uncertanity_sampling(X, y, current_model, batch_size, seed):
    X = X[np.invert(y.known)]
    if hasattr(current_model, "decision_function"):
        # Settles page 12
        ids =  np.argsort(np.abs(current_model.decision_function(X)))[:batch_size]
    elif hasattr(current_model, "predict_proba"):
        p = current_model.predict_proba(X)
        # Settles page 13
        ids =  np.argsort(np.sum(p * np.log(p), axis=1))[:batch_size]
    return y.unknown_ids[ids]


def query_by_bagging(X, y, current_model, batch_size, seed, base_model, n_bags, method):
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
        ids =  np.argsort(np.sum(pc * np.log(pc), axis=1))[:batch_size]
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
        ids = np.argsort(np.mean(np.sum(p * np.log(p / pc), axis=2), axis=0))[-batch_size:]

    return y.unknown_ids[ids]


def _uncertainty(X, model):
    assert hasattr(model, 'predict_proba'), "Model with probability prediction needs to be passed to this strategy!"
    p = model.predict_proba(X)
    # Settles page 13
    return np.sum(p * np.log(p), axis=1)


def _jaccard_dist(x1, x2):
    return 1 - jaccard_similarity_score_fast(x1, x2)


def quasi_greedy_batch(X, y, current_model, batch_size, seed,
                       c=0.3,
                       fitness_function=_uncertainty,
                       dist=_jaccard_dist):
    """
    :param c: Used for averaging (1-C)*example_fitness + C*normalized_distance_to_current_set
    :param sampling:
    :param dist:
    :return:
    """

    def score(idx):
        main_logger.debug("calculating score for example %i" % idx)
        A = picked
        A.append(idx)
        u_score = np.mean(sampling(X[A], model))
        d_score = np.mean([dist(X[j[0]], X[j[1]]) for j in product(A, A)])
        return (1 - c) * u_score + c * d_score

    if not any(y.known):
        picked = [np.argmax(sampling(X, model))]
    else:
        picked = np.where(y.known == True)[0].tolist()
    while len(picked) < len(y.known) + batch_size:
        unpicked_score = [score(i) for i in xrange(X.shape[0]) if i not in picked]
        picked.append(np.argmax(unpicked_score))
        main_logger.debug("quasi greedy batch is picking %i th example from %i" % (len(picked), len(y.known) + batch_size))

    main_logger.debug("quasi greedy batch picked %i examples from %i set" % (len(picked), len(y.unknown_ids)))

    return picked



kaggle_ninja.register("query_by_bagging", query_by_bagging)
kaggle_ninja.register("uncertanity_sampling", uncertanity_sampling)
kaggle_ninja.register("random_query", random_query)
kaggle_ninja.register("quasi_greedy_batch", quasi_greedy_batch)

